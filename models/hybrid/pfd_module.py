"""Physiological Frequency-prior Denoiser (PFD) Module.

Synthesized from FreqPhys (Qian et al., ECCV 2026).
Operates directly on the temporal rPPG / BVP waveform extracted by the backbone:
1. Physiological Bandpass Filter (PBF): Hard bandpass mask isolating cardiac band [0.7, 3.0] Hz.
2. Physiological Spectral Modulation (PSM): Harmonic-consistent complex spectrum modulation.
3. Adaptive Spectrum Selection (ASS): Data-driven adaptive noise suppression (Eqs. 7-9).
Includes both a training-free NumPy denoiser and a trainable PyTorch PFDBlock (nn.Module).
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysiologicalFrequencyDenoiser:
    """Zero-shot harmonic-prior spectral denoiser implementing FreqPhys (ECCV 2026)."""

    def __init__(
        self,
        fs: float = 30.0,
        low_hz: float = 0.7,        # 42 bpm
        high_hz: float = 3.0,       # 180 bpm
        gamma_harm: float = 0.6,    # Harmonic reinforcement weight (PSM)
        gamma_sub: float = 0.5,     # Fundamental boost for dicrotic sub-harmonics
        alpha_ass: float = 0.5,     # Adaptive threshold scale (ASS)
        beta: float = 0.05,         # Residual floor attenuation
    ):
        self.fs = float(fs)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.gamma_harm = float(gamma_harm)
        self.gamma_sub = float(gamma_sub)
        self.alpha_ass = float(alpha_ass)
        self.beta = float(beta)

    def denoise_spectrum(self, bvp_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply PFD (PBF + PSM + ASS) to an extracted BVP signal.

        Args:
            bvp_signal: 1D numpy array of raw BVP signal from backbone.

        Returns:
            cleaned_bvp: 1D time-domain cleaned signal.
            freqs: frequency axis in Hz.
            cleaned_psd: cleaned power spectral density.
        """
        x = np.asarray(bvp_signal, dtype=np.float64)
        n = len(x)
        if n < 8:
            return x, np.array([]), np.array([])

        # 1. Fourier transformation with Hanning window
        window = np.hanning(n)
        x_win = (x - np.mean(x)) * window
        X_fft = np.fft.rfft(x_win)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        # 2. Physiological Bandpass Filter (PBF, FreqPhys Eqs. 1-2)
        in_band = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        X_pbf = np.zeros_like(X_fft)
        X_pbf[in_band] = X_fft[in_band]

        mag_full = np.abs(X_fft)
        max_inband = float(np.max(mag_full[in_band])) if np.any(in_band) else 1.0
        max_inband = max(max_inband, 1e-12)

        # 3. Physiological Spectrum Modulation (PSM, FreqPhys Eqs. 3-5)
        # Interpolate 2x harmonic amplitude for bins in cardiac band [0.7, 3.0] Hz
        G = np.ones(len(freqs), dtype=np.float64)
        for i in np.where(in_band)[0]:
            f_i = freqs[i]
            f_2i = 2.0 * f_i
            # Interpolate magnitude at 2f_i
            if f_2i <= freqs[-1]:
                m2 = float(np.interp(f_2i, freqs, mag_full))
            else:
                m2 = 0.0
            G[i] = 1.0 + self.gamma_harm * (m2 / max_inband)

        # Sub-harmonic / dicrotic notch check:
        # If candidate peak > 1.8 Hz has a strong sub-harmonic at f/2, boost fundamental
        if np.any(in_band):
            inband_indices = np.where(in_band)[0]
            peak_idx = inband_indices[np.argmax(mag_full[in_band])]
            peak_f = freqs[peak_idx]
            if peak_f > 1.8:
                half_f = peak_f / 2.0
                if half_f >= self.low_hz:
                    half_mag = float(np.interp(half_f, freqs, mag_full))
                    if half_mag >= 0.65 * mag_full[peak_idx]:
                        # Boost fundamental bin
                        half_idx = int(np.argmin(np.abs(freqs - half_f)))
                        G[half_idx] *= (1.0 + self.gamma_sub)

        # Apply complex modulation
        X_psm = X_pbf * G

        # 4. Adaptive Spectrum Selection (ASS, FreqPhys Eqs. 7-9)
        E_band = np.abs(X_psm[in_band])
        if len(E_band) > 0:
            med_e = float(np.median(E_band))
            mad_e = float(np.median(np.abs(E_band - med_e)))
            tau = med_e + self.alpha_ass * mad_e
        else:
            tau = 0.0

        # Apply soft/hard selection mask
        X_ass = np.zeros_like(X_psm)
        for i in np.where(in_band)[0]:
            e_i = np.abs(X_psm[i])
            if e_i >= tau:
                X_ass[i] = X_psm[i]
            else:
                atten = self.beta * (e_i / (tau + 1e-12))
                X_ass[i] = X_psm[i] * atten

        # 5. Inverse FFT and zero-phase Butterworth bandpass filter
        x_clean = np.fft.irfft(X_ass, n=n)

        nyq = 0.5 * self.fs
        b, a = signal.butter(
            4,
            [min(self.low_hz / nyq, 0.98), min(self.high_hz / nyq, 0.99)],
            btype="bandpass",
        )
        x_clean_filtered = signal.filtfilt(b, a, x_clean)
        psd_clean = np.abs(X_ass) ** 2

        return x_clean_filtered, freqs, psd_clean


class PFDBlock(nn.Module):
    """Trainable Physiological Frequency-prior Denoiser block (FreqPhys Eq. 4, 7-9).

    Differentiable complex linear modulation layer and adaptive spectral selection
    using a Straight-Through Estimator (STE) for end-to-end training.
    """

    def __init__(self, in_features: int = 1, out_features: int = 1):
        super().__init__()
        self.W_re = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.W_im = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
        self.B_re = nn.Parameter(torch.zeros(out_features))
        self.B_im = nn.Parameter(torch.zeros(out_features))
        self.tau = nn.Parameter(torch.tensor(0.5))

    def forward(self, Z_re: torch.Tensor, Z_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for complex spectrum modulation and threshold selection.

        Args:
            Z_re: Real part of spectral representation (B, F, in_features).
            Z_im: Imaginary part of spectral representation (B, F, in_features).

        Returns:
            Z_hat_re, Z_hat_im: Denoised real and imaginary spectral tensors.
        """
        # Complex linear transformation (FreqPhys Eq. 4)
        M_re = F.relu(torch.matmul(Z_re, self.W_re) - torch.matmul(Z_im, self.W_im) + self.B_re)
        M_im = F.relu(torch.matmul(Z_re, self.W_im) + torch.matmul(Z_im, self.W_re) + self.B_im)

        Z_tilde_re = M_re * Z_re - M_im * Z_im
        Z_tilde_im = M_re * Z_im + M_im * Z_re

        # Adaptive Spectrum Selection with Straight-Through Estimator (Eqs. 7-9)
        E = torch.sqrt(Z_tilde_re ** 2 + Z_tilde_im ** 2 + 1e-8)
        soft_m = torch.sigmoid(5.0 * (E - self.tau))
        hard_m = (E >= self.tau).float()
        mask = hard_m - soft_m.detach() + soft_m

        # Soft attenuation for sub-threshold bins
        scale = mask + (1.0 - mask) * 0.05 * (E / (torch.relu(self.tau) + 1e-6))
        return Z_tilde_re * scale, Z_tilde_im * scale
