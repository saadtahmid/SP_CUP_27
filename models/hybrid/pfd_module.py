"""Physiological Frequency-prior Denoiser (PFD) Module.

Synthesized from FreqPhys (Qian et al., ECCV 2026) and verified in Stage 3B/3C/4:
1. Physiological Bandpass Filter (PBF): Hard bandpass mask isolating cardiac band [0.7, 3.0] Hz.
2. Fundamental-Dominant Spectral Modulation (PSM): Reinforces f_i by its 2nd harmonic 2*f_i
   ONLY when M(f_i) > 1.25 * M(2*f_i) and M(f_i) >= 0.50 * max_M, preventing sub-harmonic inversion.
3. Adaptive Spectrum Selection (ASS): Attenuates weak noise floor bins (< 0.25 * median) while
   preserving a uniform time-domain reconstruction envelope.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    _HAS_TORCH = False


class PhysiologicalFrequencyDenoiser:
    """Zero-shot harmonic-prior spectral denoiser implementing FreqPhys (ECCV 2026)."""

    def __init__(
        self,
        fs: float = 30.0,
        low_hz: float = 0.7,        # 42 bpm
        high_hz: float = 3.0,       # 180 bpm
        gamma_harm: float = 0.5,    # Harmonic reinforcement weight (PSM)
        gamma_sub: float = 0.5,     # Fundamental boost for dicrotic sub-harmonics
        alpha_ass: float = 0.25,    # Adaptive threshold scale (ASS)
        beta: float = 0.20,         # Residual floor attenuation
    ):
        self.fs = float(fs)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.gamma_harm = float(gamma_harm)
        self.gamma_sub = float(gamma_sub)
        self.alpha_ass = float(alpha_ass)
        self.beta = float(beta)

    def denoise_spectrum(self, bvp_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply Stage 3B/3C verified PFD (PBF + Fundamental-Verified PSM + ASS) to a BVP signal."""
        x = np.asarray(bvp_signal, dtype=np.float64)
        n = len(x)
        if n < 8:
            return x, np.array([]), np.array([])

        x_det = signal.detrend(x)
        X_win = np.fft.rfft(x_det * np.hanning(n))
        X_raw = np.fft.rfft(x_det)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        in_band = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        mag = np.abs(X_win)
        max_m = max(float(np.max(mag[in_band])) if np.any(in_band) else 1.0, 1e-12)

        G = np.zeros_like(freqs, dtype=np.float64)
        G[in_band] = 1.0

        for i in np.where(in_band)[0]:
            f_i = freqs[i]
            m1 = float(mag[i])
            m2 = float(np.interp(2.0 * f_i, freqs, mag)) if 2.0 * f_i <= freqs[-1] else 0.0
            # Only reinforce f_i if f_i is already a dominant fundamental (prevents f_HR/2 sub-harmonic inversion)
            if m1 > 1.25 * m2 and m1 >= 0.50 * max_m:
                G[i] = 1.0 + self.gamma_harm * (m2 / max_m)

        eb = mag[in_band] * G[in_band]
        tau = self.alpha_ass * float(np.median(eb)) if len(eb) > 0 else 0.0
        for i in np.where(in_band)[0]:
            if mag[i] * G[i] < tau:
                G[i] *= self.beta

        x_clean = np.fft.irfft(X_raw * G, n=n)
        psd_clean = np.abs(X_win * G) ** 2

        return x_clean, freqs, psd_clean


if _HAS_TORCH:
    class PFDBlock(nn.Module):
        """Trainable Physiological Frequency-prior Denoiser block (FreqPhys Eq. 4, 7-9)."""

        def __init__(self, in_features: int = 1, out_features: int = 1):
            super().__init__()
            self.W_re = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
            self.W_im = nn.Parameter(torch.randn(in_features, out_features) * 0.02)
            self.B_re = nn.Parameter(torch.zeros(out_features))
            self.B_im = nn.Parameter(torch.zeros(out_features))
            self.tau = nn.Parameter(torch.tensor(0.5))

        def forward(self, Z_re: torch.Tensor, Z_im: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass for complex spectrum modulation and threshold selection."""
            M_re = F.relu(torch.matmul(Z_re, self.W_re) - torch.matmul(Z_im, self.W_im) + self.B_re)
            M_im = F.relu(torch.matmul(Z_re, self.W_im) + torch.matmul(Z_im, self.W_re) + self.B_im)

            Z_tilde_re = M_re * Z_re - M_im * Z_im
            Z_tilde_im = M_re * Z_im + M_im * Z_re

            E = torch.sqrt(Z_tilde_re ** 2 + Z_tilde_im ** 2 + 1e-8)
            soft_m = torch.sigmoid(5.0 * (E - self.tau))
            hard_m = (E >= self.tau).float()
            mask = hard_m - soft_m.detach() + soft_m

            scale = mask + (1.0 - mask) * 0.05 * (E / (torch.relu(self.tau) + 1e-6))
            return Z_tilde_re * scale, Z_tilde_im * scale
else:
    class PFDBlock:  # type: ignore
        pass
