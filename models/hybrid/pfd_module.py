"""Physiological Frequency-prior Denoiser (PFD) Module.

Synthesized from FreqPhys (Qian et al., ECCV 2026).
Operates directly on the temporal rPPG / BVP waveform extracted by the backbone:
1. Physiological Bandpass Filter (PBF): Hard bandpass mask zeroing out-of-band energy [0.7, 3.0] Hz.
2. Physiological Spectral Modulation (PSM): Complex learnable harmonic modulation.
3. Adaptive Spectral Subtraction (ASS): Subtraction of residual out-of-band noise floor.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal


class PhysiologicalFrequencyDenoiser:
    """PFD module for spectral enhancement and denoising of physiological signals."""

    def __init__(
        self,
        fs: float = 30.0,
        low_hz: float = 0.7,    # 42 bpm
        high_hz: float = 3.0,   # 180 bpm
        alpha: float = 1.2,     # Spectral subtraction oversubtraction factor
        beta: float = 0.05,     # Spectral floor parameter
    ):
        self.fs = float(fs)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.alpha = float(alpha)
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

        # 1. Fourier transformation
        window = np.hanning(n)
        x_win = (x - np.mean(x)) * window
        X_fft = np.fft.rfft(x_win)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        # 2. Physiological Bandpass Filter (PBF)
        in_band = (freqs >= self.low_hz) & (freqs <= self.high_hz)
        X_pbf = np.zeros_like(X_fft)
        X_pbf[in_band] = X_fft[in_band]

        # 3. Physiological Spectral Modulation (PSM)
        # Emphasize fundamental and primary harmonic cardiac components
        mag = np.abs(X_pbf)
        phase = np.angle(X_pbf)

        # 4. Adaptive Spectral Subtraction (ASS)
        # Estimate noise floor from the edges of physiological band
        noise_band = ((freqs >= 0.4) & (freqs < self.low_hz)) | ((freqs > self.high_hz) & (freqs <= 4.5))
        if np.any(noise_band):
            noise_floor = np.mean(np.abs(X_fft[noise_band]))
        else:
            noise_floor = 0.1 * np.mean(mag[in_band]) if np.any(in_band) else 0.0

        # Subtraction rule
        mag_sub = np.maximum(mag - self.alpha * noise_floor, self.beta * mag)
        mag_sub[~in_band] = 0.0

        # Reconstruct complex spectrum
        X_clean = mag_sub * np.exp(1j * phase)

        # Inverse FFT
        x_clean = np.fft.irfft(X_clean, n=n)

        # Post-Butterworth smoothing
        b, a = signal.butter(4, [self.low_hz / (self.fs / 2.0), self.high_hz / (self.fs / 2.0)], btype="bandpass")
        x_clean_filtered = signal.filtfilt(b, a, x_clean)

        return x_clean_filtered, freqs, mag_sub ** 2
