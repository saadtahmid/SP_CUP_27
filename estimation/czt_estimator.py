"""Chirp-Z Transform (CZT) Zoomed Spectral HR Estimator.

Replaces standard coarse FFT with high-resolution Chirp-Z Transform, concentrating
all spectral bins exclusively within the physiological heart rate band:
[0.7 Hz, 3.0 Hz] corresponding to [42 bpm, 180 bpm].

Ablation evidence (BeatFormer, ICCVW 2025):
CZT improves rPPG MAE by ~5 bpm over standard FFT on motion/uncontrolled benchmarks
by providing ~13x higher spectral resolution without zero-padding distortion.
"""

from typing import Tuple, Optional, Union
import numpy as np
from scipy import signal


class CZTHRTransformer:
    """High-resolution Chirp-Z Transform Heart Rate Estimator."""

    def __init__(
        self,
        fs: float = 30.0,
        low_hz: float = 0.7,    # 42 bpm (lower bound of competition band)
        high_hz: float = 3.0,   # 180 bpm (upper bound of competition band)
        num_bins: int = 512,    # high-resolution zoom bins
    ):
        self.fs = float(fs)
        self.low_hz = float(low_hz)
        self.high_hz = float(high_hz)
        self.num_bins = int(num_bins)

        # Precompute frequencies corresponding to each CZT bin
        self.freq_bins = np.linspace(self.low_hz, self.high_hz, self.num_bins)
        self.bpm_bins = self.freq_bins * 60.0

    def compute_spectrum(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the zoomed power spectrum of a 1D physiological signal.

        Args:
            sig: 1D numpy array of temporal rPPG / BVP values.

        Returns:
            bpm_bins: 1D array of BPM frequencies in [42, 180].
            power: 1D array of normalized power spectral density values.
        """
        x = np.asarray(sig, dtype=np.float64)
        n = len(x)
        if n < 4:
            return self.bpm_bins, np.zeros(self.num_bins)

        # Detrend and apply Hanning window to suppress boundary discontinuities
        x_detrended = signal.detrend(x)
        window = np.hanning(n)
        x_windowed = x_detrended * window

        # Configure CZT on the unit circle
        # A = exp(j * 2*pi * f_low / fs)
        # W = exp(-j * 2*pi * (f_high - f_low) / ((M - 1) * fs))
        w_step = (self.high_hz - self.low_hz) / (self.num_bins - 1)
        a = np.exp(1j * 2.0 * np.pi * self.low_hz / self.fs)
        w = np.exp(-1j * 2.0 * np.pi * w_step / self.fs)

        czt_op = signal.CZT(n, m=self.num_bins, w=w, a=a)
        czt_complex = czt_op(x_windowed)

        # Power spectrum
        power = np.abs(czt_complex) ** 2
        max_p = np.max(power)
        if max_p > 1e-12:
            power /= max_p

        return self.bpm_bins, power

    def estimate_hr(
        self,
        sig: np.ndarray,
        return_spectrum: bool = False,
    ) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
        """Estimate Heart Rate in Beats Per Minute (BPM) from a 1D rPPG signal.

        Args:
            sig: 1D numpy array of temporal rPPG signal.
            return_spectrum: Whether to also return the (bpm_bins, power) arrays.

        Returns:
            hr_bpm: Dominant HR frequency in BPM.
            (optional) bpm_bins, power spectrum arrays.
        """
        bpm_bins, power = self.compute_spectrum(sig)

        # Identify dominant physiological peak
        peak_idx = int(np.argmax(power))
        hr_bpm = float(bpm_bins[peak_idx])

        # Parabolic peak interpolation for sub-bin precision
        if 0 < peak_idx < len(power) - 1:
            alpha = power[peak_idx - 1]
            beta = power[peak_idx]
            gamma = power[peak_idx + 1]
            denom = alpha - 2.0 * beta + gamma
            if abs(denom) > 1e-12:
                delta = 0.5 * (alpha - gamma) / denom
                dbpm = (self.high_hz - self.low_hz) / (self.num_bins - 1) * 60.0
                hr_bpm = float(bpm_bins[peak_idx] + delta * dbpm)

        # Clamp to competition physiological range [42, 180]
        hr_bpm = max(self.low_hz * 60.0, min(self.high_hz * 60.0, hr_bpm))

        if return_spectrum:
            return hr_bpm, bpm_bins, power
        return hr_bpm


def estimate_hr_czt(sig: np.ndarray, fs: float = 30.0) -> float:
    """Convenience functional interface for CZT HR estimation."""
    estimator = CZTHRTransformer(fs=fs)
    return estimator.estimate_hr(sig)
