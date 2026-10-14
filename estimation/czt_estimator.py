"""Chirp-Z Transform (CZT) Zoomed Spectral HR Estimator.

Replaces standard coarse FFT with high-resolution Chirp-Z Transform, concentrating
all spectral bins exclusively within the physiological heart rate band:
[0.7 Hz, 3.0 Hz] corresponding to [42 bpm, 180 bpm].
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

        self.freq_bins = np.linspace(self.low_hz, self.high_hz, self.num_bins)
        self.bpm_bins = self.freq_bins * 60.0

    def compute_spectrum(self, sig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the zoomed power spectrum of a 1D physiological signal."""
        x = np.asarray(sig, dtype=np.float64)
        n = len(x)
        if n < 4:
            return self.bpm_bins, np.zeros(self.num_bins)

        x_detrended = signal.detrend(x)
        window = np.hanning(n)
        x_windowed = x_detrended * window

        w_step = (self.high_hz - self.low_hz) / (self.num_bins - 1)
        a = np.exp(1j * 2.0 * np.pi * self.low_hz / self.fs)
        w = np.exp(-1j * 2.0 * np.pi * w_step / self.fs)

        czt_op = signal.CZT(n, m=self.num_bins, w=w, a=a)
        czt_complex = czt_op(x_windowed)

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
        """Estimate Heart Rate in Beats Per Minute (BPM) from a 1D rPPG signal."""
        bpm_bins, power = self.compute_spectrum(sig)

        peak_idx = int(np.argmax(power))
        hr_bpm = float(bpm_bins[peak_idx])

        if 0 < peak_idx < len(power) - 1:
            alpha = power[peak_idx - 1]
            beta = power[peak_idx]
            gamma = power[peak_idx + 1]
            denom = alpha - 2.0 * beta + gamma
            if abs(denom) > 1e-12:
                delta = 0.5 * (alpha - gamma) / denom
                dbpm = (self.high_hz - self.low_hz) / (self.num_bins - 1) * 60.0
                hr_bpm = float(bpm_bins[peak_idx] + delta * dbpm)

        hr_bpm = max(self.low_hz * 60.0, min(self.high_hz * 60.0, hr_bpm))

        if return_spectrum:
            return hr_bpm, bpm_bins, power
        return hr_bpm

    def compute_snr_db(
        self,
        power: np.ndarray,
        freq_axis: np.ndarray,
        hr_bpm: float,
        half_bandwidth_bpm: float = 6.0,
    ) -> float:
        """Compute physiological SNR in dB (de Haan & Jeanne 2013)."""
        axis = np.asarray(freq_axis)
        if np.max(axis) < 15.0:
            axis_bpm = axis * 60.0
        else:
            axis_bpm = axis

        f0 = hr_bpm
        f1 = 2.0 * hr_bpm

        sig_mask = (np.abs(axis_bpm - f0) <= half_bandwidth_bpm)
        if f1 <= self.high_hz * 60.0:
            sig_mask = sig_mask | (np.abs(axis_bpm - f1) <= half_bandwidth_bpm)

        sig_power = float(np.sum(power[sig_mask]))
        noise_power = float(np.sum(power[~sig_mask]))

        if noise_power <= 1e-12:
            return 20.0
        if sig_power <= 1e-12:
            return -20.0

        snr = 10.0 * np.log10(sig_power / noise_power)
        return float(np.clip(snr, -20.0, 30.0))


def estimate_hr_czt(sig: np.ndarray, fs: float = 30.0) -> float:
    """Convenience functional interface for CZT HR estimation."""
    estimator = CZTHRTransformer(fs=fs)
    return estimator.estimate_hr(sig)
