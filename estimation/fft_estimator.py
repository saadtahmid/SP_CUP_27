"""Baseline Welch / FFT HR spectral estimator for ablation comparisons."""

from typing import Tuple, Union
import numpy as np
from scipy import signal


def estimate_hr_welch(
    sig: np.ndarray,
    fs: float = 30.0,
    low_hz: float = 0.7,
    high_hz: float = 3.0,
    return_spectrum: bool = False,
) -> Union[float, Tuple[float, np.ndarray, np.ndarray]]:
    """Estimate Heart Rate in BPM using Welch's Power Spectral Density method.

    Args:
        sig: 1D numpy array of temporal rPPG signal.
        fs: Sampling rate in Hz (default: 30.0).
        low_hz: Lower bound of HR band in Hz (0.7 Hz = 42 bpm).
        high_hz: Upper bound of HR band in Hz (3.0 Hz = 180 bpm).
        return_spectrum: Whether to return frequency and power arrays.

    Returns:
        hr_bpm: Dominant HR frequency in BPM.
        (optional) bpm_bins, power spectrum arrays.
    """
    x = np.asarray(sig, dtype=np.float64)
    n = len(x)
    if n < 4:
        return 72.0

    nperseg = min(n, 256)
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg, nfft=max(1024, nperseg * 4))

    # Mask to physiological band [low_hz, high_hz]
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    in_band_freqs = freqs[mask]
    in_band_psd = psd[mask]

    if len(in_band_psd) == 0:
        return 72.0

    peak_idx = int(np.argmax(in_band_psd))
    hr_bpm = float(in_band_freqs[peak_idx] * 60.0)

    if return_spectrum:
        return hr_bpm, in_band_freqs * 60.0, in_band_psd
    return hr_bpm
