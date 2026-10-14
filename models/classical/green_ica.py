"""Green-channel rPPG and Independent Component Analysis (ICA) rPPG modules."""

from typing import Optional
import numpy as np
from scipy import signal


def green_process(
    rgb_signals: np.ndarray,
    fs: float = 30.0,
) -> np.ndarray:
    """Extract BVP signal from Green channel (Verkruysse et al., 2008).

    Args:
        rgb_signals: 2D numpy array of shape (T, 3).
        fs: Sampling frequency in Hz.

    Returns:
        bvp: 1D numpy array of shape (T,).
    """
    g = np.asarray(rgb_signals[:, 1], dtype=np.float64)
    # Detrend
    g_detrended = signal.detrend(g)
    # Bandpass filter Butterworth 0.7-3.0 Hz
    b, a = signal.butter(4, [0.7 / (fs / 2.0), 3.0 / (fs / 2.0)], btype="bandpass")
    return signal.filtfilt(b, a, g_detrended)


def ica_process(
    rgb_signals: np.ndarray,
    fs: float = 30.0,
) -> np.ndarray:
    """Extract BVP signal using FastICA blind source separation (Poh et al., 2010).

    Args:
        rgb_signals: 2D numpy array of shape (T, 3).
        fs: Sampling frequency in Hz.

    Returns:
        bvp: 1D numpy array of shape (T,).
    """
    from sklearn.decomposition import FastICA

    rgb = np.asarray(rgb_signals, dtype=np.float64)
    # Zero-mean unit-variance normalization
    norm_rgb = (rgb - np.mean(rgb, axis=0, keepdims=True)) / (np.std(rgb, axis=0, keepdims=True) + 1e-12)

    try:
        ica = FastICA(n_components=3, random_state=42, max_iter=500)
        sources = ica.fit_transform(norm_rgb)  # shape (T, 3)
    except Exception:
        # Fallback to PCA if ICA convergence fails
        u, s, vt = np.linalg.svd(norm_rgb, full_matrices=False)
        sources = u * s

    b, a = signal.butter(4, [0.7 / (fs / 2.0), 3.0 / (fs / 2.0)], btype="bandpass")

    # Select the component that exhibits the highest peak in physiological band 0.7-3.0 Hz
    best_comp = None
    best_peak_ratio = -1.0

    for i in range(sources.shape[1]):
        filtered = signal.filtfilt(b, a, sources[:, i])
        freqs, psd = signal.welch(filtered, fs=fs, nperseg=min(len(filtered), 256))
        band_mask = (freqs >= 0.7) & (freqs <= 3.0)
        if np.any(band_mask):
            peak_power = np.max(psd[band_mask])
            total_power = np.sum(psd) + 1e-12
            ratio = peak_power / total_power
            if ratio > best_peak_ratio:
                best_peak_ratio = ratio
                best_comp = filtered

    if best_comp is None:
        best_comp = green_process(rgb_signals, fs=fs)

    return best_comp
