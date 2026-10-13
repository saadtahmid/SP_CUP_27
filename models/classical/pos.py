"""Plane-Orthogonal-to-Skin (POS) rPPG Algorithm.

Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
Algorithmic principles of remote PPG. IEEE TBME, 64(7), 1479-1491.
"""

from typing import Optional
import numpy as np
from scipy import signal


def pos_process(
    rgb_signals: np.ndarray,
    fs: float = 30.0,
    window_sec: float = 1.6,
    stride_sec: float = 0.8,
) -> np.ndarray:
    """Extract BVP pulse signal from raw temporal RGB traces using POS algorithm.

    Args:
        rgb_signals: 2D numpy array of shape (T, 3) representing temporal RGB mean traces.
        fs: Sampling frequency in Hz (default: 30.0).
        window_sec: Processing window length in seconds (standard: 1.6s).
        stride_sec: Stride length between windows in seconds (standard: 0.8s).

    Returns:
        bvp: 1D numpy array of shape (T,) containing reconstructed BVP pulse signal.
    """
    rgb = np.asarray(rgb_signals, dtype=np.float64)
    t_len, num_ch = rgb.shape
    if num_ch != 3:
        raise ValueError(f"Expected rgb_signals shape (T, 3), got {rgb.shape}")

    win_len = int(window_sec * fs)
    stride = int(stride_sec * fs)

    if t_len < win_len:
        win_len = t_len
        stride = t_len

    b, a = signal.butter(4, [0.7 / (fs / 2.0), 3.0 / (fs / 2.0)], btype="bandpass")

    h_total = np.zeros(t_len, dtype=np.float64)
    weight_total = np.zeros(t_len, dtype=np.float64)

    start = 0
    while start + win_len <= t_len:
        end = start + win_len
        sub_rgb = rgb[start:end, :]

        # Normalize each windowed segment by temporal channel mean
        means = np.mean(sub_rgb, axis=0, keepdims=True)
        means = np.where(means < 1e-6, 1e-6, means)
        c_n = sub_rgb / means

        # POS projection matrix
        # S1 = G - B
        # S2 = G + B - 2*R
        s1 = c_n[:, 1] - c_n[:, 2]
        s2 = c_n[:, 1] + c_n[:, 2] - 2.0 * c_n[:, 0]

        std_s1 = np.std(s1)
        std_s2 = np.std(s2)

        alpha = std_s1 / (std_s2 + 1e-12)
        h = s1 + alpha * s2

        win_weights = np.hanning(win_len)
        h_total[start:end] += (h - np.mean(h)) * win_weights
        weight_total[start:end] += win_weights

        start += stride

    valid_mask = weight_total > 1e-6
    h_total[valid_mask] /= weight_total[valid_mask]

    # Bandpass filter cleanup to competition band 0.7-3.0 Hz
    padlen = min(len(h_total) - 1, 3 * max(len(a), len(b)))
    if padlen > 0 and len(h_total) > 8:
        bvp = signal.filtfilt(b, a, h_total, padlen=padlen)
    else:
        bvp = h_total
    return bvp
