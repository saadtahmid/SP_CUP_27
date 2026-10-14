"""Plane-Orthogonal-to-Skin (POS) rPPG Algorithm with frPPG In-Band Pre-Filtering.

Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
Algorithmic principles of remote PPG. IEEE TBME, 64(7), 1479-1491.
Enhanced with in-band pre-filtering prior to alpha estimation per Dawoodjee & Ghahramani (frPPG, IEEE Access 2025).
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
    """Extract BVP pulse signal from raw temporal RGB traces using POS algorithm."""
    rgb = np.asarray(rgb_signals, dtype=np.float64)
    t_len, num_ch = rgb.shape
    if num_ch != 3:
        raise ValueError(f"Expected rgb_signals shape (T, 3), got {rgb.shape}")

    win_len = max(45, int(round(window_sec * fs)))
    stride = max(8, int(round(stride_sec * fs)))

    if t_len < win_len:
        win_len = t_len
        stride = max(1, t_len)

    order = 2 if fs < 18.0 else 4
    nyq = 0.5 * fs
    b, a = signal.butter(order, [min(0.7 / nyq, 0.90), min(3.0 / nyq, 0.96)], btype="bandpass")

    h_total = np.zeros(t_len, dtype=np.float64)
    weight_total = np.zeros(t_len, dtype=np.float64)

    def _project_window(sub_rgb: np.ndarray) -> np.ndarray:
        means = np.mean(sub_rgb, axis=0, keepdims=True)
        means = np.where(means < 1e-6, 1e-6, means)
        c_n = (sub_rgb / means) - 1.0

        s1 = c_n[:, 1] - c_n[:, 2]
        s2 = c_n[:, 1] + c_n[:, 2] - 2.0 * c_n[:, 0]

        # Pre-filter s1 and s2 to cardiac band before computing alpha = std(s1)/std(s2) (frPPG, IEEE Access 2025)
        pad_w = min(len(s1) - 1, 3 * max(len(a), len(b)))
        if pad_w > 0 and len(s1) > 12:
            s1_f = signal.filtfilt(b, a, s1, padlen=pad_w)
            s2_f = signal.filtfilt(b, a, s2, padlen=pad_w)
        else:
            s1_f = signal.detrend(s1)
            s2_f = signal.detrend(s2)

        alpha = np.std(s1_f) / (np.std(s2_f) + 1e-12)
        return s1_f + alpha * s2_f

    start = 0
    while start + win_len <= t_len:
        end = start + win_len
        h = _project_window(rgb[start:end, :])
        win_weights = np.hanning(win_len)
        h_total[start:end] += (h - np.mean(h)) * win_weights
        weight_total[start:end] += win_weights
        start += stride

    if start < t_len and end < t_len:
        s_rem = max(0, t_len - win_len)
        sub_rgb = rgb[s_rem:t_len, :]
        h = _project_window(sub_rgb)
        win_weights = np.hanning(len(sub_rgb))
        h_total[s_rem:t_len] += (h - np.mean(h)) * win_weights
        weight_total[s_rem:t_len] += win_weights

    valid_mask = weight_total > 1e-6
    h_total[valid_mask] /= weight_total[valid_mask]

    padlen = min(len(h_total) - 1, 3 * max(len(a), len(b)))
    if padlen > 0 and len(h_total) > 8:
        bvp = signal.filtfilt(b, a, h_total, padlen=padlen)
    else:
        bvp = h_total
    return bvp

