"""Chrominance-based rPPG (CHROM).

De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG.
IEEE TBME, 60(10), 2878-2886.
"""

from typing import Optional
import numpy as np
from scipy import signal


def chrom_process(
    rgb_signals: np.ndarray,
    fs: float = 30.0,
    window_sec: float = 1.6,
    stride_sec: float = 0.8,
) -> np.ndarray:
    """Extract BVP pulse signal from raw temporal RGB traces using CHROM algorithm."""
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

    start = 0
    while start + win_len <= t_len:
        end = start + win_len
        sub_rgb = rgb[start:end, :]

        means = np.mean(sub_rgb, axis=0, keepdims=True)
        means = np.where(means < 1e-6, 1e-6, means)
        norm_rgb = sub_rgb / means

        xs = 3.0 * norm_rgb[:, 0] - 2.0 * norm_rgb[:, 1]
        ys = 1.5 * norm_rgb[:, 0] + norm_rgb[:, 1] - 1.5 * norm_rgb[:, 2]

        padlen = min(len(xs) - 1, 3 * max(len(a), len(b)))
        if padlen > 0 and len(xs) > 8:
            xf = signal.filtfilt(b, a, xs, padlen=padlen)
            yf = signal.filtfilt(b, a, ys, padlen=padlen)
        else:
            xf = xs - np.mean(xs)
            yf = ys - np.mean(ys)

        std_xf = np.std(xf)
        std_yf = np.std(yf)

        alpha = std_xf / (std_yf + 1e-12)
        s = xf - alpha * yf

        win_weights = np.hanning(win_len)
        h_total[start:end] += s * win_weights
        weight_total[start:end] += win_weights

        start += stride

    if start < t_len and end < t_len:
        s_rem = max(0, t_len - win_len)
        sub_rgb = rgb[s_rem:t_len, :]
        means = np.mean(sub_rgb, axis=0, keepdims=True)
        norm_rgb = sub_rgb / np.where(means < 1e-6, 1e-6, means)
        xs = 3.0 * norm_rgb[:, 0] - 2.0 * norm_rgb[:, 1]
        ys = 1.5 * norm_rgb[:, 0] + norm_rgb[:, 1] - 1.5 * norm_rgb[:, 2]
        padlen = min(len(xs) - 1, 3 * max(len(a), len(b)))
        if padlen > 0 and len(xs) > 8:
            xf = signal.filtfilt(b, a, xs, padlen=padlen)
            yf = signal.filtfilt(b, a, ys, padlen=padlen)
        else:
            xf = xs - np.mean(xs)
            yf = ys - np.mean(ys)
        s = xf - (np.std(xf) / (np.std(yf) + 1e-12)) * yf
        win_weights = np.hanning(len(sub_rgb))
        h_total[s_rem:t_len] += s * win_weights
        weight_total[s_rem:t_len] += win_weights

    valid_mask = weight_total > 1e-6
    h_total[valid_mask] /= weight_total[valid_mask]

    padlen = min(len(h_total) - 1, 3 * max(len(a), len(b)))
    if padlen > 0 and len(h_total) > 8:
        bvp = signal.filtfilt(b, a, h_total, padlen=padlen)
    else:
        bvp = h_total
    return bvp

