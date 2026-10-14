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
    """Extract BVP pulse signal from raw temporal RGB traces using CHROM algorithm.

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
        # Fallback to single window
        win_len = t_len
        stride = t_len

    # Bandpass filter Butterworth 0.7 - 3.0 Hz
    b, a = signal.butter(4, [0.7 / (fs / 2.0), 3.0 / (fs / 2.0)], btype="bandpass")

    h_total = np.zeros(t_len, dtype=np.float64)
    weight_total = np.zeros(t_len, dtype=np.float64)

    start = 0
    while start + win_len <= t_len:
        end = start + win_len
        sub_rgb = rgb[start:end, :]

        # Temporal zero-mean unit-variance normalization per channel in window
        means = np.mean(sub_rgb, axis=0, keepdims=True)
        means = np.where(means < 1e-6, 1e-6, means)
        norm_rgb = sub_rgb / means

        # Build two orthogonal chrominance signals
        xs = 3.0 * norm_rgb[:, 0] - 2.0 * norm_rgb[:, 1]
        ys = 1.5 * norm_rgb[:, 0] + norm_rgb[:, 1] - 1.5 * norm_rgb[:, 2]

        # Apply bandpass filter
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

        # Hanning window weighting for smooth overlap-add
        win_weights = np.hanning(win_len)
        h_total[start:end] += s * win_weights
        weight_total[start:end] += win_weights

        start += stride

    # Handle final remainder window if needed
    if start < t_len and np.max(weight_total[start:]) < 1e-6:
        sub_rgb = rgb[start:, :]
        rem_len = len(sub_rgb)
        if rem_len > 8:
            means = np.mean(sub_rgb, axis=0, keepdims=True)
            norm_rgb = sub_rgb / np.where(means < 1e-6, 1e-6, means)
            xs = 3.0 * norm_rgb[:, 0] - 2.0 * norm_rgb[:, 1]
            ys = 1.5 * norm_rgb[:, 0] + norm_rgb[:, 1] - 1.5 * norm_rgb[:, 2]
            b_rem, a_rem = signal.butter(2, [0.7 / (fs / 2.0), 3.0 / (fs / 2.0)], btype="bandpass")
            xf = signal.filtfilt(b_rem, a_rem, xs)
            yf = signal.filtfilt(b_rem, a_rem, ys)
            s = xf - (np.std(xf) / (np.std(yf) + 1e-12)) * yf
            h_total[start:] += s
            weight_total[start:] += 1.0

    valid_mask = weight_total > 1e-6
    h_total[valid_mask] /= weight_total[valid_mask]

    # Final bandpass cleanup
    bvp = signal.filtfilt(b, a, h_total)
    return bvp
