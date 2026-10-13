"""Constraint C4 (Low Resolution) and Constraint C5 (Frame-Rate Decimation).

C4: Downsamples HD video (1280x720) to VGA (640x480) and QVGA (320x240) using bicubic interpolation.
C5: Decimates video frame rate from >=30 fps down to 10 fps (stress-testing near the 6 fps Nyquist floor).
"""

from typing import Tuple, List
import cv2
import numpy as np


def apply_low_resolution_c4(
    frame: np.ndarray,
    target_resolution: str = "VGA",  # 'VGA' (640x480), 'QVGA' (320x240)
) -> np.ndarray:
    """Downsample a frame to VGA or QVGA using bicubic interpolation (MATLAB imresize).

    Args:
        frame: Image array of shape (H, W, 3).
        target_resolution: 'VGA' (640x480) or 'QVGA' (320x240).

    Returns:
        resized: Downsampled frame.
    """
    if target_resolution.upper() == "QVGA":
        target_w, target_h = 320, 240
    elif target_resolution.upper() == "VGA":
        target_w, target_h = 640, 480
    else:
        target_w, target_h = 640, 480

    return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def simulate_low_res_video(
    frames: np.ndarray,
    target_resolution: str = "VGA",
) -> np.ndarray:
    """Downsample an entire video sequence to VGA or QVGA."""
    h_orig, w_orig = frames.shape[1], frames.shape[2]
    if target_resolution.upper() == "QVGA":
        target_w, target_h = 320, 240
    else:
        target_w, target_h = 640, 480

    t_len = len(frames)
    out_frames = np.empty((t_len, target_h, target_w, 3), dtype=np.uint8)
    for i in range(t_len):
        out_frames[i] = cv2.resize(frames[i], (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return out_frames


def apply_frame_decimation_c5(
    frames: np.ndarray,
    original_fps: float = 30.0,
    target_fps: float = 10.0,
) -> Tuple[np.ndarray, float]:
    """Sub-sample baseline video frames to target fps (e.g. 10 fps).

    Args:
        frames: Array of frames of shape (T, H, W, 3).
        original_fps: Original sampling rate in Hz (e.g. 30.0).
        target_fps: Downsampled sampling rate in Hz (e.g. 10.0).

    Returns:
        decimated_frames: Sub-sampled frames array.
        effective_fps: Actual effective frame rate.
    """
    step = int(round(original_fps / target_fps))
    if step < 1:
        step = 1

    decimated = frames[::step]
    effective_fps = original_fps / step
    return decimated, effective_fps
