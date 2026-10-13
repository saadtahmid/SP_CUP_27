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


def apply_bitrate_compression_c5(
    input_video_path: str,
    output_video_path: str,
    target_kbps: int = 400,
) -> str:
    """Re-encode video with constrained target bitrate (<= 500 kbps, MPEG-4).

    Complies with IEEE SP Cup Section 2.2 C5:
    'Re-encode the video using VideoWriter with the MPEG-4 profile and a constrained target bitrate <= 500 kbps'

    Args:
        input_video_path: Path to source video.
        output_video_path: Destination path for compressed video.
        target_kbps: Target bitrate in kbps (default 400 kbps).

    Returns:
        output_video_path: Path to resulting encoded video.
    """
    import subprocess
    import os
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", input_video_path,
        "-c:v", "mpeg4",
        "-b:v", f"{int(target_kbps)}k",
        "-maxrate", f"{int(target_kbps)}k",
        "-bufsize", f"{int(target_kbps * 2)}k",
        output_video_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return output_video_path


def simulate_bitrate_compression_frames(
    frames: np.ndarray,
    target_kbps: int = 400,
    fps: float = 30.0,
) -> np.ndarray:
    """In-memory simulation of bitrate-constrained MPEG-4 compression on a frame array.

    Args:
        frames: Array of frames of shape (T, H, W, 3) in uint8 BGR.
        target_kbps: Target bitrate in kbps.
        fps: Frame rate.

    Returns:
        compressed_frames: Array of compressed frames after encode-decode cycle.
    """
    import tempfile
    import os
    t_len, h, w, _ = frames.shape
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_mp4 = os.path.join(tmpdir, "raw.mp4")
        comp_mp4 = os.path.join(tmpdir, "compressed.mp4")

        # Write uncompressed frames
        writer = cv2.VideoWriter(raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

        # Apply bitrate compression via ffmpeg
        apply_bitrate_compression_c5(raw_mp4, comp_mp4, target_kbps=target_kbps)

        # Read back compressed frames
        cap = cv2.VideoCapture(comp_mp4)
        comp_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            comp_frames.append(frame)
        cap.release()

        if len(comp_frames) == t_len:
            return np.array(comp_frames, dtype=np.uint8)
        elif len(comp_frames) > 0:
            # Interpolate or pad/truncate to match length
            res = np.zeros_like(frames)
            n_read = min(len(comp_frames), t_len)
            res[:n_read] = np.array(comp_frames[:n_read], dtype=np.uint8)
            if n_read < t_len:
                res[n_read:] = comp_frames[-1]
            return res
        return frames
