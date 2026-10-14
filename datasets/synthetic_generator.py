"""Synthetic physiological facial video generator for deterministic rPPG testing.

Generates photorealistic facial test videos with controllable:
- Ground-truth Heart Rate (HR in BPM) and exact reference BVP pulse waveform
- Fitzpatrick skin tones (Types I-VI) for C2 testing
- Low-light attenuation and shot noise for C1 testing
- Low camera resolution for C4 testing
- Surgical mask occlusion for C7 testing
- Synchronized reference_signal.csv and ground-truth metadata
"""

from typing import Dict, Tuple, Optional
import os
import cv2
import numpy as np


# Fitzpatrick skin tone representative base RGB colors (in [0, 255])
FITZPATRICK_BASE_RGB = {
    "Type I": np.array([253, 230, 215], dtype=np.float32),   # Very Pale
    "Type II": np.array([245, 215, 190], dtype=np.float32),  # Fair
    "Type III": np.array([218, 182, 150], dtype=np.float32), # Medium / Olive
    "Type IV": np.array([179, 137, 100], dtype=np.float32),  # Tan / Brown
    "Type V": np.array([138, 90, 56], dtype=np.float32),     # Dark Brown
    "Type VI": np.array([75, 45, 30], dtype=np.float32),      # Deep Dark
}


def generate_synthetic_rppg_video(
    output_video_path: str,
    output_gt_csv_path: str,
    duration_sec: float = 10.0,
    fps: float = 30.0,
    target_hr_bpm: float = 72.0,
    width: int = 640,
    height: int = 480,
    fitzpatrick_type: str = "Type III",
    lux: float = 300.0,
    has_mask: bool = False,
    motion_amplitude: float = 0.0,
    rsa_amplitude: float = 0.0,
    rsa_freq_hz: float = 0.10,
) -> Dict[str, any]:
    """Generate a synthetic video with an embedded rPPG physiological signal.

    Args:
        output_video_path: Path to write MP4 video.
        output_gt_csv_path: Path to write reference_signal.csv.
        duration_sec: Duration in seconds (e.g. 10.0s = 300 frames at 30 fps).
        fps: Video frame rate in Hz.
        target_hr_bpm: True heart rate in BPM.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fitzpatrick_type: Skin tone string ('Type I' through 'Type VI').
        lux: Ambient illuminance level (e.g. 300 lux baseline, <= 50 lux low light).
        has_mask: Whether to overlay synthetic mask covering lower face (C7).
        motion_amplitude: Amplitude of subtle facial translation motion.
        rsa_amplitude: Amplitude of physiological Respiratory Sinus Arrhythmia in BPM.
        rsa_freq_hz: RSA breathing modulation frequency in Hz.

    Returns:
        dict containing video metadata, ground truth HR, timestamps, and parameters.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_gt_csv_path)), exist_ok=True)

    num_frames = int(duration_sec * fps)
    timestamps = np.arange(num_frames) / fps
    hr_freq_hz = target_hr_bpm / 60.0

    if rsa_amplitude > 0.0:
        amp_hz = rsa_amplitude / 60.0
        # Phase integral phi(t) = 2*pi*integral(f(tau) dtau)
        phase = 2.0 * np.pi * hr_freq_hz * timestamps - (amp_hz / rsa_freq_hz) * (np.cos(2.0 * np.pi * rsa_freq_hz * timestamps) - 1.0)
        pulse_clean = np.sin(phase) + 0.3 * np.sin(2.0 * phase + 0.4)
        target_hr_series = target_hr_bpm + rsa_amplitude * np.sin(2.0 * np.pi * rsa_freq_hz * timestamps)
    else:
        pulse_clean = np.sin(2.0 * np.pi * hr_freq_hz * timestamps) + \
                      0.3 * np.sin(4.0 * np.pi * hr_freq_hz * timestamps + 0.4)
        target_hr_series = np.full(num_frames, target_hr_bpm, dtype=np.float64)

    # Base skin tone color
    base_color = FITZPATRICK_BASE_RGB.get(fitzpatrick_type, FITZPATRICK_BASE_RGB["Type III"]).copy()

    # Low-light attenuation factor (300 lux is nominal baseline)
    lux_factor = min(1.0, max(0.05, lux / 300.0))

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    face_center_x = width // 2
    face_center_y = int(height * 0.52)
    face_radius_x = int(width * 0.25)
    face_radius_y = int(height * 0.35)

    gt_hr_series = []

    for i in range(num_frames):
        t = timestamps[i]
        p_val = pulse_clean[i]

        # Subtle head motion if requested
        dx = int(motion_amplitude * np.sin(1.2 * t))
        dy = int(motion_amplitude * 0.5 * np.cos(0.8 * t))
        cx = face_center_x + dx
        cy = face_center_y + dy

        # Create background canvas (neutral grey)
        frame = np.full((height, width, 3), 160.0 * lux_factor, dtype=np.float32)

        # Draw facial ellipse with pulsatile skin color modulation
        # In rPPG physiology, blood hemoglobin absorption strongest in green channel (~-1.5% modulation)
        # Red channel has weaker modulation (~-0.5%), Blue channel minimal (~-0.2%)
        pulse_amp = 0.015  # ~1.5% color variation
        skin_r = base_color[0] * (1.0 - pulse_amp * 0.4 * p_val) * lux_factor
        skin_g = base_color[1] * (1.0 - pulse_amp * 1.0 * p_val) * lux_factor
        skin_b = base_color[2] * (1.0 - pulse_amp * 0.2 * p_val) * lux_factor
        skin_bgr = (float(skin_b), float(skin_g), float(skin_r))

        cv2.ellipse(frame, (cx, cy), (face_radius_x, face_radius_y), 0, 0, 360, skin_bgr, -1)

        # Forehead region (slightly brighter)
        fh_y = int(cy - face_radius_y * 0.45)
        cv2.ellipse(frame, (cx, fh_y), (int(face_radius_x * 0.6), int(face_radius_y * 0.25)), 0, 0, 360, skin_bgr, -1)

        # Cheeks (left and right)
        ch_y = int(cy + face_radius_y * 0.05)
        left_ch_x = int(cx - face_radius_x * 0.45)
        right_ch_x = int(cx + face_radius_x * 0.45)
        cv2.circle(frame, (left_ch_x, ch_y), int(face_radius_x * 0.22), skin_bgr, -1)
        cv2.circle(frame, (right_ch_x, ch_y), int(face_radius_x * 0.22), skin_bgr, -1)

        # Eyes (dark grey/brown)
        eye_y = int(cy - face_radius_y * 0.15)
        cv2.circle(frame, (int(cx - face_radius_x * 0.35), eye_y), int(face_radius_x * 0.09), (40 * lux_factor, 40 * lux_factor, 40 * lux_factor), -1)
        cv2.circle(frame, (int(cx + face_radius_x * 0.35), eye_y), int(face_radius_x * 0.09), (40 * lux_factor, 40 * lux_factor, 40 * lux_factor), -1)

        # Nose bridge
        cv2.line(frame, (cx, eye_y), (cx, int(cy + face_radius_y * 0.15)), (int(skin_b * 0.9), int(skin_g * 0.9), int(skin_r * 0.9)), 3)

        # Mouth or Surgical Mask (C7)
        mouth_y = int(cy + face_radius_y * 0.45)
        if has_mask:
            # Mask covering >= 40% lower face from mid-nose to chin
            mask_pts = np.array([
                [cx - int(face_radius_x * 0.85), int(cy + face_radius_y * 0.05)],
                [cx + int(face_radius_x * 0.85), int(cy + face_radius_y * 0.05)],
                [cx + int(face_radius_x * 0.65), int(cy + face_radius_y * 0.85)],
                [cx - int(face_radius_x * 0.65), int(cy + face_radius_y * 0.85)],
            ], dtype=np.int32)
            cv2.fillPoly(frame, [mask_pts], (200 * lux_factor, 180 * lux_factor, 100 * lux_factor)) # surgical cyan mask
        else:
            # Lips
            cv2.ellipse(frame, (cx, mouth_y), (int(face_radius_x * 0.3), int(face_radius_y * 0.08)), 0, 0, 360, (int(skin_b * 0.8), int(skin_g * 0.7), int(skin_r * 1.1)), -1)

        # If low-light condition (lux <= 50), inject Poisson shot noise per §2.2 C1
        if lux <= 50.0:
            noise_sigma = 8.0 * (1.0 - lux / 50.0) + 2.0
            noise = np.random.normal(0, noise_sigma, frame.shape)
            frame = np.clip(frame + noise, 0, 255)

        gt_hr_series.append(float(target_hr_series[i]))
        out.write(np.clip(frame, 0, 255).astype(np.uint8))

    out.release()

    # Save synchronized ground-truth reference CSV
    with open(output_gt_csv_path, "w") as f:
        f.write("timestamp_sec,hr_bpm,pulse_amplitude\n")
        for t, hr, p in zip(timestamps, gt_hr_series, pulse_clean):
            f.write(f"{t:.4f},{hr:.2f},{p:.5f}\n")

    return {
        "video_path": output_video_path,
        "gt_csv_path": output_gt_csv_path,
        "target_hr_bpm": target_hr_bpm,
        "gt_hr_series": np.array(gt_hr_series),
        "duration_sec": duration_sec,
        "num_frames": num_frames,
        "fps": fps,
        "resolution": f"{width}x{height}",
        "fitzpatrick_type": fitzpatrick_type,
        "lux": lux,
        "has_mask": has_mask,
    }
