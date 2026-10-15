"""Dataset Loader for MMPD (Multi-Domain Mobile Video Physiology Dataset).

Reference:
Tang, J., Wang, K., Fan, Y., Ji, J., Shi, Y., Wang, Y. (2023).
MMPD: Multi-Domain Mobile Video Physiology Dataset.
IEEE Transactions on Biomedical Engineering (TBME).

Dataset Characteristics:
- 33 subjects (mobile smartphone capture: Samsung Galaxy S22)
- Multi-domain conditions:
    * Fitzpatrick Skin Types: III, IV, V, VI (demographic fairness)
    * Lighting conditions: LED-low, LED-high, Incandescent, Natural (low-light validation)
    * Activities: Stationary, Rotation, Talking, Walking (motion robustness)
- Ground truth: Synchronized contact PPG sensor waveforms and heart rate series
"""

import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np
import scipy.io as sio

from preprocessing.roi_extractor import ROIExtractor, filter_skin_pixels
from preprocessing.skin_tone import analyze_skin_tone
from estimation.czt_estimator import CZTHRTransformer


MMPD_LIGHT_CONDITIONS = ["LED-low", "LED-high", "Incandescent", "Natural"]
MMPD_ACTIVITIES = ["Stationary", "Rotation", "Talking", "Walking"]
MMPD_SKIN_TYPES = ["Type III", "Type IV", "Type V", "Type VI"]


class MMPDDataLoader:
    """Loader for the MMPD cross-dataset benchmark."""

    def __init__(self, data_root: str = "data/MMPD"):
        self.data_root = Path(data_root)
        self.recordings = self._discover_recordings()

    def _discover_recordings(self) -> List[Dict[str, Any]]:
        """Discover all available MMPD recordings in the dataset root."""
        if not self.data_root.exists():
            return []

        recordings = []
        # Search for .mat files directly or inside subject subdirectories
        mat_files = sorted(list(self.data_root.glob("**/*.mat")))

        for mat_path in mat_files:
            name = mat_path.stem
            # Find matching video file if stored separately
            vid_candidates = [
                mat_path.with_suffix(".mp4"),
                mat_path.with_suffix(".avi"),
                mat_path.parent / f"{name}.mp4",
            ]
            video_file = None
            for vc in vid_candidates:
                if vc.exists():
                    video_file = vc
                    break

            recordings.append({
                "recording_id": name,
                "mat_path": mat_path,
                "video_file": video_file,
                "dir": mat_path.parent,
            })

        return recordings

    @staticmethod
    def parse_mmpd_mat(mat_path: Path) -> Dict[str, Any]:
        """Parse MMPD .mat metadata, ground truth PPG, and video (if embedded)."""
        mat = sio.loadmat(str(mat_path))

        # Ground truth PPG waveform
        ppg_signal = None
        for k in ["d_ppg", "ppg", "PPG", "bvp", "ground_truth"]:
            if k in mat:
                ppg_signal = np.squeeze(mat[k]).astype(np.float64)
                break

        # Ground truth heart rate
        gt_hr_bpm = None
        for k in ["gt_HR", "hr", "pulse_rate", "HR", "mean_hr"]:
            if k in mat:
                val = np.squeeze(mat[k])
                if val.size == 1:
                    gt_hr_bpm = float(val)
                elif val.size > 1:
                    valid = val[val > 35.0]
                    gt_hr_bpm = float(np.mean(valid)) if len(valid) > 0 else float(np.mean(val))
                break

        # Extract metadata tags: light, motion, skin
        light = str(mat.get("light", ["Unknown"])[0]) if "light" in mat else "Unknown"
        motion = str(mat.get("motion", ["Unknown"])[0]) if "motion" in mat else "Unknown"
        skin = str(mat.get("skin_color", ["Unknown"])[0]) if "skin_color" in mat else "Unknown"
        fps = float(np.squeeze(mat["fs"])) if "fs" in mat else 30.0

        # If heart rate is not stored directly, compute via 512-bin CZT on PPG
        if gt_hr_bpm is None and ppg_signal is not None and len(ppg_signal) >= 128:
            czt = CZTHRTransformer(fs=fps, low_hz=0.7, high_hz=3.0, num_bins=512)
            gt_hr_bpm = float(czt.estimate_hr(ppg_signal))
        elif gt_hr_bpm is None:
            gt_hr_bpm = 75.0

        # Video frames if embedded in .mat
        video_frames = None
        if "video" in mat:
            raw_vid = mat["video"]
            # Typically [T, H, W, 3] or [H, W, 3, T]
            if raw_vid.ndim == 4:
                if raw_vid.shape[-1] == 3:
                    video_frames = raw_vid
                elif raw_vid.shape[2] == 3:
                    video_frames = np.transpose(raw_vid, (3, 0, 1, 2))

        return {
            "gt_hr_bpm": float(gt_hr_bpm),
            "ppg": ppg_signal,
            "light": light,
            "motion": motion,
            "skin": skin,
            "fps": fps,
            "video_frames": video_frames,
        }

    def load_recording_frames(
        self, rec_info: Dict[str, Any], max_frames: Optional[int] = 450
    ) -> Tuple[np.ndarray, float, float]:
        """Load frames and ground truth HR for an MMPD recording."""
        parsed = self.parse_mmpd_mat(rec_info["mat_path"])
        fps = parsed["fps"]
        gt_hr = parsed["gt_hr_bpm"]

        if parsed["video_frames"] is not None:
            frames = parsed["video_frames"]
            if max_frames is not None:
                frames = frames[:max_frames]
            return frames.astype(np.uint8), fps, gt_hr

        # Fallback to separate video file
        frames = []
        if rec_info.get("video_file") and rec_info["video_file"].exists():
            cap = cv2.VideoCapture(str(rec_info["video_file"]))
            fps = cap.get(cv2.CAP_PROP_FPS) or fps
            idx = 0
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret or (max_frames is not None and idx >= max_frames):
                    break
                frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                idx += 1
            cap.release()

        if len(frames) == 0:
            return np.zeros((0, 240, 320, 3), dtype=np.uint8), fps, gt_hr

        return np.array(frames, dtype=np.uint8), fps, gt_hr
