"""Dataset Loader for UBFC-rPPG Dataset.

Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., Dubois, J. (2017).
Unsupervised skin tissue segmentation for remote photoplethysmography.
Pattern Recognition Letters.

Folder layout:
datasets/ubfc/
└── subject{N}/
    ├── vid.avi (or vid.mp4)
    └── ground_truth.txt
"""

import os
import glob
import subprocess
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from preprocessing.roi_extractor import ROIExtractor
from preprocessing.skin_tone import analyze_skin_tone


class UBFCDataLoader:
    """Loader for UBFC-rPPG dataset."""

    def __init__(self, data_root: str = "datasets/ubfc"):
        self.data_root = data_root
        self.subjects = self._discover_subjects()

    def _discover_subjects(self) -> List[Dict[str, str]]:
        subj_dirs = sorted(glob.glob(os.path.join(self.data_root, "subject*")))
        subjects = []
        for d in subj_dirs:
            subj_id = os.path.basename(d)
            mp4_path = os.path.join(d, "vid.mp4")
            avi_path = os.path.join(d, "vid.avi")
            gt_path = os.path.join(d, "ground_truth.txt")

            chosen_vid = mp4_path if os.path.exists(mp4_path) else avi_path
            has_vid = os.path.exists(mp4_path) or os.path.exists(avi_path)

            if os.path.exists(gt_path):
                subjects.append({
                    "subject_id": subj_id,
                    "dir": d,
                    "video_path": chosen_vid,
                    "gt_path": gt_path,
                    "has_video": has_vid,
                })
        return subjects

    @staticmethod
    def load_ground_truth(gt_path: str) -> Dict[str, np.ndarray]:
        """Parse UBFC ground_truth.txt (3 lines: PPG, HR, Time)."""
        with open(gt_path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if len(lines) < 3:
            raise ValueError(f"Invalid UBFC ground_truth file: {gt_path}")

        ppg = np.fromstring(lines[0], sep=" ", dtype=np.float64)
        hr = np.fromstring(lines[1], sep=" ", dtype=np.float64)
        timestamps = np.fromstring(lines[2], sep=" ", dtype=np.float64)

        # Filter out oximeter sensor dropouts (HR <= 40.0 bpm)
        valid_hr = hr[hr > 40.0]
        mean_hr_bpm = float(np.mean(valid_hr)) if len(valid_hr) > 0 else float(np.mean(hr))

        return {
            "ppg": ppg,
            "hr": hr,
            "timestamps": timestamps,
            "mean_hr_bpm": mean_hr_bpm,
        }

    def process_subject(self, subject_info: Dict[str, str], max_frames: Optional[int] = None) -> Dict[str, any]:
        """Extract RGB traces and ground-truth HR from a UBFC subject."""
        gt_data = self.load_ground_truth(subject_info["gt_path"])
        vid_path = subject_info["video_path"]

        # Ensure playable video format if raw AVI
        if vid_path.endswith(".avi") and not os.path.exists(vid_path.replace(".avi", ".mp4")):
            mp4_path = vid_path.replace(".avi", ".mp4")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", vid_path, "-c:v", "libx264", "-crf", "18", mp4_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
                )
                vid_path = mp4_path
            except Exception:
                pass

        if not os.path.exists(vid_path):
            raise FileNotFoundError(f"Video not found for {subject_info['subject_id']}")

        # Read frames up to max_frames (by default all frames of vid.mp4 ~1800 frames)
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames is not None and count >= max_frames):
                break
            frames.append(frame)
            count += 1
        cap.release()

        frames_arr = np.array(frames)
        extractor = ROIExtractor()
        extracted = extractor.process_video(frames_arr)

        skin_info = analyze_skin_tone(extracted["skin_sample"])

        # Align ground truth HR to extracted duration and filter sensor dropouts
        aligned_gt_hr = gt_data["hr"][:len(frames_arr)] if len(gt_data["hr"]) >= len(frames_arr) else gt_data["hr"]
        valid_aligned = aligned_gt_hr[aligned_gt_hr > 40.0]
        mean_gt_hr = float(np.mean(valid_aligned)) if len(valid_aligned) > 0 else float(np.mean(aligned_gt_hr))

        return {
            "subject_id": subject_info["subject_id"],
            "rgb_signals": extracted["rgb_signals"],
            "cropped_faces": extracted["cropped_faces"],
            "roi_masks": extracted["roi_masks"],
            "frames": extracted["frames"],
            "fps": fps,
            "skin_tone": skin_info,
            "gt_hr_series": aligned_gt_hr,
            "gt_timestamps": gt_data["timestamps"][:len(frames_arr)],
            "gt_mean_hr": mean_gt_hr,
        }
