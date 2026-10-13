"""Dataset Loader for AiPEX rppg_biases (Annexure 1).

Dataset contains 88 subjects from India (n=44) and Sierra Leone (n=44).
Each subject has 3 ROI video recordings:
- forehead: video{N}_forehead.avi
- leftcheek: video{N}_leftcheek.avi
- rightcheek: video{N}_rightcheek.avi
Ground truth mean heart rates stored in India_GT.csv and Sierra_Leone_GT.csv.

Memory-optimized streaming reader: computes spatial averages per frame on the fly.
"""

import os
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd

from preprocessing.skin_tone import analyze_skin_tone


def _read_video_mean_rgb(video_path: str) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """Stream video and compute spatial mean RGB per frame with minimal memory footprint."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((0, 3)), 15.0, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
    mid_idx = total_frames // 2

    means = []
    mid_frame_rgb = None
    idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        # Convert BGR mean to RGB mean: frame_bgr[:, :, ::-1].mean(axis=(0, 1))
        # Equivalent: compute mean per channel directly
        b = float(np.mean(frame_bgr[:, :, 0]))
        g = float(np.mean(frame_bgr[:, :, 1]))
        r = float(np.mean(frame_bgr[:, :, 2]))
        means.append([r, g, b])

        if idx == mid_idx:
            mid_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        idx += 1

    cap.release()
    return np.array(means, dtype=np.float64), fps, mid_frame_rgb


class AiPEXDatasetLoader:
    """Loader for the official competition Annexure 1 AiPEX bias dataset."""

    def __init__(self, base_dir: str = "third_party/rppg_biases"):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "Data")
        self.gt_dir = os.path.join(base_dir, "Ground_Truth")

        self.india_gt_path = os.path.join(self.gt_dir, "India_GT.csv")
        self.sl_gt_path = os.path.join(self.gt_dir, "Sierra_Leone_GT.csv")

        self.subjects = self._discover_subjects()

    def _discover_subjects(self) -> List[Dict[str, any]]:
        subjects = []
        for cohort, gt_file in [("India", self.india_gt_path), ("Sierra_Leone", self.sl_gt_path)]:
            if not os.path.exists(gt_file):
                continue
            df = pd.read_csv(gt_file)
            df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
            subj_col = [c for c in df.columns if "subject" in c.lower()][0]
            hr_col = [c for c in df.columns if "heart_rate" in c.lower() or "bpm" in c.lower()][0]

            cohort_dir = os.path.join(self.data_dir, cohort)
            for _, row in df.iterrows():
                subj_id = str(row[subj_col]).strip()
                gt_hr = float(row[hr_col])
                fh_path = os.path.join(cohort_dir, f"{subj_id}_forehead.avi")
                lc_path = os.path.join(cohort_dir, f"{subj_id}_leftcheek.avi")
                rc_path = os.path.join(cohort_dir, f"{subj_id}_rightcheek.avi")

                if os.path.exists(fh_path):
                    subjects.append({
                        "subject_id": subj_id,
                        "cohort": cohort,
                        "gt_hr_bpm": gt_hr,
                        "forehead_video": fh_path,
                        "leftcheek_video": lc_path,
                        "rightcheek_video": rc_path,
                    })
        return subjects

    def load_subject_rgb_traces(self, subject_info: Dict[str, any]) -> Dict[str, any]:
        """Load and extract mean RGB traces across forehead and cheeks for a subject."""
        fh_mean, fps, mid_frame = _read_video_mean_rgb(subject_info["forehead_video"])

        lc_mean, _, _ = _read_video_mean_rgb(subject_info.get("leftcheek_video", ""))
        rc_mean, _, _ = _read_video_mean_rgb(subject_info.get("rightcheek_video", ""))

        combined_mean = fh_mean
        if len(lc_mean) == len(fh_mean) and len(rc_mean) == len(fh_mean) and len(fh_mean) > 0:
            combined_mean = 0.5 * fh_mean + 0.25 * lc_mean + 0.25 * rc_mean

        # Skin tone analysis on sample frame pixels
        if mid_frame is not None:
            sample_skin = mid_frame.reshape(-1, 3)
            skin_info = analyze_skin_tone(sample_skin)
        else:
            skin_info = {"ita_deg": 15.0, "fitzpatrick_type": "Type IV"}

        return {
            "subject_id": subject_info["subject_id"],
            "cohort": subject_info["cohort"],
            "gt_hr_bpm": subject_info["gt_hr_bpm"],
            "fps": fps,
            "rgb_signals": combined_mean,
            "num_frames": len(combined_mean),
            "skin_tone": skin_info,
        }
