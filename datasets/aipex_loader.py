"""Dataset Loader for AiPEX rppg_biases (Annexure 1).

Dataset contains 88 subjects from India (n=44) and Sierra Leone (n=44).
Each subject has 3 ROI video recordings:
- forehead: video{N}_forehead.avi
- leftcheek: video{N}_leftcheek.avi
- rightcheek: video{N}_rightcheek.avi
Ground truth mean heart rates stored in India_GT.csv and Sierra_Leone_GT.csv.

Uses a Frame-0 static intensity/glare mask per ROI stream to prevent per-frame
YCbCr threshold-switching discontinuities on dark skin tones (Fitzpatrick Types V-VI),
and fuses sub-ROIs via dynamic spectral SNR weighting.
"""

import os
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import pandas as pd

from preprocessing.skin_tone import analyze_skin_tone
from preprocessing.roi_extractor import filter_skin_pixels, fuse_sub_rois_by_snr


def _read_video_mean_rgb(video_path: str, max_frames: Optional[int] = 450) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """Stream pre-cropped ROI video and compute spatial mean RGB using a Frame-0 static mask."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((0, 3)), 15.0, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_COUNT)) if hasattr(cv2, "CAP_PROP_COUNT") else int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
    effective_total = min(total_frames, max_frames) if max_frames else total_frames
    mid_idx = max(0, effective_total // 2)

    means = []
    mid_frame_rgb = None
    static_mask = None
    idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or (max_frames is not None and idx >= max_frames):
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pix = frame_rgb.reshape(-1, 3)

        # Lock non-glare / non-shadow mask on Frame 0 so sampled pixel set never switches mid-video
        if static_mask is None or len(static_mask) != len(pix):
            cand_mask = (
                (pix[:, 0] > 3) & (pix[:, 0] < 245) &
                (pix[:, 1] > 3) & (pix[:, 1] < 245) &
                (pix[:, 2] > 3) & (pix[:, 2] < 245)
            )
            static_mask = cand_mask if np.sum(cand_mask) > 0.25 * len(pix) else np.ones(len(pix), dtype=bool)

        mean_rgb = np.mean(pix[static_mask], axis=0)
        means.append(mean_rgb)

        if idx == mid_idx:
            mid_frame_rgb = frame_rgb
        idx += 1

    cap.release()
    return np.array(means, dtype=np.float64), float(fps), mid_frame_rgb


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

    def load_subject_rgb_traces(self, subject_info: Dict[str, any], max_frames: Optional[int] = 450) -> Dict[str, any]:
        """Load and extract SNR-fused RGB traces across forehead and cheeks for a subject."""
        fh_mean, fps, mid_frame = _read_video_mean_rgb(subject_info["forehead_video"], max_frames=max_frames)
        lc_mean, _, _ = _read_video_mean_rgb(subject_info.get("leftcheek_video", ""), max_frames=max_frames)
        rc_mean, _, _ = _read_video_mean_rgb(subject_info.get("rightcheek_video", ""), max_frames=max_frames)

        min_len = len(fh_mean)
        if len(lc_mean) > 0 and len(rc_mean) > 0:
            min_len = min(len(fh_mean), len(lc_mean), len(rc_mean))

        if min_len >= 30 and len(lc_mean) >= min_len and len(rc_mean) >= min_len:
            sub_rois = {
                "forehead": fh_mean[:min_len],
                "left_cheek": lc_mean[:min_len],
                "right_cheek": rc_mean[:min_len],
            }
            combined_mean, _ = fuse_sub_rois_by_snr(sub_rois, fs=fps)
        else:
            combined_mean = fh_mean

        if mid_frame is not None:
            sample_skin = filter_skin_pixels(mid_frame.reshape(-1, 3))
            skin_info = analyze_skin_tone(sample_skin if len(sample_skin) > 0 else mid_frame.reshape(-1, 3))
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

