"""Audit Contact PPG CZT Power Spectra across all 42 UBFC Subjects for Bimodality.

Identifies recordings where the secondary cardiac peak has relative power >= 50%
of the dominant peak, evaluates clinical oximeter concurrence, and tracks branch peak selection.
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from datasets.ubfc_loader import UBFCDataLoader
from estimation.czt_estimator import CZTHRTransformer
from preprocessing.roi_extractor import ROIExtractor
from models.classical.pipeline import chrom_process, pos_process
from models.hybrid.beatformer import BeatFormer


def audit_subject_dual_peaks(
    subject_id: str,
    max_frames: int = 450,
) -> Dict[str, any]:
    gt_path = f"datasets/ubfc/{subject_id}/ground_truth.txt"
    vid_path = f"datasets/ubfc/{subject_id}/vid.mp4"
    if not os.path.exists(vid_path):
        vid_path = f"datasets/ubfc/{subject_id}/vid.avi"

    if not (os.path.exists(gt_path) and os.path.exists(vid_path)):
        return None

    gt_data = UBFCDataLoader.load_ground_truth(gt_path)
    cap = cv2.VideoCapture(vid_path)
    vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames)

    n_ext = len(frames)
    ts = gt_data["timestamps"][:n_ext]
    fps = float((len(ts) - 1) / (ts[-1] - ts[0])) if len(ts) > 1 and (ts[-1] - ts[0]) > 0 else vid_fps

    aligned_ppg = gt_data["ppg"][:n_ext] if len(gt_data["ppg"]) >= n_ext else gt_data["ppg"]
    oximeter_trace = gt_data["hr"][:n_ext] if len(gt_data["hr"]) >= n_ext else gt_data["hr"]
    oximeter_mean = float(np.mean(oximeter_trace)) if len(oximeter_trace) > 0 else float("nan")

    czt = CZTHRTransformer(fs=fps, low_hz=0.7, high_hz=3.0, num_bins=512)
    bpm_bins, psd_ppg = czt.compute_spectrum(aligned_ppg)

    # Find local spectral peaks
    local_peaks = []
    for i in range(1, len(psd_ppg) - 1):
        if psd_ppg[i] > psd_ppg[i - 1] and psd_ppg[i] > psd_ppg[i + 1]:
            local_peaks.append((float(psd_ppg[i]), float(bpm_bins[i])))

    local_peaks.sort(key=lambda x: x[0], reverse=True)
    if not local_peaks:
        return None

    p1_power, p1_bpm = local_peaks[0]
    p2_power, p2_bpm = (local_peaks[1] if len(local_peaks) > 1 else (0.0, float("nan")))
    peak_ratio = float(p2_power / max(1e-12, p1_power))
    is_bimodal = bool(peak_ratio >= 0.50)

    # For bimodal subjects, extract ROI and determine which peak each branch selected
    chr_hr, pos_hr, bf_hr = float("nan"), float("nan"), float("nan")
    chr_choice, pos_choice, bf_choice = "N/A", "N/A", "N/A"

    if is_bimodal and len(frames) >= 30:
        extractor = ROIExtractor()
        ext = extractor.process_video(frames, fs=fps)
        rgb = ext["rgb_signals"]

        bvp_chr = chrom_process(rgb, fs=fps)
        chr_hr = float(czt.estimate_hr(bvp_chr))

        bvp_pos = pos_process(rgb, fs=fps)
        pos_hr = float(czt.estimate_hr(bvp_pos))

        try:
            bf = BeatFormer()
            hr_b, _, _ = bf.estimate_hr_from_trace(rgb)
            bf_hr = float(hr_b)
        except Exception:
            bf_hr = float("nan")

        def assign_choice(hr_val):
            if np.isnan(hr_val):
                return "N/A"
            d1 = abs(hr_val - p1_bpm)
            d2 = abs(hr_val - p2_bpm)
            return "Peak 1 (Dominant)" if d1 <= d2 else "Peak 2 (Secondary)"

        chr_choice = assign_choice(chr_hr)
        pos_choice = assign_choice(pos_hr)
        bf_choice = assign_choice(bf_hr)

    return {
        "subject_id": subject_id,
        "fps": round(fps, 2),
        "dominant_peak_bpm": round(p1_bpm, 2),
        "secondary_peak_bpm": round(p2_bpm, 2) if not np.isnan(p2_bpm) else None,
        "peak_power_ratio": round(peak_ratio, 3),
        "is_bimodal": is_bimodal,
        "oximeter_mean_bpm": round(oximeter_mean, 2),
        "oximeter_vs_dominant_delta": round(abs(oximeter_mean - p1_bpm), 2),
        "oximeter_vs_secondary_delta": round(abs(oximeter_mean - p2_bpm), 2) if not np.isnan(p2_bpm) else None,
        "chrom_hr_bpm": round(chr_hr, 2) if not np.isnan(chr_hr) else None,
        "chrom_peak_choice": chr_choice,
        "pos_hr_bpm": round(pos_hr, 2) if not np.isnan(pos_hr) else None,
        "pos_peak_choice": pos_choice,
        "beatformer_hr_bpm": round(bf_hr, 2) if not np.isnan(bf_hr) else None,
        "beatformer_peak_choice": bf_choice,
    }


def main():
    loader = UBFCDataLoader(data_root="datasets/ubfc")
    subjects = [s["subject_id"] for s in loader.subjects if s["has_video"]]
    subjects.sort(key=lambda x: int(x.replace("subject", "")))

    print(f"Auditing CZT Contact PPG Spectra for {len(subjects)} UBFC Subjects (First 450 Frames)...")
    records = []
    for sid in subjects:
        rec = audit_subject_dual_peaks(sid)
        if rec:
            records.append(rec)
            status = " [BIMODAL]" if rec["is_bimodal"] else ""
            print(f"  {sid:10s}: Dom={rec['dominant_peak_bpm']:.1f} BPM, Sec={rec['secondary_peak_bpm']}, Ratio={rec['peak_power_ratio']:.2f}{status}")

    df = pd.DataFrame(records)
    os.makedirs("artifacts/constraints", exist_ok=True)
    out_path = "artifacts/constraints/gt_dual_peak_census.csv"
    df.to_csv(out_path, index=False)
    print(f"\nCensus written to {out_path}")

    bimodal_df = df[df["is_bimodal"]]
    print(f"\nTotal Bimodal Subjects (Ratio >= 0.50): {len(bimodal_df)} / {len(df)}")
    print(bimodal_df[["subject_id", "dominant_peak_bpm", "secondary_peak_bpm", "peak_power_ratio", "oximeter_mean_bpm", "chrom_peak_choice", "beatformer_peak_choice"]].to_string(index=False))


if __name__ == "__main__":
    main()
