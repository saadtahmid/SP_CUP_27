"""Packaging runner script for Round 1 submission bundle.

Generates:
/submission/Round1_Baseline/
  ├── Round1_Baseline_None.mat
  ├── process_round1.m
  ├── target_video.mp4
  ├── reference_signal.csv
  └── processed_output.csv
"""

import os
import numpy as np

from datasets.synthetic_generator import generate_synthetic_rppg_video
from preprocessing.roi_extractor import ROIExtractor
from preprocessing.skin_tone import analyze_skin_tone
from models.classical.pipeline import ClassicalRPPGPipeline
from evaluation.metrics import compute_all_metrics
from pipeline_packaging.exporter import export_submission_bundle


def main():
    print("=" * 70)
    print("PACKAGING ROUND 1 OFFICIAL SUBMISSION BUNDLE")
    print("=" * 70)

    output_dir = "submission/Round1_Baseline"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # 1. Generate representative official target baseline video (HD 1280x720, 30 fps, 300 lux, 75.0 BPM)
    target_video_raw = "artifacts/official_target_video.mp4"
    target_gt_raw = "artifacts/official_reference_signal.csv"
    target_hr = 75.0

    print("Generating official baseline video...")
    video_meta = generate_synthetic_rppg_video(
        output_video_path=target_video_raw,
        output_gt_csv_path=target_gt_raw,
        duration_sec=10.0,
        fps=30.0,
        target_hr_bpm=target_hr,
        width=1280,
        height=720,
        fitzpatrick_type="Type III",
        lux=300.0,
    )

    # 2. Run extraction pipeline
    print("Extracting facial ROIs and running Classical POS + CZT pipeline...")
    extractor = ROIExtractor()
    extracted = extractor.process_video(target_video_raw)

    pipeline = ClassicalRPPGPipeline(method="POS", fs=30.0, use_czt=True)
    result = pipeline.process(extracted["rgb_signals"], return_spectrum=True)
    estimated_hr = result["hr_bpm"]

    skin_info = analyze_skin_tone(extracted["skin_sample"])
    print(f"Skin Tone Analysis: {skin_info}")
    print(f"Ground Truth HR: {target_hr:.2f} BPM | Estimated HR: {estimated_hr:.2f} BPM")

    # 3. Format signals and metrics
    t_len = len(extracted["frames"])
    reference_hr_series = np.full(t_len, target_hr, dtype=np.float64)
    estimated_hr_series = np.full(t_len, estimated_hr, dtype=np.float64)

    metrics = compute_all_metrics(estimated_hr_series, reference_hr_series)
    print("Computed Submission Metrics:", metrics)

    params = {
        "resolution": "1280x720",
        "fps": 30.0,
        "lux": 300.0,
        "camera_distance_cm": 50.0,
        "fitzpatrick_type": skin_info["fitzpatrick_type"],
        "age": 25,
        "constraints_applied": "None (Baseline)",
    }

    # 4. Export submission bundle
    print("Exporting submission bundle to:", output_dir)
    bundle_status = export_submission_bundle(
        output_dir=output_dir,
        video_path=target_video_raw,
        gt_csv_path=target_gt_raw,
        facial_frames=extracted["frames"],
        roi_masks=extracted["roi_masks"],
        reference_hr_series=reference_hr_series,
        estimated_hr_series=estimated_hr_series,
        metrics=metrics,
        params=params,
    )

    print("\nSubmission Bundle Successfully Created:")
    for k, v in bundle_status.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
