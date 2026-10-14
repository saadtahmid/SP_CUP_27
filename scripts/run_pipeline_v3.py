"""Run and evaluate Pipeline v3: Hybrid Innovation (ME-rPPG + FreqPhys PFD + CZT)."""

import os
import argparse
import numpy as np

from datasets.synthetic_generator import generate_synthetic_rppg_video
from preprocessing.roi_extractor import ROIExtractor
from models.hybrid.hybrid_system import HybridRPPGSystem
from evaluation.metrics import compute_all_metrics


def main():
    print("=== Running Pipeline v3: Hybrid (ME-rPPG + FreqPhys PFD + CZT) ===")
    os.makedirs("artifacts", exist_ok=True)
    test_vid = "artifacts/sample_v3_video.mp4"
    test_gt = "artifacts/sample_v3_gt.csv"

    target_hr = 92.0
    generate_synthetic_rppg_video(
        output_video_path=test_vid,
        output_gt_csv_path=test_gt,
        duration_sec=10.0,
        fps=30.0,
        target_hr_bpm=target_hr,
        fitzpatrick_type="Type IV",
        lux=250.0,
    )

    extractor = ROIExtractor()
    extracted = extractor.process_video(test_vid)
    cropped_faces = extracted["cropped_faces"]

    hybrid = HybridRPPGSystem(
        onnx_model_path="third_party/ME-rPPG/model.onnx",
        state_json_path="third_party/ME-rPPG/state.json",
        fs=30.0,
        enable_pfd=True,
        enable_czt=True,
    )

    res = hybrid.process(cropped_faces, return_spectrum=True)
    estimated_hr = res["hr_bpm"]
    error = abs(estimated_hr - target_hr)

    print(f"Target True HR: {target_hr:.2f} BPM")
    print(f"Estimated HR:   {estimated_hr:.2f} BPM")
    print(f"Absolute Error: {error:.2f} BPM")
    print("Pipeline v3 run successful!")


if __name__ == "__main__":
    main()
