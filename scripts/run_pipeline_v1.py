"""Run and evaluate Pipeline v1 (Classical POS, CHROM, GREEN, ICA with CZT).

Evaluates classical methods on synthetic calibrated benchmarks or AiPEX dataset,
comparing coarse FFT vs high-resolution CZT spectral estimation.
"""

import os
import argparse
import numpy as np
import pandas as pd

from datasets.synthetic_generator import generate_synthetic_rppg_video
from preprocessing.roi_extractor import ROIExtractor
from models.classical.pipeline import ClassicalRPPGPipeline
from evaluation.metrics import compute_all_metrics
from evaluation.bland_altman import plot_bland_altman


def main():
    parser = argparse.ArgumentParser(description="Run Pipeline v1 Classical Baselines")
    parser.add_argument("--method", default="POS", choices=["POS", "CHROM", "GREEN", "ICA"])
    parser.add_argument("--use_czt", action="store_true", default=True, help="Use Chirp-Z Transform")
    parser.add_argument("--test_data", default="synthetic", choices=["synthetic", "aipex"])
    args = parser.parse_args()

    print(f"=== Running Pipeline v1: {args.method} (CZT={args.use_czt}) on {args.test_data} ===")

    if args.test_data == "synthetic":
        os.makedirs("artifacts", exist_ok=True)
        test_vid = "artifacts/sample_baseline_video.mp4"
        test_gt = "artifacts/sample_baseline_gt.csv"

        # Generate a clean 10-second synthetic test video at 78 BPM, 30 fps, Type III, 300 lux
        target_hr = 78.0
        generate_synthetic_rppg_video(
            output_video_path=test_vid,
            output_gt_csv_path=test_gt,
            duration_sec=10.0,
            fps=30.0,
            target_hr_bpm=target_hr,
            fitzpatrick_type="Type III",
            lux=300.0,
        )

        extractor = ROIExtractor()
        extracted = extractor.process_video(test_vid)
        rgb_signals = extracted["rgb_signals"]

        pipeline = ClassicalRPPGPipeline(method=args.method, fs=30.0, use_czt=args.use_czt)
        res = pipeline.process(rgb_signals, return_spectrum=True)

        estimated_hr = res["hr_bpm"]
        error = abs(estimated_hr - target_hr)

        print(f"Algorithm: {args.method}")
        print(f"Target True HR: {target_hr:.2f} BPM")
        print(f"Estimated HR:   {estimated_hr:.2f} BPM")
        print(f"Absolute Error: {error:.2f} BPM")

        metrics = compute_all_metrics(np.array([estimated_hr]), np.array([target_hr]))
        print("Metrics:", metrics)
        print("Pipeline v1 run successful!")


if __name__ == "__main__":
    main()
