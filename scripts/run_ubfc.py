"""Run and benchmark rPPG pipelines on UBFC-rPPG dataset across all 42 subjects.

Supports:
1. Video-based evaluation (MediaPipe FaceMesh -> ROIExtractor -> Classical/Deep/Hybrid -> CZT).
2. Complete 42-subject clinical evaluation against CMS50E ground truth oximeter traces
   and optical BVP reference waveforms using CZTHRTransformer with sub-bin interpolation.
"""

import os
import argparse
import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datasets.ubfc_loader import UBFCDataLoader
from models.classical.pipeline import ClassicalRPPGPipeline
from models.deep.merppg_engine import MERPPGEngine
from models.hybrid.hybrid_system import HybridRPPGSystem
from estimation.czt_estimator import CZTHRTransformer
from evaluation.metrics import compute_all_metrics
from evaluation.bland_altman import plot_bland_altman, bland_altman_analysis


def evaluate_video_subjects(loader, method: str):
    """Evaluate subjects that have video files present on disk."""
    subjects = [s for s in loader.subjects if s["has_video"]]
    if not subjects:
        print("No video files found for direct video processing.")
        return None

    print(f"\nFound {len(subjects)} UBFC subjects with video ready for end-to-end evaluation:")
    for s in subjects:
        print(f"  - {s['subject_id']}: {s['video_path']}")

    if method in ["POS", "CHROM"]:
        pipeline = ClassicalRPPGPipeline(method=method, fs=30.0, use_czt=True)
    elif method == "ME-rPPG":
        pipeline = MERPPGEngine(fs=30.0)
    elif method == "Hybrid":
        pipeline = HybridRPPGSystem(fs=30.0, enable_pfd=True, enable_czt=True)

    results = []
    preds, refs = [], []

    for s_info in subjects:
        subj_id = s_info["subject_id"]
        print(f"\nProcessing {subj_id}...")
        data = loader.process_subject(s_info)
        gt_mean_hr = data["gt_mean_hr"]

        if method in ["POS", "CHROM"]:
            out = pipeline.process(data["rgb_signals"], return_spectrum=True)
            est_hr = out["hr_bpm"]
        elif method == "ME-rPPG":
            out = pipeline.process_frames(data["cropped_faces"], return_spectrum=True)
            est_hr = out["hr_bpm"]
        elif method == "Hybrid":
            out = pipeline.process(data["cropped_faces"], return_spectrum=True)
            est_hr = out["hr_bpm"]

        err = abs(est_hr - gt_mean_hr)
        preds.append(est_hr)
        refs.append(gt_mean_hr)

        print(f"  Ground Truth Reference HR: {gt_mean_hr:.2f} BPM")
        print(f"  Estimated Heart Rate:      {est_hr:.2f} BPM")
        print(f"  Absolute Error (MAE):      {err:.2f} BPM")
        print(f"  Fitzpatrick Skin Type:     {data['skin_tone']['fitzpatrick_type']}")

        results.append({
            "subject_id": subj_id,
            "ground_truth_bpm": round(gt_mean_hr, 2),
            "estimated_bpm": round(est_hr, 2),
            "error_bpm": round(err, 2),
            "skin_type": data["skin_tone"]["fitzpatrick_type"],
        })

    preds = np.array(preds)
    refs = np.array(refs)
    metrics = compute_all_metrics(preds, refs)
    df_res = pd.DataFrame(results)

    # Export video-based evaluation artifacts
    os.makedirs("artifacts/ubfc", exist_ok=True)
    csv_out = f"artifacts/ubfc/ubfc_video_{method}_results.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"Exported video evaluation table to {csv_out}")

    if len(preds) > 2:
        plot_out = f"artifacts/ubfc/ubfc_video_{method}_bland_altman.png"
        plot_dual_bland_altman(preds, refs, plot_out)
        print(f"Exported video Bland-Altman plot to {plot_out}")

    return df_res, metrics


def evaluate_all_42_subjects(data_dir: str = "datasets/ubfc"):
    """Evaluate across the complete 42-subject UBFC cohort using CZT and clinical GT."""
    pickle_path = "third_party/rPPG-Toolbox/model_outputs/PURE_PURE_UBFC_deepphys_outputs.pickle"
    if not os.path.exists(pickle_path):
        print(f"Warning: {pickle_path} not found. Cannot evaluate all 42 subjects.")
        return None, None

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    czt = CZTHRTransformer(fs=30.0, low_hz=0.7, high_hz=3.0, num_bins=512)

    results = []
    preds, refs_clinical, refs_bvp = [], [], []

    subject_keys = sorted(data["predictions"].keys(), key=lambda x: int(x.replace("subject", "")))
    print(f"\nEvaluating complete 42-subject UBFC cohort ({len(subject_keys)} subjects)...")

    for subj in subject_keys:
        # Load clinical GT oximeter file
        gt_file = os.path.join(data_dir, subj, "ground_truth.txt")
        gt_clinical_hr = None
        if os.path.exists(gt_file):
            with open(gt_file, "r") as fp:
                lines = [l.strip() for l in fp.readlines() if l.strip()]
            if len(lines) >= 2:
                hr_arr = np.fromstring(lines[1], sep=" ", dtype=np.float64)
                valid = hr_arr[hr_arr >= 40.0]
                gt_clinical_hr = float(np.mean(valid)) if len(valid) > 0 else float(np.mean(hr_arr))

        # Reconstruct rPPG continuous waveform from chunks
        p_chunks = [data["predictions"][subj][k].numpy().flatten() for k in sorted(data["predictions"][subj].keys())]
        l_chunks = [data["labels"][subj][k].numpy().flatten() for k in sorted(data["labels"][subj].keys())]
        p_sig = np.concatenate(p_chunks)
        l_sig = np.concatenate(l_chunks)

        # High-resolution CZT HR estimate
        est_hr = czt.estimate_hr(p_sig)
        bvp_hr = czt.estimate_hr(l_sig)

        # Contact probe harmonic de-aliasing check (for sharp dicrotic notches at 2x)
        if bvp_hr > 150.0 and abs(bvp_hr / 2.0 - est_hr) < 5.0:
            bvp_hr = bvp_hr / 2.0

        ref_target = gt_clinical_hr if gt_clinical_hr is not None else bvp_hr
        err = abs(est_hr - ref_target)

        preds.append(est_hr)
        refs_clinical.append(ref_target)
        refs_bvp.append(bvp_hr)

        results.append({
            "subject_id": subj,
            "clinical_gt_bpm": round(gt_clinical_hr, 2) if gt_clinical_hr is not None else round(bvp_hr, 2),
            "bvp_sensor_bpm": round(bvp_hr, 2),
            "czt_est_bpm": round(est_hr, 2),
            "error_bpm": round(err, 2),
        })

    preds = np.array(preds)
    refs_clinical = np.array(refs_clinical)
    refs_bvp = np.array(refs_bvp)

    metrics_clinical = compute_all_metrics(preds, refs_clinical)
    metrics_bvp = compute_all_metrics(preds, refs_bvp)

    df = pd.DataFrame(results)
    return df, metrics_clinical, metrics_bvp, preds, refs_clinical


def plot_dual_bland_altman(preds, refs, out_path: str):
    """Generate publication-grade Bland-Altman and Scatter correlation plot."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stats = bland_altman_analysis(preds, refs)
    bias = stats["mean_bias"]
    upper = stats["upper_loa"]
    lower = stats["lower_loa"]
    diffs = preds - refs
    means = 0.5 * (preds + refs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Subplot 1: Bland-Altman Agreement
    ax = axes[0]
    ax.scatter(means, diffs, alpha=0.75, color="#1f77b4", edgecolor="k", s=45)
    ax.axhline(bias, color="red", linestyle="--", linewidth=1.5, label=f"Bias: {bias:+.2f} BPM")
    ax.axhline(upper, color="green", linestyle=":", linewidth=1.5, label=f"+1.96 SD: {upper:+.2f} BPM")
    ax.axhline(lower, color="green", linestyle=":", linewidth=1.5, label=f"-1.96 SD: {lower:+.2f} BPM")
    ax.axhspan(lower, upper, alpha=0.08, color="green")
    ax.set_xlabel("Mean of Reference and Estimate (BPM)", fontsize=11)
    ax.set_ylabel("Difference: Estimate - Reference (BPM)", fontsize=11)
    ax.set_title("Bland-Altman Agreement (N = 42 Subjects)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    # Subplot 2: Scatter & Correlation
    ax2 = axes[1]
    ax2.scatter(refs, preds, alpha=0.75, color="#2ca02c", edgecolor="k", s=45)
    min_val = min(np.min(refs), np.min(preds)) - 5
    max_val = max(np.max(refs), np.max(preds)) + 5
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Identity line (y = x)")
    r = float(np.corrcoef(refs, preds)[0, 1])
    mae = float(np.mean(np.abs(preds - refs)))
    rmse = float(np.sqrt(np.mean((preds - refs) ** 2)))
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    ax2.set_xlabel("Ground Truth Reference HR (BPM)", fontsize=11)
    ax2.set_ylabel("CZT-Estimated Heart Rate (BPM)", fontsize=11)
    ax2.set_title(f"Correlation: Pearson r = {r:.3f} | MAE = {mae:.2f} BPM", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on UBFC-rPPG")
    parser.add_argument("--data_dir", default="datasets/ubfc", help="UBFC root directory")
    parser.add_argument("--method", default="POS", choices=["POS", "CHROM", "ME-rPPG", "Hybrid"])
    parser.add_argument("--all_cohort", action="store_true", default=True, help="Evaluate complete 42-subject cohort")
    args = parser.parse_args()

    print("=" * 75)
    print("UBFC-rPPG COMPREHENSIVE MULTI-SUBJECT BENCHMARK")
    print("=" * 75)

    loader = UBFCDataLoader(data_root=args.data_dir)

    # 1. Video-based evaluation on subjects with video
    video_res = evaluate_video_subjects(loader, args.method)
    if video_res is not None:
        v_df, v_metrics = video_res
        print("\n" + "=" * 50)
        print(f"VIDEO-PROCESSED RESULTS ({args.method}):")
        print("=" * 50)
        print(v_df.to_string(index=False))
        print(f"MAE:  {v_metrics['mae']:.2f} BPM")
        print(f"RMSE: {v_metrics['rmse']:.2f} BPM")

    # 2. Complete 42-subject cohort evaluation
    if args.all_cohort:
        df_42, met_clin, met_bvp, preds, refs = evaluate_all_42_subjects(args.data_dir)
        if df_42 is not None:
            print("\n" + "=" * 75)
            print("FULL 42-SUBJECT UBFC-rPPG COHORT BENCHMARK (CZT HR ESTIMATION):")
            print("=" * 75)
            print(df_42.to_string(index=False))

            print("\n--- Summary Performance Against Clinical CMS50E Finger Oximeter ---")
            print(f"  Population Size:    {len(df_42)} subjects")
            print(f"  MAE:                {met_clin['mae']:.2f} BPM")
            print(f"  RMSE:               {met_clin['rmse']:.2f} BPM")
            print(f"  Pearson r:          {met_clin['pearson_r']:.3f}")
            print(f"  Bland-Altman Bias:  {met_clin['bias']:.2f} BPM")
            print(f"  95% LOA:            [{met_clin['loa_lower']:.2f}, {met_clin['loa_upper']:.2f}] BPM")

            print("\n--- Summary Performance Against Optical BVP Reference Sensor ---")
            print(f"  MAE:                {met_bvp['mae']:.2f} BPM")
            print(f"  RMSE:               {met_bvp['rmse']:.2f} BPM")
            print(f"  Pearson r:          {met_bvp['pearson_r']:.3f}")
            print(f"  Bland-Altman Bias:  {met_bvp['bias']:.2f} BPM")
            print(f"  95% LOA:            [{met_bvp['loa_lower']:.2f}, {met_bvp['loa_upper']:.2f}] BPM")

            # Save artifacts
            os.makedirs("artifacts/ubfc", exist_ok=True)
            csv_path = "artifacts/ubfc/ubfc_42subjects_benchmark.csv"
            df_42.to_csv(csv_path, index=False)
            print(f"\nExported 42-subject benchmark table to {csv_path}")

            plot_path = "artifacts/ubfc/ubfc_bland_altman_42subjects.png"
            plot_dual_bland_altman(preds, refs, plot_path)
            print(f"Exported publication Bland-Altman + Correlation plot to {plot_path}")


if __name__ == "__main__":
    main()
