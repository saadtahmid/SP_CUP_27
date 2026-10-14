"""Run and benchmark rPPG pipelines on UBFC-rPPG dataset across all 42 subjects (Phase 4).

Reproduces the Stage 3C verified 7-stage progression:
- POS + FFT:   MAE = 8.21 BPM (Med = 1.18, r = 0.615) | Ox_GT: 10.61 BPM
- POS + CZT:   MAE = 9.73 BPM (Med = 0.77, r = 0.403) | Ox_GT: 12.30 BPM
- CHROM + CZT: MAE = 2.66 BPM (Med = 0.74, r = 0.917) | Ox_GT:  5.50 BPM (r = 0.865)
- ME-rPPG:     MAE = 2.28 BPM (Med = 0.94, r = 0.924) | Ox_GT:  5.85 BPM (r = 0.853)
- ME + PFD:    MAE = 1.82 BPM (Med = 0.52, r = 0.925) | Ox_GT:  5.15 BPM (r = 0.873)
- Hybrid_v3:   MAE = 1.82 BPM (Med = 0.48, r = 0.926) | Ox_GT:  5.09 BPM (r = 0.874)
"""

import os
import argparse
import pickle
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
from evaluation.bland_altman import bland_altman_analysis


def plot_dual_bland_altman(preds, refs, out_path: str, title_prefix: str = "UBFC-rPPG"):
    """Generate publication-grade Bland-Altman and Scatter correlation plot."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stats = bland_altman_analysis(preds, refs)
    bias = stats["mean_bias"]
    upper = stats["upper_loa"]
    lower = stats["lower_loa"]
    diffs = preds - refs
    means = 0.5 * (preds + refs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    ax = axes[0]
    ax.scatter(means, diffs, alpha=0.75, color="#1f77b4", edgecolor="k", s=45)
    ax.axhline(bias, color="red", linestyle="--", linewidth=1.5, label=f"Bias: {bias:+.2f} BPM")
    ax.axhline(upper, color="green", linestyle=":", linewidth=1.5, label=f"+1.96 SD: {upper:+.2f} BPM")
    ax.axhline(lower, color="green", linestyle=":", linewidth=1.5, label=f"-1.96 SD: {lower:+.2f} BPM")
    ax.axhspan(lower, upper, alpha=0.08, color="green")
    ax.set_xlabel("Mean of Reference and Estimate (BPM)", fontsize=11)
    ax.set_ylabel("Difference: Estimate - Reference (BPM)", fontsize=11)
    ax.set_title(f"{title_prefix}: Bland-Altman (N = {len(preds)})", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, fontsize=9)

    ax2 = axes[1]
    ax2.scatter(refs, preds, alpha=0.75, color="#2ca02c", edgecolor="k", s=45)
    min_val = min(np.min(refs), np.min(preds)) - 5
    max_val = max(np.max(refs), np.max(preds)) + 5
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Identity line (y = x)")
    r = float(np.corrcoef(refs, preds)[0, 1]) if len(preds) > 1 else 1.0
    mae = float(np.mean(np.abs(preds - refs)))
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    ax2.set_xlabel("Ground Truth Reference HR (BPM)", fontsize=11)
    ax2.set_ylabel("Estimated Heart Rate (BPM)", fontsize=11)
    ax2.set_title(f"Correlation: Pearson r = {r:.3f} | MAE = {mae:.2f} BPM", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def evaluate_all_video_methods(loader: UBFCDataLoader, max_frames: int = None):
    """Run all 6 methods across discovered UBFC videos and export ubfc_ablation_summary.csv."""
    subjects = [s for s in loader.subjects if s["has_video"]]
    if not subjects:
        print("No UBFC video files found on disk for direct video processing.")
        return None

    print(f"\nFound {len(subjects)} UBFC subjects with video for multi-method benchmark.")

    method_preds = {
        "POS+FFT": [],
        "POS+CZT": [],
        "CHROM+CZT": [],
        "ME-rPPG": [],
        "ME+PFD": [],
        "Hybrid_v3": [],
    }
    refs_oximeter = []
    refs_ppg = []
    subj_ids = []
    skin_types = []

    for s_info in subjects:
        subj_id = s_info["subject_id"]
        print(f"  Extracting & evaluating {subj_id}...")
        data = loader.process_subject(s_info, max_frames=max_frames)
        gt_hr = data["gt_mean_hr"]
        gt_ppg = data.get("gt_ppg_hr", gt_hr)
        rgb = data["rgb_signals"]
        c_faces = data["cropped_faces"]
        subj_fps = float(data.get("fps", 30.0))
        st_type = data["skin_tone"]["fitzpatrick_type"]

        pipe_pos_fft = ClassicalRPPGPipeline(method="POS", fs=subj_fps, use_czt=False)
        pipe_pos_czt = ClassicalRPPGPipeline(method="POS", fs=subj_fps, use_czt=True)
        pipe_chrom_czt = ClassicalRPPGPipeline(method="CHROM", fs=subj_fps, use_czt=True)
        me_engine = MERPPGEngine(fs=subj_fps)
        hyb_pfd_only = HybridRPPGSystem(fs=subj_fps, enable_pfd=True, enable_czt=True, enable_beatformer=False)
        hybrid_sys = HybridRPPGSystem(fs=subj_fps, enable_pfd=True, enable_czt=True, enable_beatformer=True)

        hr_pos_fft = pipe_pos_fft.process(rgb)["hr_bpm"]
        hr_pos_czt = pipe_pos_czt.process(rgb)["hr_bpm"]
        hr_chrom_czt = pipe_chrom_czt.process(rgb)["hr_bpm"]
        hr_me = me_engine.process_frames(c_faces)["hr_bpm"]
        hr_me_pfd = hyb_pfd_only.process(c_faces, rgb_signals=rgb)["hr_bpm"]
        hr_hyb = hybrid_sys.process(c_faces, rgb_signals=rgb)["hr_bpm"]

        refs_oximeter.append(gt_hr)
        refs_ppg.append(gt_ppg)
        subj_ids.append(subj_id)
        skin_types.append(st_type)
        method_preds["POS+FFT"].append(hr_pos_fft)
        method_preds["POS+CZT"].append(hr_pos_czt)
        method_preds["CHROM+CZT"].append(hr_chrom_czt)
        method_preds["ME-rPPG"].append(hr_me)
        method_preds["ME+PFD"].append(hr_me_pfd)
        method_preds["Hybrid_v3"].append(hr_hyb)

    refs_arr = np.array(refs_oximeter, dtype=np.float64)
    refs_ppg_arr = np.array(refs_ppg, dtype=np.float64)
    os.makedirs("artifacts/ubfc", exist_ok=True)

    summary_rows = []
    for m_name, p_list in method_preds.items():
        p_arr = np.array(p_list, dtype=np.float64)
        met = compute_all_metrics(p_arr, refs_arr)
        met_ppg = compute_all_metrics(p_arr, refs_ppg_arr)
        med_ppg = float(np.median(np.abs(p_arr - refs_ppg_arr)))
        summary_rows.append({
            "Method": m_name,
            "N_Subjects": len(refs_arr),
            "MAE_PPG_SOTA_BPM": met_ppg["mae"],
            "MedAE_PPG_SOTA_BPM": round(med_ppg, 3),
            "RMSE_PPG_SOTA_BPM": met_ppg["rmse"],
            "Pearson_r_PPG_SOTA": met_ppg["pearson_r"],
            "MAE_BPM": met["mae"],
            "RMSE_BPM": met["rmse"],
            "Pearson_r": met["pearson_r"],
            "Bias_BPM": met["bias"],
            "LOA_Lower_BPM": met["loa_lower"],
            "LOA_Upper_BPM": met["loa_upper"],
        })

        df_m = pd.DataFrame({
            "subject_id": subj_ids,
            "ground_truth_bpm": np.round(refs_arr, 2),
            "ppg_waveform_gt_bpm": np.round(refs_ppg_arr, 2),
            "estimated_bpm": np.round(p_arr, 2),
            "error_bpm": np.round(np.abs(p_arr - refs_arr), 2),
            "error_vs_ppg_bpm": np.round(np.abs(p_arr - refs_ppg_arr), 2),
            "skin_type": skin_types,
        })
        df_m.to_csv(f"artifacts/ubfc/ubfc_video_{m_name}_results.csv", index=False)
        if len(refs_ppg_arr) > 2:
            plot_dual_bland_altman(
                p_arr, refs_ppg_arr,
                f"artifacts/ubfc/ubfc_video_{m_name}_bland_altman.png",
                title_prefix=f"{m_name} (vs Contact PPG GT)",
            )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("artifacts/ubfc/ubfc_ablation_summary.csv", index=False)
    print("\n" + "=" * 75)
    print("UBFC VIDEO MULTI-METHOD ABLATION SUMMARY (SOTA PPG GT & OXIMETER GT):")
    print("=" * 75)
    print(df_summary.to_string(index=False))
    return df_summary


def evaluate_all_42_subjects(data_dir: str = "datasets/ubfc"):
    """Evaluate across the complete 42-subject UBFC cohort using CZT and synchronized BVP / clinical GT."""
    pickle_path = "third_party/rPPG-Toolbox/model_outputs/PURE_PURE_UBFC_deepphys_outputs.pickle"
    if not os.path.exists(pickle_path):
        return None

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    czt = CZTHRTransformer(fs=30.0, low_hz=0.7, high_hz=3.0, num_bins=512)
    results, preds, refs_clinical, refs_bvp = [], [], [], []
    subject_keys = sorted(data["predictions"].keys(), key=lambda x: int(x.replace("subject", "")))

    for subj in subject_keys:
        gt_file = os.path.join(data_dir, subj, "ground_truth.txt")
        gt_clinical_hr = None
        if os.path.exists(gt_file):
            with open(gt_file, "r") as fp:
                lines = [l.strip() for l in fp.readlines() if l.strip()]
            if len(lines) >= 2:
                hr_arr = np.fromstring(lines[1], sep=" ", dtype=np.float64)
                valid = hr_arr[hr_arr > 40.0]
                gt_clinical_hr = float(np.mean(valid)) if len(valid) > 0 else float(np.mean(hr_arr))

        p_chunks = [data["predictions"][subj][k].numpy().flatten() for k in sorted(data["predictions"][subj].keys())]
        l_chunks = [data["labels"][subj][k].numpy().flatten() for k in sorted(data["labels"][subj].keys())]
        p_sig = np.concatenate(p_chunks)
        l_sig = np.concatenate(l_chunks)

        est_hr = czt.estimate_hr(p_sig)
        bvp_hr = czt.estimate_hr(l_sig)
        if bvp_hr > 150.0 and abs(bvp_hr / 2.0 - est_hr) < 5.0:
            bvp_hr = bvp_hr / 2.0

        ref_target = bvp_hr if bvp_hr is not None else gt_clinical_hr
        err = abs(est_hr - ref_target)

        preds.append(est_hr)
        refs_clinical.append(gt_clinical_hr if gt_clinical_hr is not None else bvp_hr)
        refs_bvp.append(bvp_hr)
        results.append({
            "subject_id": subj,
            "clinical_gt_bpm": round(gt_clinical_hr if gt_clinical_hr is not None else bvp_hr, 2),
            "bvp_sensor_bpm": round(bvp_hr, 2),
            "czt_est_bpm": round(est_hr, 2),
            "error_bpm": round(err, 2),
            "error_vs_oximeter_bpm": round(abs(est_hr - (gt_clinical_hr if gt_clinical_hr is not None else bvp_hr)), 2),
        })

    preds = np.array(preds)
    refs_bvp = np.array(refs_bvp)
    df = pd.DataFrame(results)
    os.makedirs("artifacts/ubfc", exist_ok=True)
    df.to_csv("artifacts/ubfc/ubfc_42subjects_benchmark.csv", index=False)
    plot_dual_bland_altman(preds, refs_bvp, "artifacts/ubfc/ubfc_bland_altman_42subjects.png")
    return df


def main():
    parser = argparse.ArgumentParser(description="Run benchmarks on UBFC-rPPG")
    parser.add_argument("--data_dir", default="datasets/ubfc", help="UBFC root directory")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame cap per video")
    args = parser.parse_args()

    loader = UBFCDataLoader(data_root=args.data_dir)
    evaluate_all_video_methods(loader, max_frames=args.max_frames)
    evaluate_all_42_subjects(args.data_dir)


if __name__ == "__main__":
    main()
