"""Phase 4 Multi-Subject Component Ablation Matrix Runner (scripts/run_ablation_matrix.py).

Evaluates all 6 genuine pipeline configurations across the stratified 10-subject UBFC cohort
(without single-subject tuning or artificial blending) under:
  1. UBFC Clean Baseline (300 lux)
  2. C1 Low-Light Threshold (50 lux physical linear attenuation)
  3. C1 Extreme Low-Light Floor (10 lux physical 1/30 attenuation, ~6 ADC counts)
  4. C7 Surgical Mask Occlusion (>= 40% lower-face occlusion)
Operates strictly within the competition-specified [0.7, 3.0] Hz (42-180 BPM) band.
Exports:
  - artifacts/ablation_matrix.csv
  - artifacts/ablation_matrix.md
"""

import os
import argparse
import cv2
import numpy as np
import pandas as pd

from preprocessing.roi_extractor import ROIExtractor
from models.classical.pipeline import ClassicalRPPGPipeline
from models.deep.merppg_engine import MERPPGEngine
from models.hybrid.hybrid_system import HybridRPPGSystem
from constraints.low_light import simulate_low_light_video
from constraints.mask_occlusion import simulate_masked_video
from scripts.run_constraints import load_stratified_ubfc_cohort


def extract_raw_bbox_rgb(frames: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Extract raw spatial RGB average from the full bounding box (without R4 sub-ROI or mask gate)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frames.shape[1:3]
    ry1, ry2 = max(0, y1), min(h, y2)
    rx1, rx2 = max(0, x1), min(w, x2)
    return np.mean(frames[:, ry1:ry2, rx1:rx2, ::-1], axis=(1, 2)).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="Multi-Subject Component Ablation Matrix")
    parser.add_argument("--num-subjects", type=int, default=10, help="Number of stratified UBFC subjects")
    parser.add_argument("--max-frames", type=int, default=450, help="Frames per UBFC video")
    args = parser.parse_args()

    os.makedirs("artifacts", exist_ok=True)
    print("=" * 80)
    print(f"RUNNING MULTI-SUBJECT COMPONENT ABLATION MATRIX (N = {args.num_subjects} UBFC SUBJECTS)")
    print("Cardiac Band: [0.70 Hz, 3.00 Hz] (42.0 - 180.0 BPM)")
    print("=" * 80)

    cohort = load_stratified_ubfc_cohort(max_subjects=args.num_subjects, max_frames=args.max_frames)
    extractor = ROIExtractor()

    configs = [
        ("1. Baseline v0 (POS + Welch FFT)", "Welch FFT", "Raw Full BBox", "None (POS)"),
        ("2. + CZT Zooming (POS + 512-bin CZT)", "512-bin CZT", "Raw Full BBox", "None (POS)"),
        ("3. + Locked R4 Sub-ROI & Mask Gate (CHROM)", "512-bin CZT", "Locked R4 Sub-ROI + Mask Gate", "Chrominance (CHROM)"),
        ("4. Pipeline v2 (ME-rPPG State-Space)", "512-bin CZT", "Locked BBox + 36x36 Crop", "ME-rPPG State-Space"),
        ("5. + FreqPhys PFD (Fundamental-Verified)", "512-bin CZT", "Locked BBox + 36x36 Crop", "ME-rPPG + PFD"),
        ("6. Pipeline v3/v4 (Full Consensus Hybrid)", "512-bin CZT", "Locked R4 Sub-ROI + Mask Gate", "ME-rPPG + PFD + BeatFormer + CHROM"),
    ]

    errs = {cfg[0]: {"base": [], "c1_50": [], "c1_10": [], "c7": []} for cfg in configs}

    for item in cohort:
        sid = item["subject_id"]
        frames = item["frames"]
        fps = item["fps"]
        gt_hr = item["gt_hr"]
        print(f"  Evaluating {sid} (fps={fps:.2f}, GT={gt_hr:.2f} BPM)...")

        lux = 50.0
        np.random.seed(1000 + int(lux))
        frames_50 = simulate_low_light_video(frames, target_lux=lux, baseline_lux=300.0, seed=int(1000 + lux))
        lux = 10.0
        np.random.seed(1000 + int(lux))
        frames_10 = simulate_low_light_video(frames, target_lux=lux, baseline_lux=300.0, seed=int(1000 + lux))
        masked_frames = simulate_masked_video(frames, tracker=extractor.tracker)

        ext_base = extractor.process_video(frames, fs=fps)
        bbox0 = extractor._locked_bbox.copy()
        ext_50 = extractor.process_video(frames_50, fs=fps)
        ext_10 = extractor.process_video(frames_10, fs=fps)
        ext_c7 = extractor.process_video(masked_frames, mask_occlusion_active=False, fs=fps)

        raw_base = extract_raw_bbox_rgb(frames, bbox0)
        raw_50 = extract_raw_bbox_rgb(frames_50, bbox0)
        raw_10 = extract_raw_bbox_rgb(frames_10, bbox0)
        raw_c7 = extract_raw_bbox_rgb(masked_frames, bbox0)

        pipe_pos_fft = ClassicalRPPGPipeline(method="POS", fs=fps, low_hz=0.7, high_hz=3.0, use_czt=False)
        pipe_pos_czt = ClassicalRPPGPipeline(method="POS", fs=fps, low_hz=0.7, high_hz=3.0, use_czt=True)
        pipe_chr_czt = ClassicalRPPGPipeline(method="CHROM", fs=fps, low_hz=0.7, high_hz=3.0, use_czt=True)
        me_engine = MERPPGEngine(fs=fps)
        hyb_pfd = HybridRPPGSystem(fs=fps, enable_pfd=True, enable_czt=True, enable_beatformer=False)
        hyb_full = HybridRPPGSystem(fs=fps, enable_pfd=True, enable_czt=True, enable_beatformer=True)

        # 1. POS + Welch FFT (Raw BBox)
        errs[configs[0][0]]["base"].append(abs(pipe_pos_fft.process(raw_base)["hr_bpm"] - gt_hr))
        errs[configs[0][0]]["c1_50"].append(abs(pipe_pos_fft.process(raw_50)["hr_bpm"] - gt_hr))
        errs[configs[0][0]]["c1_10"].append(abs(pipe_pos_fft.process(raw_10)["hr_bpm"] - gt_hr))
        errs[configs[0][0]]["c7"].append(abs(pipe_pos_fft.process(raw_c7)["hr_bpm"] - gt_hr))

        # 2. POS + 512-bin CZT (Raw BBox)
        errs[configs[1][0]]["base"].append(abs(pipe_pos_czt.process(raw_base)["hr_bpm"] - gt_hr))
        errs[configs[1][0]]["c1_50"].append(abs(pipe_pos_czt.process(raw_50)["hr_bpm"] - gt_hr))
        errs[configs[1][0]]["c1_10"].append(abs(pipe_pos_czt.process(raw_10)["hr_bpm"] - gt_hr))
        errs[configs[1][0]]["c7"].append(abs(pipe_pos_czt.process(raw_c7)["hr_bpm"] - gt_hr))

        # 3. Locked R4 Sub-ROI + Mask Gate (CHROM + CZT)
        errs[configs[2][0]]["base"].append(abs(pipe_chr_czt.process(ext_base["rgb_signals"])["hr_bpm"] - gt_hr))
        errs[configs[2][0]]["c1_50"].append(abs(pipe_chr_czt.process(ext_50["rgb_signals"])["hr_bpm"] - gt_hr))
        errs[configs[2][0]]["c1_10"].append(abs(pipe_chr_czt.process(ext_10["rgb_signals"])["hr_bpm"] - gt_hr))
        errs[configs[2][0]]["c7"].append(abs(pipe_chr_czt.process(ext_c7["rgb_signals"])["hr_bpm"] - gt_hr))

        # 4. Pipeline v2 (ME-rPPG State-Space)
        errs[configs[3][0]]["base"].append(abs(me_engine.process_frames(ext_base["cropped_faces"])["hr_bpm"] - gt_hr))
        errs[configs[3][0]]["c1_50"].append(abs(me_engine.process_frames(ext_50["cropped_faces"])["hr_bpm"] - gt_hr))
        errs[configs[3][0]]["c1_10"].append(abs(me_engine.process_frames(ext_10["cropped_faces"])["hr_bpm"] - gt_hr))
        errs[configs[3][0]]["c7"].append(abs(me_engine.process_frames(ext_c7["cropped_faces"])["hr_bpm"] - gt_hr))

        # 5. + FreqPhys PFD (Fundamental-Verified)
        errs[configs[4][0]]["base"].append(abs(hyb_pfd.process(ext_base["cropped_faces"], rgb_signals=ext_base["rgb_signals"])["hr_bpm"] - gt_hr))
        errs[configs[4][0]]["c1_50"].append(abs(hyb_pfd.process(ext_50["cropped_faces"], rgb_signals=ext_50["rgb_signals"])["hr_bpm"] - gt_hr))
        errs[configs[4][0]]["c1_10"].append(abs(hyb_pfd.process(ext_10["cropped_faces"], rgb_signals=ext_10["rgb_signals"])["hr_bpm"] - gt_hr))
        errs[configs[4][0]]["c7"].append(abs(hyb_pfd.process(ext_c7["cropped_faces"], rgb_signals=ext_c7["rgb_signals"])["hr_bpm"] - gt_hr))

        # 6. Pipeline v3/v4 (Full Majority-Consensus Hybrid)
        errs[configs[5][0]]["base"].append(abs(hyb_full.process(ext_base["cropped_faces"], rgb_signals=ext_base["rgb_signals"])["hr_bpm"] - gt_hr))
        errs[configs[5][0]]["c1_50"].append(abs(hyb_full.process(ext_50["cropped_faces"], rgb_signals=ext_50["rgb_signals"])["hr_bpm"] - gt_hr))
        errs[configs[5][0]]["c1_10"].append(abs(hyb_full.process(ext_10["cropped_faces"], rgb_signals=ext_10["rgb_signals"])["hr_bpm"] - gt_hr))
        errs[configs[5][0]]["c7"].append(abs(hyb_full.process(ext_c7["cropped_faces"], rgb_signals=ext_c7["rgb_signals"])["hr_bpm"] - gt_hr))

    extractor.close()

    rows = []
    for name, spec_est, bbox_flt, denoiser in configs:
        eb = np.array(errs[name]["base"])
        e50 = np.array(errs[name]["c1_50"])
        e10 = np.array(errs[name]["c1_10"])
        ec7 = np.array(errs[name]["c7"])
        rows.append({
            "Configuration": name,
            "Spectral Estimator": spec_est,
            "BBox & Skin Filter": bbox_flt,
            "Denoiser / Attention": denoiser,
            "N": len(cohort),
            "Baseline_MAE_BPM": round(float(np.mean(eb)), 2),
            "Baseline_SD_BPM": round(float(np.std(eb)), 2),
            "C1_50lux_MAE_BPM": round(float(np.mean(e50)), 2),
            "C1_50lux_SD_BPM": round(float(np.std(e50)), 2),
            "C1_10lux_MAE_BPM": round(float(np.mean(e10)), 2),
            "C1_10lux_SD_BPM": round(float(np.std(e10)), 2),
            "C7_Masked_MAE_BPM": round(float(np.mean(ec7)), 2),
            "C7_Masked_SD_BPM": round(float(np.std(ec7)), 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv("artifacts/ablation_matrix.csv", index=False)

    md_lines = [
        f"# Multi-Subject Component Ablation Matrix (N = {len(cohort)} UBFC Subjects — [0.7, 3.0] Hz)",
        "",
        "| Configuration | Spectral Estimator | BBox & Skin Filter | Denoiser / Attention | Baseline MAE ± SD (BPM) | C1 (50 lux) MAE ± SD (BPM) | C1 (10 lux) MAE ± SD (BPM) | C7 (Masked) MAE ± SD (BPM) |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| **{r['Configuration']}** | {r['Spectral Estimator']} | {r['BBox & Skin Filter']} | "
            f"{r['Denoiser / Attention']} | {r['Baseline_MAE_BPM']:.2f} ± {r['Baseline_SD_BPM']:.2f} | "
            f"{r['C1_50lux_MAE_BPM']:.2f} ± {r['C1_50lux_SD_BPM']:.2f} | "
            f"{r['C1_10lux_MAE_BPM']:.2f} ± {r['C1_10lux_SD_BPM']:.2f} | "
            f"{r['C7_Masked_MAE_BPM']:.2f} ± {r['C7_Masked_SD_BPM']:.2f} |"
        )

    with open("artifacts/ablation_matrix.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "\n".join(md_lines))
    print("\nSaved multi-subject ablation tables to artifacts/ablation_matrix.csv and artifacts/ablation_matrix.md")


if __name__ == "__main__":
    main()
