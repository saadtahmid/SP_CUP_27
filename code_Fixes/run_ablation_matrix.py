"""Phase 4 Component Ablation Matrix Runner (scripts/run_ablation_matrix.py).

Isolates each paper-driven upgrade across UBFC, AiPEX (C2), C1 (10 lux), and C7 (Masked Face)
and exports:
  - artifacts/ablation_matrix.csv
  - artifacts/ablation_matrix.md
"""

import os
import cv2
import numpy as np
import pandas as pd

from datasets.ubfc_loader import UBFCDataLoader
from datasets.aipex_loader import AiPEXDatasetLoader
from preprocessing.roi_extractor import ROIExtractor
from models.classical.pipeline import ClassicalRPPGPipeline
from models.deep.merppg_engine import MERPPGEngine
from models.hybrid.pfd_module import PhysiologicalFrequencyDenoiser
from models.hybrid.hybrid_system import HybridRPPGSystem
from estimation.czt_estimator import CZTHRTransformer
from constraints.low_light import simulate_low_light_video
from constraints.mask_occlusion import simulate_masked_video
from scripts.run_constraints import load_real_baseline_frames, estimate_robust_hr_from_bvp


def main():
    os.makedirs("artifacts", exist_ok=True)
    print("=" * 75)
    print("RUNNING PHASE 4 COMPONENT ABLATION MATRIX")
    print("=" * 75)

    # Load baseline real video (subject4)
    frames, fps, gt_hr = load_real_baseline_frames(subject_id="subject4", max_frames=450)
    frames_c1 = simulate_low_light_video(frames, target_lux=10.0, baseline_lux=300.0)

    extractor = ROIExtractor()
    masked_frames = simulate_masked_video(frames, tracker=extractor.tracker)

    ext_base = extractor.process_video(frames, fs=fps)
    ext_c1 = extractor.process_video(frames_c1, fs=fps)
    ext_c7 = extractor.process_video(masked_frames, mask_occlusion_active=False, fs=fps)

    # Raw unfiltered per-frame RGB (for v0 and +CZT ablation rows)
    raw_rgb_base = np.mean(frames[:, :, :, ::-1], axis=(1, 2)).astype(np.float64)
    raw_rgb_c1 = np.mean(frames_c1[:, :, :, ::-1], axis=(1, 2)).astype(np.float64)
    raw_rgb_c7 = np.mean(masked_frames[:, :, :, ::-1], axis=(1, 2)).astype(np.float64)

    pipe_fft = ClassicalRPPGPipeline(method="POS", fs=fps, use_czt=False)
    pipe_czt = ClassicalRPPGPipeline(method="POS", fs=fps, use_czt=True)
    me_engine = MERPPGEngine(fs=fps)
    hyb_no_bf = HybridRPPGSystem(fs=fps, enable_pfd=True, enable_czt=True, enable_beatformer=False)
    hyb_full = HybridRPPGSystem(fs=fps, enable_pfd=True, enable_czt=True, enable_beatformer=True)

    # Evaluate across configurations
    rows = [
        {
            "Configuration": "Baseline v0 (POS + Welch FFT)",
            "Spectral Estimator": "Welch FFT",
            "BBox & Skin Filter": "Raw Per-Frame",
            "Denoiser / Attention": "None (POS)",
            "UBFC Baseline Error (BPM)": round(abs(pipe_fft.process(raw_rgb_base)["hr_bpm"] - gt_hr), 2),
            "C1 (10 lux) Error (BPM)": round(abs(pipe_fft.process(raw_rgb_c1)["hr_bpm"] - gt_hr), 2),
            "C7 (Masked) Error (BPM)": round(abs(pipe_fft.process(raw_rgb_c7)["hr_bpm"] - gt_hr), 2),
        },
        {
            "Configuration": "+ CZT Zooming (BeatFormer CZT)",
            "Spectral Estimator": "512-bin CZT",
            "BBox & Skin Filter": "Raw Per-Frame",
            "Denoiser / Attention": "None (POS)",
            "UBFC Baseline Error (BPM)": round(abs(pipe_czt.process(raw_rgb_base)["hr_bpm"] - gt_hr), 2),
            "C1 (10 lux) Error (BPM)": round(abs(pipe_czt.process(raw_rgb_c1)["hr_bpm"] - gt_hr), 2),
            "C7 (Masked) Error (BPM)": round(abs(pipe_czt.process(raw_rgb_c7)["hr_bpm"] - gt_hr), 2),
        },
        {
            "Configuration": "+ ROI Stabilization & Skin/SNR Filter (frPPG + FSAM)",
            "Spectral Estimator": "512-bin CZT",
            "BBox & Skin Filter": "EMA BBox + YCbCr + NMF/SNR ROI",
            "Denoiser / Attention": "None (POS)",
            "UBFC Baseline Error (BPM)": round(abs(pipe_czt.process(ext_base["rgb_signals"])["hr_bpm"] - gt_hr), 2),
            "C1 (10 lux) Error (BPM)": round(abs(pipe_czt.process(ext_c1["rgb_signals"])["hr_bpm"] - gt_hr), 2),
            "C7 (Masked) Error (BPM)": round(abs(pipe_czt.process(ext_c7["rgb_signals"])["hr_bpm"] - gt_hr), 2),
        },
        {
            "Configuration": "Pipeline v2 (ME-rPPG State-Space)",
            "Spectral Estimator": "512-bin CZT",
            "BBox & Skin Filter": "EMA BBox + 45-Frame RMA Warmup",
            "Denoiser / Attention": "ME-rPPG State-Space",
            "UBFC Baseline Error (BPM)": round(abs(me_engine.process_frames(ext_base["cropped_faces"])["hr_bpm"] - gt_hr), 2),
            "C1 (10 lux) Error (BPM)": round(abs(me_engine.process_frames(ext_c1["cropped_faces"])["hr_bpm"] - gt_hr), 2),
            "C7 (Masked) Error (BPM)": round(abs(me_engine.process_frames(ext_c7["cropped_faces"])["hr_bpm"] - gt_hr), 2),
        },
        {
            "Configuration": "+ FreqPhys PFD (PBF + PSM + ASS)",
            "Spectral Estimator": "512-bin CZT",
            "BBox & Skin Filter": "EMA BBox + 45-Frame RMA Warmup",
            "Denoiser / Attention": "ME-rPPG + PFD",
            "UBFC Baseline Error (BPM)": round(abs(hyb_no_bf.process(ext_base["cropped_faces"], rgb_signals=ext_base["rgb_signals"])["hr_bpm"] - gt_hr), 2),
            "C1 (10 lux) Error (BPM)": round(abs(hyb_no_bf.process(ext_c1["cropped_faces"], rgb_signals=ext_c1["rgb_signals"])["hr_bpm"] - gt_hr), 2),
            "C7 (Masked) Error (BPM)": round(abs(hyb_no_bf.process(ext_c7["cropped_faces"], rgb_signals=ext_c7["rgb_signals"])["hr_bpm"] - gt_hr), 2),
        },
        {
            "Configuration": "Pipeline v3 (Full Dual-Branch Hybrid)",
            "Spectral Estimator": "512-bin CZT",
            "BBox & Skin Filter": "EMA BBox + NMF/SNR ROI",
            "Denoiser / Attention": "ME-rPPG + PFD + BeatFormer-SCL",
            "UBFC Baseline Error (BPM)": round(abs(hyb_full.process(ext_base["cropped_faces"], rgb_signals=ext_base["rgb_signals"])["hr_bpm"] - gt_hr), 2),
            "C1 (10 lux) Error (BPM)": round(abs(hyb_full.process(ext_c1["cropped_faces"], rgb_signals=ext_c1["rgb_signals"])["hr_bpm"] - gt_hr), 2),
            "C7 (Masked) Error (BPM)": round(abs(hyb_full.process(ext_c7["cropped_faces"], rgb_signals=ext_c7["rgb_signals"])["hr_bpm"] - gt_hr), 2),
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv("artifacts/ablation_matrix.csv", index=False)

    md_lines = [
        "# Component Ablation Matrix (2027 IEEE SP Cup — FaceVitals)",
        "",
        "| Configuration | Spectral Estimator | BBox & Skin Filter | Denoiser / Attention | UBFC Baseline Error (BPM) | C1 (10 lux) Error (BPM) | C7 (Masked) Error (BPM) |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| **{r['Configuration']}** | {r['Spectral Estimator']} | {r['BBox & Skin Filter']} | "
            f"{r['Denoiser / Attention']} | {r['UBFC Baseline Error (BPM)']:.2f} | "
            f"{r['C1 (10 lux) Error (BPM)']:.2f} | {r['C7 (Masked) Error (BPM)']:.2f} |"
        )

    with open("artifacts/ablation_matrix.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n".join(md_lines))
    print("\nSaved ablation tables to artifacts/ablation_matrix.csv and artifacts/ablation_matrix.md")


if __name__ == "__main__":
    main()
