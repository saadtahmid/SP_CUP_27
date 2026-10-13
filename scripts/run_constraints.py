"""Run comprehensive evaluation across competition constraints C1, C2, C4, C7, C5 (§2.2).

Produces:
- C1: Low-light degradation curve (300 -> 50 -> 25 -> 10 lux)
- C2: Skin tone ITA stratification & fairness evaluation across AiPEX subjects
- C4: Low-resolution evaluation (HD -> VGA -> QVGA)
- C7: Masked-face occlusion robustness test
- C5: Frame-rate decimation test (30 fps -> 10 fps)
- Publication-quality Bland-Altman plots saved to artifacts/
"""

import os
import numpy as np
import pandas as pd

from datasets.synthetic_generator import generate_synthetic_rppg_video
from datasets.aipex_loader import AiPEXDatasetLoader
from preprocessing.roi_extractor import ROIExtractor
from preprocessing.skin_tone import analyze_skin_tone
from models.classical.pipeline import ClassicalRPPGPipeline
from models.deep.merppg_engine import MERPPGEngine
from models.hybrid.hybrid_system import HybridRPPGSystem
from constraints.low_light import simulate_low_light_video
from constraints.low_res import simulate_low_res_video, apply_frame_decimation_c5
from constraints.mask_occlusion import simulate_masked_video
from evaluation.metrics import compute_all_metrics
from evaluation.bland_altman import plot_bland_altman
from evaluation.fairness import evaluate_skin_tone_fairness, generate_fairness_markdown_table


def main():
    os.makedirs("artifacts/constraints", exist_ok=True)
    print("=" * 70)
    print("2027 IEEE SIGNAL PROCESSING CUP: CONSTRAINT EVALUATION SUITE")
    print("=" * 70)

    # =========================================================================
    # 1. C2: Skin Tone Evaluation on Real AiPEX Dataset (India & Sierra Leone)
    # =========================================================================
    print("\n--- [Constraint C2] Diverse Skin Tone Generalization on AiPEX ---")
    aipex_loader = AiPEXDatasetLoader(base_dir="third_party/rppg_biases")
    subjects = aipex_loader.subjects
    print(f"Total discovered AiPEX subjects: {len(subjects)}")

    pipeline_pos = ClassicalRPPGPipeline(method="POS", fs=15.0, use_czt=True)
    preds = []
    refs = []
    fitz_types = []
    itas = []

    # Run on a representative diverse cohort of subjects from India and Sierra Leone
    eval_subset = subjects[:24]  # 12 India + 12 Sierra Leone
    for s_info in eval_subset:
        data = aipex_loader.load_subject_rgb_traces(s_info)
        gt_hr = data["gt_hr_bpm"]
        rgb = data["rgb_signals"]
        st = data["skin_tone"]

        res = pipeline_pos.process(rgb)
        est_hr = res["hr_bpm"]

        preds.append(est_hr)
        refs.append(gt_hr)
        fitz_types.append(st["fitzpatrick_type"])
        itas.append(st["ita_deg"])

    preds = np.array(preds)
    refs = np.array(refs)

    c2_fairness = evaluate_skin_tone_fairness(preds, refs, fitz_types, tolerance_bpm=3.5)
    print("\nSkin Tone Stratification Results:")
    print(generate_fairness_markdown_table(c2_fairness))
    print(f"Tolerance Compliance: {c2_fairness['tolerance_met']}")

    # Save Bland-Altman for C2
    plot_bland_altman(
        preds, refs,
        output_path="artifacts/constraints/bland_altman_c2_skin_tone.png",
        title="Bland-Altman Agreement Across Diverse Skin Tones (C2 - AiPEX)",
    )

    # =========================================================================
    # 2. C1: Low Light (<= 50 lux) Degradation Curve
    # =========================================================================
    print("\n--- [Constraint C1] Low-Light Operation Degradation Curve ---")
    lux_levels = [300.0, 50.0, 25.0, 10.0]
    c1_results = []
    target_hr_c1 = 75.0

    for lux in lux_levels:
        v_path = f"artifacts/constraints/c1_lux_{int(lux)}.mp4"
        gt_path = f"artifacts/constraints/c1_lux_{int(lux)}_gt.csv"
        generate_synthetic_rppg_video(
            output_video_path=v_path,
            output_gt_csv_path=gt_path,
            duration_sec=6.0,
            fps=30.0,
            target_hr_bpm=target_hr_c1,
            lux=lux,
        )

        extractor = ROIExtractor()
        extracted = extractor.process_video(v_path)
        rgb = extracted["rgb_signals"]
        c_faces = extracted["cropped_faces"]

        # Test Classical POS vs Hybrid (ME-rPPG + PFD)
        p_res = pipeline_pos.process(rgb)
        err_pos = abs(p_res["hr_bpm"] - target_hr_c1)

        hybrid = HybridRPPGSystem(fs=30.0, enable_pfd=True, enable_czt=True)
        h_res = hybrid.process(c_faces)
        err_hybrid = abs(h_res["hr_bpm"] - target_hr_c1)

        c1_results.append({
            "lux": lux,
            "error_pos_bpm": round(err_pos, 2),
            "error_hybrid_pfd_bpm": round(err_hybrid, 2),
        })

    c1_df = pd.DataFrame(c1_results)
    print(c1_df.to_string(index=False))
    c1_df.to_csv("artifacts/constraints/c1_degradation_curve.csv", index=False)

    # =========================================================================
    # 3. C4: Low Resolution (HD 1280x720 -> VGA 640x480 -> QVGA 320x240)
    # =========================================================================
    print("\n--- [Constraint C4] Low Camera Resolution Robustness ---")
    c4_results = []
    target_hr_c4 = 82.0
    for res_name, (w, h) in [("HD (1280x720)", (1280, 720)), ("VGA (640x480)", (640, 480)), ("QVGA (320x240)", (320, 240))]:
        v_path = f"artifacts/constraints/c4_{res_name.split()[0]}.mp4"
        gt_path = f"artifacts/constraints/c4_{res_name.split()[0]}_gt.csv"
        generate_synthetic_rppg_video(
            output_video_path=v_path,
            output_gt_csv_path=gt_path,
            duration_sec=6.0,
            fps=30.0,
            target_hr_bpm=target_hr_c4,
            width=w,
            height=h,
        )

        extractor = ROIExtractor()
        extracted = extractor.process_video(v_path)
        p_res = pipeline_pos.process(extracted["rgb_signals"])
        err = abs(p_res["hr_bpm"] - target_hr_c4)
        c4_results.append({"resolution": res_name, "estimated_hr": p_res["hr_bpm"], "error_bpm": round(err, 2)})

    c4_df = pd.DataFrame(c4_results)
    print(c4_df.to_string(index=False))

    # =========================================================================
    # 4. C7: Masked-Face Operation (>= 40% Occlusion)
    # =========================================================================
    print("\n--- [Constraint C7] Masked-Face Operation (Surgical Mask) ---")
    v_clean = "artifacts/constraints/c7_clean.mp4"
    gt_clean = "artifacts/constraints/c7_clean_gt.csv"
    v_masked = "artifacts/constraints/c7_masked.mp4"
    gt_masked = "artifacts/constraints/c7_masked_gt.csv"

    target_hr_c7 = 88.0
    generate_synthetic_rppg_video(v_clean, gt_clean, duration_sec=6.0, target_hr_bpm=target_hr_c7, has_mask=False)
    generate_synthetic_rppg_video(v_masked, gt_masked, duration_sec=6.0, target_hr_bpm=target_hr_c7, has_mask=True)

    extractor = ROIExtractor()
    ext_clean = extractor.process_video(v_clean, mask_occlusion_active=False)
    ext_masked = extractor.process_video(v_masked, mask_occlusion_active=True)

    res_clean = pipeline_pos.process(ext_clean["rgb_signals"])
    res_masked = pipeline_pos.process(ext_masked["rgb_signals"])

    print(f"Clean Face - Estimated HR:  {res_clean['hr_bpm']:.2f} BPM (Error: {abs(res_clean['hr_bpm'] - target_hr_c7):.2f})")
    print(f"Masked Face - Estimated HR: {res_masked['hr_bpm']:.2f} BPM (Error: {abs(res_masked['hr_bpm'] - target_hr_c7):.2f})")

    # =========================================================================
    # 5. C5: Frame-Rate Decimation (30 fps -> 10 fps)
    # =========================================================================
    print("\n--- [Constraint C5] Frame-Rate Decimation to 10 fps ---")
    frames_30 = ext_clean["frames"]
    frames_10, eff_fps = apply_frame_decimation_c5(frames_30, original_fps=30.0, target_fps=10.0)
    print(f"Decimated from {len(frames_30)} frames to {len(frames_10)} frames (Effective FPS: {eff_fps:.1f})")

    ext_10 = extractor.process_video(frames_10)
    pipeline_10 = ClassicalRPPGPipeline(method="POS", fs=eff_fps, use_czt=True)
    res_10 = pipeline_10.process(ext_10["rgb_signals"])
    print(f"10 fps Extracted HR: {res_10['hr_bpm']:.2f} BPM (Error: {abs(res_10['hr_bpm'] - target_hr_c7):.2f} BPM)")

    print("\nAll constraint benchmarks executed successfully!")


if __name__ == "__main__":
    main()
