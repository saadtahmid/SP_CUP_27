#!/usr/bin/env python3
"""Cross-Dataset Generalization Evaluation Script (PURE and MMPD).

Evaluates the Stage 3C / Stage 4 Majority-Consensus Hybrid Architecture (v0.5.1)
across unseen external benchmark cohorts to test:
1. Motion Robustness on PURE (6 protocols: Steady, Talking, Slow/Fast Translation, Small/Medium Rotation)
2. Demographic & Low-Light Generalization on MMPD (Fitzpatrick III-VI, 4 lighting conditions, 4 activities)

Outputs:
- artifacts/cross_dataset/pure_results.csv
- artifacts/cross_dataset/pure_protocol_summary.csv
- artifacts/cross_dataset/mmpd_results.csv
- artifacts/cross_dataset/mmpd_summary.csv
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import cv2

from preprocessing.roi_extractor import ROIExtractor, filter_skin_pixels, fuse_sub_rois_by_snr
from preprocessing.skin_tone import analyze_skin_tone
from models.classical.pipeline import ClassicalRPPGPipeline
from models.deep.merppg_engine import MERPPGEngine
from models.hybrid.hybrid_system import HybridRPPGSystem
from evaluation.metrics import compute_all_metrics
from datasets.pure_loader import PUREDataLoader, PURE_PROTOCOLS
from datasets.mmpd_loader import MMPDDataLoader
from datasets.synthetic_generator import generate_synthetic_rppg_video


def extract_session_signals(
    frames_bgr: np.ndarray,
    extractor: ROIExtractor,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Extract spatial mean RGB, cropped faces (36x36), and R4 sub-ROIs from BGR frames."""
    data = extractor.process_video(frames_bgr)
    return data["rgb_signals"], data["cropped_faces"], data["sub_roi_rgb"]


def generate_synthetic_pure_session(
    subject_id: str,
    protocol_code: str,
    fps: float = 30.0,
    duration_sec: float = 15.0,
) -> Dict[str, Any]:
    """Generate synthetic PURE session with realistic motion and pulse characteristics."""
    import tempfile
    protocol_name = PURE_PROTOCOLS.get(protocol_code, "Steady")
    base_hr = 65.0 + (int(subject_id.replace("sub", "").replace("subject", "") or "1") * 5) % 45

    motion_levels = {
        "01": 0.00,  # None (Steady)
        "02": 0.05,  # Jaw/Talking
        "03": 0.12,  # Slow Translation
        "04": 0.35,  # Fast Translation
        "05": 0.15,  # Small Rotation
        "06": 0.30,  # Medium Rotation
    }
    motion_amp = motion_levels.get(protocol_code, 0.0)

    with tempfile.TemporaryDirectory() as td:
        tmp_vid = os.path.join(td, "pure_sim.mp4")
        tmp_gt = os.path.join(td, "pure_gt.csv")
        meta = generate_synthetic_rppg_video(
            output_video_path=tmp_vid,
            output_gt_csv_path=tmp_gt,
            duration_sec=duration_sec,
            fps=fps,
            target_hr_bpm=base_hr,
            width=640,
            height=480,
            motion_amplitude=motion_amp,
        )

        cap = cv2.VideoCapture(tmp_vid)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    return {
        "session_id": f"{subject_id}-{protocol_code}",
        "subject_id": subject_id,
        "protocol_code": protocol_code,
        "protocol_name": protocol_name,
        "frames": np.array(frames, dtype=np.uint8),
        "gt_hr": float(base_hr),
        "fps": float(fps),
    }



def evaluate_pure(
    data_dir: str,
    output_dir: Optional[str] = "artifacts/cross_dataset",
    max_subjects: Optional[int] = None,
    max_frames: int = 450,
    synthetic: bool = False,
) -> pd.DataFrame:
    """Evaluate pipeline on PURE dataset across all 6 motion protocols."""
    print(f"\n========================================================")
    print(f"  CROSS-DATASET BENCHMARK: PURE (Pulse Rate Detection)")
    print(f"========================================================")

    loader = PUREDataLoader(data_root=data_dir)
    sessions = loader.sessions

    if len(sessions) == 0 and not synthetic:
        print(f"[!] No PURE sessions found in '{data_dir}'.")
        print(f"    To run with mock/synthetic PURE data for pipeline verification, use --synthetic.")
        return pd.DataFrame()

    extractor = ROIExtractor()
    results = []

    # If running in synthetic mode or fallback
    if synthetic or len(sessions) == 0:
        print(f"[*] Running synthetic PURE benchmark across 6 motion protocols...")
        test_subjects = [f"0{i}" if i < 10 else f"{i}" for i in range(1, 11)]
        if max_subjects:
            test_subjects = test_subjects[:max_subjects]

        eval_queue = []
        for s in test_subjects:
            for p_code in sorted(PURE_PROTOCOLS.keys()):
                eval_queue.append((s, p_code))

        for idx, (s, p_code) in enumerate(eval_queue, 1):
            sess_data = generate_synthetic_pure_session(s, p_code, fps=30.0, duration_sec=max_frames / 30.0)
            frames = sess_data["frames"]
            gt_hr = sess_data["gt_hr"]
            fps = sess_data["fps"]
            p_name = sess_data["protocol_name"]

            fused_rgb, crops, sub_rois = extract_session_signals(frames, extractor)

            pipe_pos = ClassicalRPPGPipeline(method="POS", fs=fps, use_czt=True)
            pipe_chrom = ClassicalRPPGPipeline(method="CHROM", fs=fps, use_czt=True)
            me_engine = MERPPGEngine(fs=fps)
            hybrid_sys = HybridRPPGSystem(fs=fps, enable_pfd=True, enable_czt=True, enable_beatformer=True)

            hr_pos = pipe_pos.process(fused_rgb)["hr_bpm"]
            hr_chrom = pipe_chrom.process(fused_rgb)["hr_bpm"]
            hr_me = me_engine.process_frames(crops)["hr_bpm"]
            res_hyb = hybrid_sys.process(crops, rgb_signals=fused_rgb)
            hr_hyb = res_hyb["hr_bpm"]

            err_pos = abs(hr_pos - gt_hr)
            err_chrom = abs(hr_chrom - gt_hr)
            err_hyb = abs(hr_hyb - gt_hr)

            print(f"  [{idx}/{len(eval_queue)}] {s}-{p_code} ({p_name}): GT={gt_hr:.1f} | POS={hr_pos:.1f} (Δ={err_pos:.2f}) | CHROM={hr_chrom:.1f} (Δ={err_chrom:.2f}) | Hybrid={hr_hyb:.1f} (Δ={err_hyb:.2f})")

            results.append({
                "session_id": f"{s}-{p_code}",
                "subject_id": s,
                "protocol_code": p_code,
                "protocol_name": p_name,
                "gt_hr_bpm": gt_hr,
                "pos_czt_bpm": hr_pos,
                "chrom_czt_bpm": hr_chrom,
                "me_rppg_bpm": hr_me,
                "hybrid_v4_bpm": hr_hyb,
                "pos_error_bpm": err_pos,
                "chrom_error_bpm": err_chrom,
                "hybrid_error_bpm": err_hyb,
                "snr_db": res_hyb.get("snr_db", 0.0),
            })
    else:
        # Real dataset execution
        if max_subjects:
            unique_subjs = sorted(list(set(s["subject_id"] for s in sessions)))[:max_subjects]
            sessions = [s for s in sessions if s["subject_id"] in unique_subjs]

        print(f"[*] Processing {len(sessions)} real PURE sessions...")
        for idx, s_info in enumerate(sessions, 1):
            s_id = s_info["session_id"]
            p_code = s_info["protocol_code"]
            p_name = s_info["protocol_name"]

            frames, fps, time_window = loader.load_session_frames(s_info, max_frames=max_frames, return_time_window=True)
            if len(frames) == 0:
                print(f"  [-] Skipping {s_id}: no frames extracted.")
                continue

            gt_info = loader.parse_pure_json(s_info["json_path"], time_window=time_window)
            gt_ppg = gt_info["czt_hr_bpm"]
            gt_ox = gt_info["mean_hr_bpm"]

            fused_rgb, crops, sub_rois = extract_session_signals(frames, extractor)

            pipe_pos = ClassicalRPPGPipeline(method="POS", fs=fps, use_czt=True)
            pipe_chrom = ClassicalRPPGPipeline(method="CHROM", fs=fps, use_czt=True)
            me_engine = MERPPGEngine(fs=fps)
            hybrid_sys = HybridRPPGSystem(fs=fps, enable_pfd=True, enable_czt=True, enable_beatformer=True)

            hr_pos = pipe_pos.process(fused_rgb)["hr_bpm"]
            hr_chrom = pipe_chrom.process(fused_rgb)["hr_bpm"]
            hr_me_standalone = me_engine.process_frames(crops)["hr_bpm"]
            res_hyb = hybrid_sys.process(crops, rgb_signals=fused_rgb)
            hr_hyb = res_hyb["hr_bpm"]
            hr_me_anchor = res_hyb.get("hr_anchor_bpm", hr_me_standalone)

            err_pos_ppg = abs(hr_pos - gt_ppg)
            err_chrom_ppg = abs(hr_chrom - gt_ppg)
            err_hyb_ppg = abs(hr_hyb - gt_ppg)
            err_hyb_ox = abs(hr_hyb - gt_ox)

            print(f"  [{idx}/{len(sessions)}] {s_id} ({p_name}): PPG GT={gt_ppg:.1f} (Ox={gt_ox:.1f}) | POS={hr_pos:.1f} | CHROM={hr_chrom:.1f} | Hybrid={hr_hyb:.1f} (Δ_ppg={err_hyb_ppg:.2f}, Δ_ox={err_hyb_ox:.2f})")

            results.append({
                "session_id": s_id,
                "subject_id": s_info["subject_id"],
                "protocol_code": p_code,
                "protocol_name": p_name,
                "gt_hr_bpm": gt_ppg,
                "gt_ppg_bpm": gt_ppg,
                "gt_oximeter_bpm": gt_ox,
                "pos_czt_bpm": hr_pos,
                "chrom_czt_bpm": hr_chrom,
                "me_rppg_bpm": hr_me_anchor,
                "me_rppg_raw_bpm": res_hyb.get("hr_me_raw_bpm", hr_me_standalone),
                "hybrid_v4_bpm": hr_hyb,
                "pos_error_bpm": err_pos_ppg,
                "chrom_error_bpm": err_chrom_ppg,
                "hybrid_error_bpm": err_hyb_ppg,
                "pos_error_ppg": err_pos_ppg,
                "chrom_error_ppg": err_chrom_ppg,
                "hybrid_error_ppg": err_hyb_ppg,
                "hybrid_error_ox": err_hyb_ox,
                "snr_db": res_hyb.get("snr_db", 0.0),
            })

    df = pd.DataFrame(results)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "pure_results.csv", index=False)

    # Compute per-protocol summary
    summary_rows = []
    for p_code in sorted(PURE_PROTOCOLS.keys()):
        sub_df = df[df["protocol_code"] == p_code]
        if len(sub_df) == 0:
            continue
        p_name = PURE_PROTOCOLS[p_code]
        gt = sub_df["gt_hr_bpm"].values
        pred_hyb = sub_df["hybrid_v4_bpm"].values
        pred_chrom = sub_df["chrom_czt_bpm"].values
        pred_pos = sub_df["pos_czt_bpm"].values

        m_hyb = compute_all_metrics(pred_hyb, gt)
        m_chrom = compute_all_metrics(pred_chrom, gt)
        m_pos = compute_all_metrics(pred_pos, gt)

        summary_rows.append({
            "protocol_code": p_code,
            "protocol_name": p_name,
            "count": len(sub_df),
            "pos_mae": m_pos["mae"],
            "chrom_mae": m_chrom["mae"],
            "hybrid_mae": m_hyb["mae"],
            "hybrid_medae": m_hyb["medae"],
            "hybrid_rmse": m_hyb["rmse"],
            "hybrid_pearson_r": m_hyb["pearson_r"],
        })

    sum_df = pd.DataFrame(summary_rows)
    if output_dir:
        sum_df.to_csv(out_dir / "pure_protocol_summary.csv", index=False)

    print("\n--------------------------------------------------------------------------------")
    print("PURE MOTION PROTOCOL SUMMARY (Per-Protocol Degradation Profile)")
    print("--------------------------------------------------------------------------------")
    print(sum_df.to_string(index=False))

    pooled_gt = df["gt_hr_bpm"].values
    pooled_hyb = df["hybrid_v4_bpm"].values
    pooled_metrics = compute_all_metrics(pooled_hyb, pooled_gt)
    print("--------------------------------------------------------------------------------")
    print(f"POOLED PURE MAE:   {pooled_metrics['mae']:.2f} BPM  (MedAE: {pooled_metrics['medae']:.2f} BPM, RMSE: {pooled_metrics['rmse']:.2f} BPM, r: {pooled_metrics['pearson_r']:.4f})")
    print("--------------------------------------------------------------------------------\n")

    return df


def evaluate_mmpd(
    data_dir: str = "data/MMPD",
    output_dir: Optional[str] = "artifacts/cross_dataset",
    max_subjects: Optional[int] = None,
    max_frames: int = 450,
) -> pd.DataFrame:
    """Benchmark FaceVitals Hybrid v4 and baseline classical methods on real MMPD recordings."""
    print("================================================================================")
    print("MMPD CROSS-DATASET BENCHMARK HARNESS (Skin Tone, Motion & Lighting Generalization)")
    print("================================================================================")

    loader = MMPDDataLoader(data_root=data_dir)
    recordings = loader.recordings
    if len(recordings) == 0:
        print(f"[!] No MMPD recordings found in {data_dir}.")
        return pd.DataFrame()

    if max_subjects:
        unique_subjs = sorted(list(set(r["recording_id"].split("_")[0] for r in recordings)))[:max_subjects]
        recordings = [r for r in recordings if r["recording_id"].split("_")[0] in unique_subjs]

    print(f"[*] Found {len(recordings)} MMPD recordings to evaluate...")
    extractor = ROIExtractor()
    results = []

    for idx, rec in enumerate(recordings, 1):
        rec_id = rec["recording_id"]
        subj_id = rec_id.split("_")[0]
        meta = loader.parse_mmpd_mat(rec["mat_path"])
        light = meta.get("light", "Unknown")
        motion = meta.get("motion", "Unknown")
        skin_color = str(meta.get("skin", "Unknown"))
        exercise = meta.get("exercise", "False")

        frames, fps, gt_hr = loader.load_recording_frames(rec, max_frames=max_frames)
        if len(frames) == 0:
            print(f"  [-] Skipping {rec_id}: no frames extracted.")
            continue

        fused_rgb, crops, sub_rois = extract_session_signals(frames, extractor)

        pipe_pos = ClassicalRPPGPipeline(method="POS", fs=fps, use_czt=True)
        pipe_chrom = ClassicalRPPGPipeline(method="CHROM", fs=fps, use_czt=True)
        me_engine = MERPPGEngine(fs=fps)
        hybrid_sys = HybridRPPGSystem(fs=fps, enable_pfd=True, enable_czt=True, enable_beatformer=True)

        hr_pos = pipe_pos.process(fused_rgb)["hr_bpm"]
        hr_chrom = pipe_chrom.process(fused_rgb)["hr_bpm"]
        hr_me_standalone = me_engine.process_frames(crops)["hr_bpm"]
        res_hyb = hybrid_sys.process(crops, rgb_signals=fused_rgb)
        hr_hyb = res_hyb["hr_bpm"]
        hr_me_anchor = res_hyb.get("hr_anchor_bpm", hr_me_standalone)
        hr_me_raw = res_hyb.get("hr_me_raw_bpm", hr_me_standalone)

        err_pos = abs(hr_pos - gt_hr)
        err_chrom = abs(hr_chrom - gt_hr)
        err_me = abs(hr_me_anchor - gt_hr)
        err_hyb = abs(hr_hyb - gt_hr)

        print(f"  [{idx}/{len(recordings)}] {rec_id} (Skin {skin_color}, {light}, {motion}): GT={gt_hr:.1f} | POS={hr_pos:.1f} (Δ={err_pos:.2f}) | CHROM={hr_chrom:.1f} (Δ={err_chrom:.2f}) | ME={hr_me_anchor:.1f} (Δ={err_me:.2f}) | Hybrid={hr_hyb:.1f} (Δ={err_hyb:.2f})")

        results.append({
            "recording_id": rec_id,
            "subject_id": subj_id,
            "skin_type": skin_color,
            "light": light,
            "motion": motion,
            "exercise": exercise,
            "gt_hr_bpm": gt_hr,
            "gt_mat_scalar_bpm": meta.get("gt_mat_scalar_bpm", gt_hr),
            "pos_czt_bpm": hr_pos,
            "chrom_czt_bpm": hr_chrom,
            "me_rppg_raw_bpm": hr_me_raw,
            "me_rppg_anchor_bpm": hr_me_anchor,
            "me_rppg_bpm": hr_me_anchor,
            "hybrid_v4_bpm": hr_hyb,
            "pos_error_bpm": err_pos,
            "chrom_error_bpm": err_chrom,
            "me_error_bpm": err_me,
            "hybrid_error_bpm": err_hyb,
            "snr_db": res_hyb.get("snr_db", 0.0),
        })

    df = pd.DataFrame(results)
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "mmpd_results.csv", index=False)

    # Separate into Operational (N=64, non-exercise) and Post-Exercise (N=16) cohorts
    df_op = df[df["motion"] != "Stationary (after exercise)"]
    df_ex = df[df["motion"] == "Stationary (after exercise)"]

    summary_rows = []

    # 1. Operational Trials by Skin Type (N=64)
    for st in sorted(df_op["skin_type"].unique()):
        sub_df = df_op[df_op["skin_type"] == st]
        m_pos = compute_all_metrics(sub_df["pos_czt_bpm"].values, sub_df["gt_hr_bpm"].values)
        m_chrom = compute_all_metrics(sub_df["chrom_czt_bpm"].values, sub_df["gt_hr_bpm"].values)
        m_me = compute_all_metrics(sub_df["me_rppg_bpm"].values, sub_df["gt_hr_bpm"].values)
        m_hyb = compute_all_metrics(sub_df["hybrid_v4_bpm"].values, sub_df["gt_hr_bpm"].values)
        cond_label = f"Fitzpatrick {st.replace('Type ', '')}" if "Type" in st else f"Fitzpatrick {st}"
        summary_rows.append({
            "cohort": "Operational",
            "dimension": "Skin Tone",
            "condition": cond_label,
            "count": len(sub_df),
            "pos_mae": m_pos["mae"],
            "chrom_mae": m_chrom["mae"],
            "me_mae": m_me["mae"],
            "hybrid_mae": m_hyb["mae"],
            "hybrid_medae": m_hyb["medae"],
            "hybrid_rmse": m_hyb["rmse"],
            "hybrid_pearson_r": m_hyb["pearson_r"],
        })

    # 2. Operational Trials by Lighting Condition (N=64)
    for lt in sorted(df_op["light"].unique()):
        sub_df = df_op[df_op["light"] == lt]
        m_pos = compute_all_metrics(sub_df["pos_czt_bpm"].values, sub_df["gt_hr_bpm"].values)
        m_chrom = compute_all_metrics(sub_df["chrom_czt_bpm"].values, sub_df["gt_hr_bpm"].values)
        m_me = compute_all_metrics(sub_df["me_rppg_bpm"].values, sub_df["gt_hr_bpm"].values)
        m_hyb = compute_all_metrics(sub_df["hybrid_v4_bpm"].values, sub_df["gt_hr_bpm"].values)
        summary_rows.append({
            "cohort": "Operational",
            "dimension": "Lighting",
            "condition": lt,
            "count": len(sub_df),
            "pos_mae": m_pos["mae"],
            "chrom_mae": m_chrom["mae"],
            "me_mae": m_me["mae"],
            "hybrid_mae": m_hyb["mae"],
            "hybrid_medae": m_hyb["medae"],
            "hybrid_rmse": m_hyb["rmse"],
            "hybrid_pearson_r": m_hyb["pearson_r"],
        })

    # 3. Operational Trials by Motion Condition (N=64)
    for mt in sorted(df_op["motion"].unique()):
        sub_df = df_op[df_op["motion"] == mt]
        m_pos = compute_all_metrics(sub_df["pos_czt_bpm"].values, sub_df["gt_hr_bpm"].values)
        m_chrom = compute_all_metrics(sub_df["chrom_czt_bpm"].values, sub_df["gt_hr_bpm"].values)
        m_me = compute_all_metrics(sub_df["me_rppg_bpm"].values, sub_df["gt_hr_bpm"].values)
        m_hyb = compute_all_metrics(sub_df["hybrid_v4_bpm"].values, sub_df["gt_hr_bpm"].values)
        summary_rows.append({
            "cohort": "Operational",
            "dimension": "Motion",
            "condition": mt,
            "count": len(sub_df),
            "pos_mae": m_pos["mae"],
            "chrom_mae": m_chrom["mae"],
            "me_mae": m_me["mae"],
            "hybrid_mae": m_hyb["mae"],
            "hybrid_medae": m_hyb["medae"],
            "hybrid_rmse": m_hyb["rmse"],
            "hybrid_pearson_r": m_hyb["pearson_r"],
        })

    # 4. Operational Trials Pooled (N=64)
    if len(df_op) > 0:
        m_pos = compute_all_metrics(df_op["pos_czt_bpm"].values, df_op["gt_hr_bpm"].values)
        m_chrom = compute_all_metrics(df_op["chrom_czt_bpm"].values, df_op["gt_hr_bpm"].values)
        m_me = compute_all_metrics(df_op["me_rppg_bpm"].values, df_op["gt_hr_bpm"].values)
        m_hyb = compute_all_metrics(df_op["hybrid_v4_bpm"].values, df_op["gt_hr_bpm"].values)
        summary_rows.append({
            "cohort": "Operational",
            "dimension": "Pooled",
            "condition": f"All {len(df_op)} Operational Trials",
            "count": len(df_op),
            "pos_mae": m_pos["mae"],
            "chrom_mae": m_chrom["mae"],
            "me_mae": m_me["mae"],
            "hybrid_mae": m_hyb["mae"],
            "hybrid_medae": m_hyb["medae"],
            "hybrid_rmse": m_hyb["rmse"],
            "hybrid_pearson_r": m_hyb["pearson_r"],
        })

    # 5. Non-Stationary Recovery Trials (N=16, after exercise)
    if len(df_ex) > 0:
        for st in sorted(df_ex["skin_type"].unique()):
            sub_df = df_ex[df_ex["skin_type"] == st]
            m_pos = compute_all_metrics(sub_df["pos_czt_bpm"].values, sub_df["gt_hr_bpm"].values)
            m_chrom = compute_all_metrics(sub_df["chrom_czt_bpm"].values, sub_df["gt_hr_bpm"].values)
            m_me = compute_all_metrics(sub_df["me_rppg_bpm"].values, sub_df["gt_hr_bpm"].values)
            m_hyb = compute_all_metrics(sub_df["hybrid_v4_bpm"].values, sub_df["gt_hr_bpm"].values)
            cond_label = f"Fitzpatrick {st.replace('Type ', '')}" if "Type" in st else f"Fitzpatrick {st}"
            summary_rows.append({
                "cohort": "Post-Exercise",
                "dimension": "Skin Tone",
                "condition": cond_label,
                "count": len(sub_df),
                "pos_mae": m_pos["mae"],
                "chrom_mae": m_chrom["mae"],
                "me_mae": m_me["mae"],
                "hybrid_mae": m_hyb["mae"],
                "hybrid_medae": m_hyb["medae"],
                "hybrid_rmse": m_hyb["rmse"],
                "hybrid_pearson_r": m_hyb["pearson_r"],
            })

        m_pos = compute_all_metrics(df_ex["pos_czt_bpm"].values, df_ex["gt_hr_bpm"].values)
        m_chrom = compute_all_metrics(df_ex["chrom_czt_bpm"].values, df_ex["gt_hr_bpm"].values)
        m_me = compute_all_metrics(df_ex["me_rppg_bpm"].values, df_ex["gt_hr_bpm"].values)
        m_hyb = compute_all_metrics(df_ex["hybrid_v4_bpm"].values, df_ex["gt_hr_bpm"].values)
        summary_rows.append({
            "cohort": "Post-Exercise",
            "dimension": "Pooled",
            "condition": f"All {len(df_ex)} Post-Exercise Trials",
            "count": len(df_ex),
            "pos_mae": m_pos["mae"],
            "chrom_mae": m_chrom["mae"],
            "me_mae": m_me["mae"],
            "hybrid_mae": m_hyb["mae"],
            "hybrid_medae": m_hyb["medae"],
            "hybrid_rmse": m_hyb["rmse"],
            "hybrid_pearson_r": m_hyb["pearson_r"],
        })

    # 6. Overall Combined Pooled (N=80)
    m_pos = compute_all_metrics(df["pos_czt_bpm"].values, df["gt_hr_bpm"].values)
    m_chrom = compute_all_metrics(df["chrom_czt_bpm"].values, df["gt_hr_bpm"].values)
    m_me = compute_all_metrics(df["me_rppg_bpm"].values, df["gt_hr_bpm"].values)
    m_hyb = compute_all_metrics(df["hybrid_v4_bpm"].values, df["gt_hr_bpm"].values)
    summary_rows.append({
        "cohort": "Full Benchmark",
        "dimension": "Pooled",
        "condition": f"All {len(df)} Trials Combined",
        "count": len(df),
        "pos_mae": m_pos["mae"],
        "chrom_mae": m_chrom["mae"],
        "me_mae": m_me["mae"],
        "hybrid_mae": m_hyb["mae"],
        "hybrid_medae": m_hyb["medae"],
        "hybrid_rmse": m_hyb["rmse"],
        "hybrid_pearson_r": m_hyb["pearson_r"],
    })

    sum_df = pd.DataFrame(summary_rows)
    if output_dir:
        sum_df.to_csv(out_dir / "mmpd_summary.csv", index=False)
        # Also write separate dedicated operational and post-exercise CSVs
        sum_df[sum_df["cohort"] == "Operational"].to_csv(out_dir / "mmpd_operational_summary.csv", index=False)
        if len(df_ex) > 0:
            sum_df[sum_df["cohort"] == "Post-Exercise"].to_csv(out_dir / "mmpd_exercise_summary.csv", index=False)

    print("\n--------------------------------------------------------------------------------")
    print("MMPD MULTI-DOMAIN GENERALIZATION SUMMARY (OPERATIONAL COHORT: N=64)")
    print("--------------------------------------------------------------------------------")
    print(sum_df[sum_df["cohort"] == "Operational"].to_string(index=False))
    if len(df_ex) > 0:
        print("\n--------------------------------------------------------------------------------")
        print("MMPD NON-STATIONARY POST-EXERCISE RECOVERY SUMMARY (N=16)")
        print("--------------------------------------------------------------------------------")
        print(sum_df[sum_df["cohort"] == "Post-Exercise"].to_string(index=False))
    print("--------------------------------------------------------------------------------\n")
    return df


def main():
    parser = argparse.ArgumentParser(description="Cross-Dataset Benchmark Harness (PURE and MMPD)")
    parser.add_argument("--dataset", type=str, choices=["pure", "mmpd", "both"], default="pure",
                        help="Target dataset to evaluate ('pure', 'mmpd', 'both')")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to dataset root folder (default: data/PURE or data/MMPD)")
    parser.add_argument("--output_dir", type=str, default="artifacts/cross_dataset",
                        help="Destination directory for summary CSVs")
    parser.add_argument("--max_subjects", type=int, default=None,
                        help="Optional limit on number of subjects")
    parser.add_argument("--max_frames", type=int, default=450,
                        help="Max frames per video sequence (default: 450 = 15s)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run synthetic evaluation if raw dataset is not downloaded")
    args = parser.parse_args()

    if args.dataset in ["pure", "both"]:
        pure_dir = args.data_dir if (args.data_dir and args.dataset == "pure") else "data/PURE"
        evaluate_pure(
            data_dir=pure_dir,
            output_dir=args.output_dir,
            max_subjects=args.max_subjects,
            max_frames=args.max_frames,
            synthetic=args.synthetic,
        )

    if args.dataset in ["mmpd", "both"]:
        mmpd_dir = args.data_dir if (args.data_dir and args.dataset == "mmpd") else "data/MMPD"
        evaluate_mmpd(
            data_dir=mmpd_dir,
            output_dir=args.output_dir,
            max_subjects=args.max_subjects,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
