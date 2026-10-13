# FaceVitals — Post-Review Patch & Execution Guide for the Coding Agent

This guide provides the exact file mapping and verification steps to resolve the 6 issues identified during the code and artifact audit of commit `cac2d87` (`PROJECT_UPDATE.md`).

## 1. File Replacement Map

Copy each provided Python file into its corresponding path in the repository:

| Provided File | Target Path in Repository | Key Fixes Included |
| :--- | :--- | :--- |
| `hybrid_system.py` | `models/hybrid/hybrid_system.py` | Fixes the critical Hz-vs-BPM bug on Line 106 (`freqs_bf * 60.0`), adds POS+PFD chrominance anchor fusion, and adds parabolic sub-bin interpolation on the fused spectrum |
| `roi_extractor.py` | `preprocessing/roi_extractor.py` | Adds `rank1_nmf_attention` (`FactorizePhys` `FSAM`, Task 2.3), tracks YCbCr `skin_ratio` per sub-ROI so `C7` masked cheeks get $w pprox 0.0$ and forehead gets $w pprox 1.0$, lowers dark pixel cutoff to `3` for `C1` 10-lux frames, and passes `fs` into `process_video` for `C5` 10-fps decimation |
| `beatformer.py` | `models/hybrid/beatformer.py` | Replaces `torch.sigmoid(energy_z / energy_f)` in `ESDBlock` with zero-anchored `energy_z / (energy_f + energy_z + 1e-8)` and implements `Min-Max-Norm` 8-term complex attention in `ZOCABlock` |
| `aipex_loader.py` | `datasets/aipex_loader.py` | Applies `filter_skin_pixels` per frame while streaming AiPEX AVI videos and fuses sub-ROIs via `fuse_sub_rois_by_snr` |
| `run_constraints.py` | `scripts/run_constraints.py` | Evaluates 600 frames of real UBFC video, passes `fs=eff_fps` in `C5` (10 fps), uses `HybridRPPGSystem` on `C5` 400 kbps compression, passes `rgb_signals` in `C3`, and applies `PFD` + windowed median HR on `C2` AiPEX |
| `run_ubfc.py` | `scripts/run_ubfc.py` | Benchmarks all 5 methods (`POS+FFT`, `POS+CZT`, `CHROM+CZT`, `ME-rPPG`, `Hybrid_v3`) on UBFC videos with `rgb_signals` passed to `Hybrid_v3`, saving `artifacts/ubfc/ubfc_ablation_summary.csv` |
| `run_ablation_matrix.py` | `scripts/run_ablation_matrix.py` | **New file (Phase 4)**: Runs the 6-row component ablation across UBFC, `C1` (10 lux), and `C7` (Masked Face), saving `artifacts/ablation_matrix.csv` and `artifacts/ablation_matrix.md` |
| `package_submission.py` | `scripts/package_submission.py` | Defaults to real UBFC subject video (`subject20`) instead of synthetic cartoon ellipse when `datasets/ubfc/` videos exist on disk |

## 2. Execution & Verification Sequence

Run the following commands in order from the repository root:

```bash
# Step 1: Retrain BeatFormer with the zero-anchored ESDBlock and 8-term ZOCA
PYTHONPATH=. python scripts/train_beatformer_scl.py

# Step 2: Run the unit and submission packaging test suite
PYTHONPATH=. pytest -v tests/

# Step 3: Re-run all 7 constraints (C1-C7) including recomputing C2 with skin filtering + PFD
PYTHONPATH=. python scripts/run_constraints.py --recompute-c2

# Step 4: Run the Phase 4 UBFC multi-method benchmark and ablation matrix
PYTHONPATH=. python scripts/run_ubfc.py
PYTHONPATH=. python scripts/run_ablation_matrix.py

# Step 5: Re-package the Round 1 submission bundle using real UBFC video
PYTHONPATH=. python scripts/package_submission.py --subject subject20
```

## 3. Post-Run Verification Checks

Before committing the updated artifacts, verify that:
1. `artifacts/constraints/c7_mask_summary.csv` shows `forehead` weight $> 0.85$ and masked cheek weights $< 0.15$, with `hybrid_err < 2.5 BPM`.
2. `artifacts/constraints/c1_degradation_curve.csv` shows `err_hybrid_bpm < 3.0 BPM` across `300, 50, 25, 10 lux`.
3. `artifacts/constraints/c5_decimation_summary.csv` shows `err_bpm < 3.0 BPM` for both 10 fps frame decimation and 400 kbps MPEG-4 compression.
4. `artifacts/ubfc/ubfc_ablation_summary.csv` and `artifacts/ablation_matrix.md` exist and are non-empty.
5. `submission/Round1_Baseline/target_video.mp4` and `Round1_Baseline_None.mat` contain the real HD video of `subject20` (not the synthetic cartoon ellipse) with `pearson_r > 0.90`.
