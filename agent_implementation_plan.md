# FaceVitals (2027 IEEE SP Cup) — Agent Implementation & Upgrade Plan

## 1. Objective & Ground Rules for the Coding Agent

This specification provides a step-by-step implementation plan to fix existing pipeline bugs, close the gap between the codebase and the 8 reference papers in `papers/`, expand constraint coverage from 5 to all 7 competition constraints (`C1`–`C7`), and generate verified artifacts for the Round 1 IEEE submission.

### Core Execution Rules

1. **Maintain Backward Compatibility**: All existing unit and packaging tests in `tests/test_components.py` and `tests/test_submission_packaging.py` must continue to pass (`PYTHONPATH=. pytest tests/`), and new unit tests must be added for every upgraded module.

2. **Real Video First, Synthetic Only as Fallback**: Always run benchmarks and constraint simulations (`C1`–`C7`) on real facial videos (`datasets/ubfc/` and `third_party/rppg_biases/`) whenever present on disk; use `datasets/synthetic_generator.py` only when `--synthetic` is explicitly passed or during fast unit tests.

3. **Competition Band & Metrics Compliance**: Strictly enforce the official physiological band $[0.7\text{ Hz}, 3.0\text{ Hz}]$ ($42\text{--}180\text{ bpm}$) and compute all four mandated metrics: Mean Absolute Error (`MAE`), Root Mean Square Error (`RMSE`), Pearson correlation (`pearson_r`), and Bland–Altman 95% Limits of Agreement (`bias`, `lower_loa`, `upper_loa`).

## 2. Summary of Current Bugs & Paper-to-Code Gaps

| Priority | File(s) Affected | Issue / Gap Identified | Target Fix & Paper Reference | 
| ----- | ----- | ----- | ----- | 
| **P0 (Bug)** | `scripts/run_constraints.py` | `pipeline_pos` is initialized with `fs=15.0` for AiPEX (Line 42) and reused on `30.0 fps` videos in `C1`, `C4`, and `C7` (Lines 97, 126, 147), shifting all frequencies by $2\times$ and inflating `POS` error to $42.74\text{ bpm}$ | Instantiate separate `fs=15.0` (AiPEX) and `fs=30.0` (UBFC / C1 / C4 / C7) pipelines; run constraints on real UBFC videos | 
| **P0 (Bug)** | `pipeline_packaging/exporter.py`, `scripts/package_submission.py` | `exporter.py` (Line 142) writes constant `estimated_hr_series` into the `bvp_signal` column of `processed_output.csv`; constant HR arrays yield `pearson_r = 0.0` in `Round1_Baseline_None.mat`; `target_video.mp4` is a synthetic cartoon ellipse | Pass actual `bvp` waveform into `exporter.py`; compute sliding-window HR series (`win=8s, stride=1s`) on a real baseline video so `pearson_r` and Bland–Altman LOA are non-degenerate | 
| **P0 (Bug)** | `preprocessing/roi_extractor.py`, `models/deep/merppg_engine.py`, `datasets/ubfc_loader.py` | Per-frame unsmoothed `bbox` causes 1–3 px crop jitter on $36\times 36$ `ME-rPPG` inputs; `ME-rPPG` RMA warm-up transient (first 30–45 frames) is not discarded; `ubfc_loader.py` truncates at `max_frames=600` (20s), yielding `5.91 BPM` MAE vs. `0.47–0.68 BPM` in `merppg.pdf` | Add EMA bounding-box stabilization + $1.2\times$ crop margin; discard/pre-warm 45 frames of RMA state + add `ME-Chunk` linear detrending (`merppg.pdf` Eq. 1); evaluate full video (`max_frames=None`) with windowed HR + outlier rejection | 
| **P1 (Paper Gap)** | `models/hybrid/pfd_module.py` | `PSM` (Lines 54–57) is currently a no-op (`mag = np.abs(X_pbf)`), and `ASS` performs uniform scalar subtraction that leaves the argmax peak unchanged | Implement `FreqPhys` (`freqphys.pdf` Eqs. 3–9) complex spectrum modulation (`PSM`) + harmonic comb reinforcement ($f_0 + 2f_0$) + adaptive threshold selection (`ASS`) | 
| **P1 (Paper Gap)** | `models/hybrid/beatformer.py` (new), `models/hybrid/scl_pretrain.py` | Only post-hoc 1D `CZT` is implemented; `BeatFormer`'s 14.86K-param `ZOCA` + `ESD` network and `SCL` `EMD` loss are missing, and `scl_pretrain.py` is unused | Implement `BeatFormer` (`Martinez_BeatFormer...pdf` Eqs. 4–17) + `RePSS` spectral priors (`paper4.pdf`) for label-free adaptation on degraded videos | 
| **P2 (Paper Gap)** | `preprocessing/roi_extractor.py`, `preprocessing/skin_tone.py` | No specular highlight / non-skin pixel filtering; manual `mask_occlusion_active` flag required for `C7`; `skin_tone.py` misclassifies 29/42 UBFC subjects as `Type I` due to uncalibrated exposure $L^*$ | Add YCbCr/specular pixel filtering (`frPPG`, `review3.pdf`), automatic SNR/NMF multi-ROI weighting (`FactorizePhys`, `paper5.pdf`), and grey-world white-balance normalization before ITA | 
| **P2 (Scope)** | `constraints/` | Only 5 of 7 constraints (`C1, C2, C4, C5-fps, C7`) are implemented; competition FAQ #2 awards grading credit for total constraints addressed | Add `C5` bitrate compression ($\le 500\text{ kbps}$), `C6` aging-skin micro-texture perturbation, and `C3` multi-face Kalman/IoU target isolation | 

## 3. Phase 1: Critical Bug Fixes & Pipeline Stabilization

### Task 1.1: Fix Sampling Rate Bug & Real-Video Evaluation in `scripts/run_constraints.py`

1. **Fix Sampling Rate (`fs`) Bug**:

   * In `scripts/run_constraints.py`, keep `pipeline_pos_15 = ClassicalRPPGPipeline(method="POS", fs=15.0, use_czt=True)` strictly for the 15 fps AiPEX dataset in Section 1 (`C2`).

   * Instantiate `pipeline_pos_30 = ClassicalRPPGPipeline(method="POS", fs=30.0, use_czt=True)` and `pipeline_chrom_30 = ClassicalRPPGPipeline(method="CHROM", fs=30.0, use_czt=True)` for all 30 fps evaluations (`C1`, `C4`, `C7`).

2. **Evaluate All 88 AiPEX Subjects in `C2`**:

   * Instead of `eval_subset = subjects[:24]` (which only samples the India cohort), iterate over all discovered subjects in `aipex_loader.subjects` (or a balanced stratified split of India + Sierra Leone when `--quick` is passed).

   * Report stratified accuracy across Fitzpatrick Types I–VI for both `POS + CZT` and `CHROM + CZT`.

3. **Run `C1`, `C4`, `C5`, and `C7` on Real UBFC Videos**:

   * Load real subject videos from `datasets/ubfc/` (e.g., `subject1`, `subject4`, `subject11`, `subject35`, `subject49` spanning Fitzpatrick Types I–VI) when present on disk, and apply the competition simulators directly to the real video frames:

     * `C1`: `constraints.low_light.simulate_low_light_video(frames, target_lux=lux, noise_mode="combined")` for `lux in [300.0, 50.0, 25.0, 10.0]`.

     * `C4`: `constraints.low_res.simulate_low_res_video(frames, target_resolution=res)` for `HD`, `VGA`, and `QVGA`.

     * `C5`: `constraints.low_res.apply_frame_decimation_c5(frames, original_fps=30.0, target_fps=10.0)` plus bitrate compression ($\le 500\text{ kbps}$).

     * `C7`: `constraints.mask_occlusion.simulate_masked_video(frames, tracker=extractor.tracker)` WITHOUT passing a manual `mask_occlusion_active=True` cheat flag—verify that automatic ROI SNR weighting (Task 1.2) autonomously down-weights the masked lower face.

### Task 1.2: Stabilize Face Crops, Filter Specular Pixels, & Auto-Weight Multi-ROIs in `preprocessing/roi_extractor.py`

1. **Temporal Bounding-Box Stabilization (Crucial for `ME-rPPG`)**:

   * In `ROIExtractor.__init__`, add state variables `self._ema_bbox: Optional[np.ndarray] = None`, `self.bbox_ema_alpha: float = 0.15`, and `self.crop_margin: float = 1.25`.

   * In `process_video()`, reset `self._ema_bbox = None` at the start of each video.

   * For each frame $t$, compute the square bounding box centered on the detected face with scale factor `1.25` (following `RePSS` Team Face AI in `paper4.pdf`). If the IoU or center shift relative to `self._ema_bbox` is small ($< 8\%$ of face width), lock or smooth the bounding box via EMA:
     

     $$
     \text{bbox}_{\text{smooth}}^{(t)} = (1 - \alpha_{\text{box}})\,\text{bbox}_{\text{smooth}}^{(t-1)} + \alpha_{\text{box}}\,\text{bbox}^{(t)}
     $$

   * Slice `face_crop` using the rounded stabilized coordinates before resizing to `(36, 36)` with `cv2.INTER_AREA`. This eliminates high-frequency boundary jitter in `cropped_faces`.

2. **Specular Reflection & Non-Skin Pixel Rejection (`frPPG` & `Debnath & Kim 2025`)**:

   * Inside `extract_from_frame()`, before computing `np.mean(pixels, axis=0)` on an ROI mask, filter `pixels` (RGB uint8) to exclude specular glare and dark shadow/hair outliers:

     * Intensity bounds: keep pixels where $(20 < R < 245) \land (20 < G < 245) \land (20 < B < 245)$.

     * Chrominance skin bounds (in YCbCr): convert ROI pixels to YCbCr and retain pixels within $77 \le C_b \le 135$ and $130 \le C_r \le 180$ (if $>25\%$ of ROI pixels pass; otherwise fall back to intensity-filtered pixels so synthetic masks or extreme low-light frames never produce empty arrays).

3. **Sub-ROI Extraction & Automatic Spectral SNR Fusion (Eliminates Manual `mask_occlusion_active` Flag)**:

   * Extract three separate spatial mean RGB traces per frame: `forehead_rgb`, `left_cheek_rgb`, and `right_cheek_rgb` (shape `(T, 3)` each), stored in `extracted["sub_roi_rgb"] = {"forehead": ..., "left_cheek": ..., "right_cheek": ...}`.

   * Implement a helper `fuse_sub_rois_by_snr(sub_roi_rgb: Dict[str, np.ndarray], fs: float = 30.0) -> Tuple[np.ndarray, Dict[str, float]]`:

     * For each sub-ROI $k \in \{\text{forehead}, \text{left\_cheek}, \text{right\_cheek}\}$:

       1. If temporal variance of the green channel $\text{Var}(G_k) < 10^{-4}$ (e.g., covered by a solid synthetic mask in `C7`) or mean intensity $< 5.0$, assign weight $w_k = 0.0$.

       2. Otherwise, run a quick POS projection on sub-ROI $k$, compute its CZT power spectrum $P_k(f)$ in $[0.7, 3.0]\text{ Hz}$, and compute its spectral concentration (in-band peak-to-average power ratio):
          

          $$
          \text{SNR}_k = \frac{\max_{f \in [0.7, 3.0]} P_k(f)}{\sum_{f \in [0.7, 3.0]} P_k(f) + 10^{-12}}
          $$

     * Normalize weights $w_k = \frac{\text{SNR}_k^2}{\sum_j \text{SNR}_j^2 + 10^{-12}}$ and set `rgb_signals = sum(w_k * sub_roi_rgb[k])`.

     * Keep `mask_occlusion_active: bool = False` in the signature for backward compatibility with existing unit tests, but compute the SNR-weighted fusion by default so `C7` masked faces automatically weight the forehead at $\approx 1.0$ and masked cheeks at $\approx 0.0$.

### Task 1.3: Upgrade `MERPPGEngine` & `UBFCDataLoader` (`merppg.pdf` Section 3.2 & Table 5)

1. **Full-Length Video Evaluation in `datasets/ubfc_loader.py`**:

   * Change `process_subject(self, subject_info, max_frames: Optional[int] = None)` so that by default all frames of `vid.mp4` ($pprox 1800\text{ frames} = 60\text{ s}$) are processed rather than truncating at 600 frames.

   * In `load_ground_truth()`, filter out invalid oximeter dropouts (`hr[hr > 40.0]`) when computing `mean_hr_bpm`.

2. **RMA Warm-Up Handling & `ME-Chunk` Linear Detrending in `models/deep/merppg_engine.py`**:

   * **Pre-Warming State**: In `MERPPGEngine.process_frames(cropped_faces, ...)`, after calling `self.reset_state()`, pre-warm the recurrent state by feeding the first `warmup = min(45, len(cropped_faces))` frames through `self.step(cropped_faces[i])`, then run the full sequence (or replace the first `warmup` outputs with the mean of frames `warmup:warmup+30`) so the initial Recursive Moving Average (`RMA`) step discontinuity does not pollute the CZT spectrum.

   * **Sliding-Window HR Estimation with Outlier Rejection (`frPPG` Section III-B3)**:

     * Add method `estimate_hr_windowed(self, bvp_series: np.ndarray, win_sec: float = 10.0, stride_sec: float = 1.0) -> Tuple[float, np.ndarray]`:

       * If `len(bvp_series) < int(win_sec * self.fs)`, fall back to single-window `self.czt_estimator.estimate_hr(bvp_series)`.

       * Otherwise, compute CZT HR on each overlapping window $w_m$.

       * Apply harmonic & jump de-aliasing across windows: if a window's peak $f_{\text{peak}}$ jumps by $> 15\text{ bpm}$ from the running median and $f_{\text{peak}} / 2$ (or $2 f_{\text{peak}}$) lies within $8\text{ bpm}$ of the running median and has $> 35\%$ of the peak power, select the fundamental peak; discard windows whose spectral SNR (`compute_snr_db`) is in the bottom 15th percentile.

       * Return `float(np.median(valid_window_hrs))` as `hr_bpm`, and interpolate `valid_window_hrs` to frame timestamps `(T,)` as `hr_series`.

### Task 1.4: Fix Submission Bundle Packaging (`pipeline_packaging/exporter.py` & `scripts/package_submission.py`)

1. **Fix `bvp_signal` Column in `pipeline_packaging/exporter.py`**:

   * Update `export_submission_bundle` signature to include `bvp_signal: Optional[np.ndarray] = None`.

   * If `bvp_signal is None`, default to `estimated_hr_series` so existing tests (`test_submission_packaging.py`) pass unchanged.

   * When `bvp_signal` is provided, write `"bvp_signal": np.round(bvp_signal, 5)` to `processed_output.csv`.

2. **Fix `pearson_r = 0.0` and Use Real Baseline Video in `scripts/package_submission.py`**:

   * Check if a real subject video exists in `datasets/ubfc/subject4/vid.mp4` (or any available UBFC subject). If available:

     * Load up to 300–600 frames and resize to $1280 \times 720$ (`cv2.INTER_CUBIC`) to satisfy the §2.3 Task 1 HD baseline specification (`target_video.mp4`).

     * Extract time-varying ground-truth HR (`reference_hr_series`) and BVP from `ground_truth.txt`.

     * Compute sliding-window estimated HR (`estimated_hr_series`, e.g., `win_sec=6.0, stride_sec=0.5` interpolated to `T` frames) and pass `bvp_signal=result["bvp"]` to `export_submission_bundle()`.

   * If only synthetic video is available, modulate the synthetic target HR slightly over time:
     

     $$
     \text{HR}_{\text{gt}}(t) = 75.0 + 2.5 \sin(2\pi \cdot 0.10\,t)
     $$

     
     (simulating natural Respiratory Sinus Arrhythmia, RSA) and compute sliding-window HR estimates so `reference_signal` and `processed_signal` have non-zero variance and yield `pearson_r > 0.90` and valid Bland–Altman LOAs in `Round1_Baseline_None.mat`.

3. **Update `submission/Round1_Baseline/process_round1.m`**:

   * Add 3-point parabolic sub-bin interpolation after `[~, max_idx] = max(power_spec);` and compute a sliding-window HR column `estimated_hr_series` alongside the full-window HR so MATLAB output matches the Python pipeline.

## 4. Phase 2: Paper-Driven Algorithmic Upgrades

### Task 2.1: Upgrade `models/hybrid/pfd_module.py` to Full `FreqPhys` PFD (`freqphys.pdf` Eqs. 1–10)

Replace the no-op `PSM` and static subtraction in `PhysiologicalFrequencyDenoiser` with both a **training-free harmonic-prior spectral denoiser** (for NumPy/ONNX inference) and a **trainable PyTorch `PFDLayer` (`nn.Module`)** (for hybrid training):

1. **Step 1 — Physiological Bandpass Filter (`PBF`, `freqphys.pdf` Eqs. 1–2)**:

   * Compute `X_fft = np.fft.rfft(x_win)` and `freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)`.

   * Retain primary cardiac band $B_1 = [f_{\text{low}}, f_{\text{high}}] = [0.7, 3.0]\text{ Hz}$ and inspect first harmonic band $B_2 = [1.4, 6.0]\text{ Hz}$.

2. **Step 2 — Physiological Spectrum Modulation (`PSM`, `freqphys.pdf` Eqs. 3–5)**:

   * **Harmonic-Prior Complex Modulation (NumPy / Zero-Shot Mode)**:

     * True cardiac signals exhibit phase-locked energy at fundamental $f$ and second harmonic $2f$ (`freqphys.pdf` Fig. 1a & Section 3.1), whereas motion noise and low-light shot noise do not.

     * For each frequency bin $i$ with $f_i \in [0.7, 3.0]\text{ Hz}$, interpolate complex spectrum magnitude at $2 f_i$: $M_2(f_i) = \vert{}X_{\text{fft}}(2 f_i)\vert{}$.

     * Compute normalized modulation gain:
       

       $$
       G(f_i) = 1.0 + \gamma_{\text{harm}} \frac{M_2(f_i)}{\max_{f \in B_1} \vert{}X_{\text{fft}}(f)\vert{} + 10^{-12}}
       $$

       
       (with $\gamma_{\text{harm}} = 0.6$).

     * Additionally, suppress $2\times$ harmonic false peaks (dicrotic notch aliasing): if a candidate peak at $f_{\text{high\_peak}} > 1.8\text{ Hz}$ ($> 108\text{ bpm}$) has a sub-harmonic at $f_{\text{high\_peak}} / 2 \ge 0.75\text{ Hz}$ whose amplitude is at least $0.65 \times \vert{}X(f_{\text{high\_peak}})\vert{}$, boost the fundamental $f_{\text{high\_peak}}/2$ by $1 + \gamma_{\text{sub}}$ so the fundamental dominates.

     * Modulate complex spectrum: $\tilde{X}_F(f_i) = G(f_i) \cdot X_{\text{pbf}}(f_i)$.

   * **Trainable Complex Linear Modulation (`PFDBlock(nn.Module)` in PyTorch)**:

     * Implement exact `freqphys.pdf` Eq. 4 on spectral tensor $Z_F = Z_{\text{re}} + j Z_{\text{im}} \in \mathbb{C}^{B \times F \times D}$:

       ```
       M_re = F.relu(Z_re @ self.W_re - Z_im @ self.W_im + self.B_re)
       M_im = F.relu(Z_re @ self.W_im + Z_im @ self.W_re + self.B_im)
       Z_tilde_re = M_re * Z_re - M_im * Z_im
       Z_tilde_im = M_re * Z_im + M_im * Z_re
       
       ```

3. **Step 3 — Adaptive Spectrum Selection (`ASS`, `freqphys.pdf` Eqs. 7–9)**:

   * Compute per-frequency energy $E[i] = \vert{}\tilde{X}_F[i]\vert{}$ within $B_1 = [0.7, 3.0]\text{ Hz}$.

   * Compute data-driven threshold $\tau = \text{median}(E_{B_1}) + \alpha_{\text{ass}} \cdot \text{MAD}(E_{B_1})$ (or $\mu_{E} + 0.5 \sigma_{E}$, plus learnable `nn.Parameter` $\tau$ with Straight-Through Estimator `m = (E >= tau).float() - torch.sigmoid(E - tau).detach() + torch.sigmoid(E - tau)` in PyTorch).

   * Apply soft/hard spectral selection mask: bins with $E[i] \ge \tau$ retain full energy; bins with $E[i] < \tau$ are attenuated by $\beta = 0.05$:
     

     $$
     \hat{X}_F[i] = \tilde{X}_F[i] \cdot \begin{cases} 1.0 & \text{if } E[i] \ge \tau \\ \beta \cdot (E[i] / (\tau + 10^{-12})) & \text{otherwise} \end{cases}
     $$

   * Transform back via `np.fft.irfft` and apply 4th-order zero-phase Butterworth filter $[0.7, 3.0]\text{ Hz}$.

### Task 2.2: Implement `BeatFormer` (`ZOCA` + `ESD` + `ICZT`) & Unsupervised `SCL` Trainer (`Martinez_BeatFormer...pdf` Eqs. 1–17)

Create `models/hybrid/beatformer.py` and update `models/hybrid/scl_pretrain.py`:

1. **Differentiable `CZTLayer` & `ICZTLayer` (`Eqs. 1–3`)**:

   * Precompute the Vandermonde-diagonal matrix $V_{\text{czt}} \in \mathbb{C}^{M \times L}$ for window length $L = 150$ (5 seconds at 30 fps) and $M = 128$ bins spanning $[0.7, 3.0]\text{ Hz}$:
     

     $$
     A = e^{j 2\pi f_{\text{low}} / f_s}, \quad W = e^{-j 2\pi (f_{\text{high}} - f_{\text{low}}) / ((M - 1) f_s)}, \quad (V_{\text{czt}})_{k, n} = A^{-n} W^{k n}
     $$

   * Store $\text{Re}(V_{\text{czt}})$ and $\text{Im}(V_{\text{czt}})$ as registered PyTorch buffers (`register_buffer`). Forward `CZT` is a fast matrix multiplication $F_{\text{re}} = X \, \text{Re}(V_{\text{czt}})^\top, \; F_{\text{im}} = X \, \text{Im}(V_{\text{czt}})^\top$.

   * Precompute the regularized pseudoinverse reconstruction matrix $V_{\text{iczt}} = \text{Re}(V_{\text{czt}}^\dagger)$ for `ICZT` back to time domain $P' \in \mathbb{R}^{N \times L \times 1}$.

2. **Zoomed Orthonormal Complex Attention (`ZOCA`, `Eqs. 6–10`)**:

   * Add learnable subband positional encoding $E_{\text{pos}} \in \mathbb{R}^{1 \times M \times 3}$ (`Section 3.2.2`).

   * Implement complex linear projections $W_Q, W_K, W_V \in \mathbb{C}^{3 \times 3}$ where **row 0 is fixed (non-trainable)** to $\frac{1}{\sqrt{3}}[1, 1, 1]$ (eliminating common-mode illumination/intensity drift across RGB) and **rows 1 and 2 are learnable** and regularized by the Orthonormal Loss (`Eq. 10`):
     

     $$
     \mathcal{L}_{or} = \sum_{i < j} (a_i \cdot a_j)^2 + \sum_{n=1}^{2} (\Vert{}a_n\Vert{}_2 - 1)^2
     $$

   * Implement `Min-Max-Norm` scaled attention (`Eq. 7`) and the 8-term complex attention expansion (`Eq. 9`).

3. **Energy Spectral Density (`ESD`) Feed-Forward (`Eqs. 11–12`)**:

   * Pass `ZOCA` output $Z$ through a 2-layer MLP with `GELU` activation to obtain $Z'(f)$.

   * Compute energy contribution weight:
     

     $$
     S(f) = \frac{\vert{}Z'(f)\vert{}^2}{\vert{}F(f)\vert{}^2 + 10^{-8}}
     $$

   * Multiply and normalize $F' = \text{LayerNorm}(F \odot S)$, fuse RGB channels to 1D via linear projection, apply `ICZT`, and reconstruct the full temporal BVP waveform via Hanning-windowed overlap-add (`Eq. 12`).

4. **Spectral Contrastive Learning (`SCL`) Loss (`Eqs. 13–17`) + `RePSS` Spectral Sparsity**:

   * In `models/hybrid/scl_pretrain.py`, add:

     * `lab_transform(frame)` and `hsv_transform(frame)` (`BeatFormer` Fig. 2: convert BGR frame to CIE $L^*a^*b^*$ or HSV so chrominance pulse is preserved under altered channel statistics).

     * `temporal_resample(rgb_trace, factor)` (`RePSS` `RankContrast`, `paper4.pdf` Section 3.1.1: resample trace by factor $\alpha \in [0.75, 1.35]$ to create hard negative pairs with shifted heart rates!).

     * `emd_loss(psd_p, psd_t)` (`BeatFormer` Eq. 13): normalize PSDs to probability distributions $p, t$ over the $M$ frequency bins (`p = psd_p / (psd_p.sum(dim=-1, keepdim=True) + 1e-8)`), compute cumulative distribution functions $\text{CDF}(p) = \text{cumsum}(p, \text{dim}=-1)$, and return:
       

       $$
       \mathcal{L}_{\text{EMD}}(p, t) = \frac{1}{M} \sum_{i=1}^{M} (\text{CDF}_i(p) - \text{CDF}_i(t))^2
       $$

     * `scl_triplet_loss(anchor_psd, pos_psds, neg_psds, gamma=1.0)` (`Eqs. 14–16`):
       

       $$
       \mathcal{L}_{\text{SCL}} = \max(0, \mathcal{L}_{\text{pos}} - \mathcal{L}_{\text{neg}} + \gamma)
       $$

     * `spectral_sparsity_loss(psd)` (`RePSS` Team HFUT-VUT, `paper4.pdf` Fig. 3): entropy / Gini sparsity regularizer encouraging energy concentration around a single dominant cardiac peak.

   * Create `scripts/train_beatformer_scl.py` that trains `BeatFormer` (14.8K params, $< 30\text{ seconds}$ on CPU/GPU) on unlabeled UBFC + AiPEX + C1/C7 degraded RGB traces and saves weights to `artifacts/beatformer_scl.pt`.

   * Integrate `BeatFormer` into `models/hybrid/hybrid_system.py` as a dual-branch ensemble with `ME-rPPG + PFD`: combine the normalized CZT power spectra of `ME-rPPG + PFD` (spatial-temporal state-space branch) and `BeatFormer` (chrominance-spectral ZOCA branch) weighted by their respective in-band SNRs (`compute_snr_db`).

### Task 2.3: Add Rank-1 NMF Multi-Patch Attention (`FactorizePhys` `FSAM`, `paper5.pdf` Eqs. 1–9)

In `preprocessing/roi_extractor.py` (or a new module `models/hybrid/fsam_nmf.py`):

* Implement `rank1_nmf_attention(patch_traces: np.ndarray, n_steps: int = 6) -> np.ndarray`:

  * Given $K$ facial sub-ROI RGB traces reshaped to $V_{\text{st}} \in \mathbb{R}_{\ge 0}^{T \times 3K}$ (shifted non-negative via ReLU/min-subtraction, mapping temporal dimension $T \mapsto M$ and spatial-channel dimension $3K \mapsto N$ per `FactorizePhys` Eq. 6 & Table 1), run `n_steps = 6` multiplicative NMF updates for rank $L = 1$:
    

    $$
    H \leftarrow H \odot \frac{W^\top V_{\text{st}}}{W^\top W H + 10^{-8}}, \quad W \leftarrow W \odot \frac{V_{\text{st}} H^\top}{W H H^\top + 10^{-8}}
    $$

  * Use the reconstructed low-rank attention $\hat{V}_{\text{st}} = W H$ to weight the sub-ROI traces before feeding classical/BeatFormer pipelines.

## 5. Phase 3: Complete All 7 Competition Constraints (`C1`–`C7`) & Fairness Calibration

### Task 3.1: Illumination-Normalized ITA Skin-Tone Classifier (`preprocessing/skin_tone.py`)

* Currently, bright directional lighting on UBFC subjects inflates $L^*$ in CIE $L^*a^*b^*$, causing `compute_ita` to classify 29/42 subjects as `Type I`.

* In `compute_ita(roi_pixels, reference_white=None)`:

  * Apply a Grey-World / luminance-percentile normalization step so excessive camera auto-exposure glare (top 5% brightest specular pixels) is trimmed (keep 10th to 85th luminance percentiles of skin ROI pixels) before computing `mean_L` and `mean_b`.

  * Retain full compatibility with `test_skin_tone_ita_and_fitzpatrick` in `tests/test_components.py`.

### Task 3.2: Add Missing Constraints `C3`, `C5` (Bitrate), and `C6` (Aging Skin)

1. **`C5` Bitrate-Limited Compression (`constraints/low_res.py`)**:

   * Add `apply_bitrate_compression_c5(input_video_path: str, output_video_path: str, target_kbps: int = 400) -> str` using `ffmpeg -y -i <in> -c:v mpeg4 -b:v 400k <out>` (per §2.2 C5: *"Re-encode the video using VideoWriter with the 'MPEG-4' profile and a constrained target bitrate* $\le 500\text{ kbps}$*"*).

2. **`C6` Aging-Skin Micro-Texture Stress Test (`constraints/aging_skin.py`)**:

   * Implement `apply_aging_skin_c6(frame_bgr: np.ndarray, wrinkle_strength: float = 0.35) -> np.ndarray` per §2.2 C6 (*"Apply localized micro-texture/shadow perturbation via imfilter with edge-emphasizing kernels to approximate wrinkle-induced reflectance variation"*):

     * Apply an anisotropic directional DoG / high-pass Sobel micro-shadow kernel across the forehead and outer eye/cheek regions to simulate age-induced micro-shadowing, and verify in `scripts/run_constraints.py` that PFD + CZT maintains $< 2.0\text{ bpm}$ error.

3. **`C3` Multi-Person Frame Isolation (`constraints/multi_face.py` & `preprocessing/face_mesh.py`)**:

   * In `FaceMeshTracker.process_frame()`, add support for `return_all_faces: bool = False` when `max_num_faces > 1`.

   * Create `constraints/multi_face.py` with:

     * `simulate_multi_person_video(target_frames: np.ndarray, distractor_hr_bpm: float = 115.0) -> np.ndarray`: embeds a second synthetic/cropped bystander face pulsing at a different heart rate (`115 BPM`) in the background/side of the frame.

     * `TargetFaceKalmanTracker`: tracks multiple detected face bounding boxes across frames using a constant-velocity Kalman filter + Hungarian/greedy IoU assignment (mirroring MATLAB's `vision.KalmanFilter` + `assignDetectionsToTracks` in §2.2 C3), locking onto the primary subject (largest central face in frame 0) and rejecting the bystander's 115 BPM signal.

## 6. Phase 4: Comprehensive Benchmarking & Ablation Harness

Create `scripts/run_ablation_matrix.py` and update `scripts/run_ubfc.py` to generate the final tables and plots for the IEEE report:

1. **UBFC-rPPG 42-Subject Video Benchmark (`scripts/run_ubfc.py`)**:

   * Run all 4 methods (`POS + FFT`, `POS + CZT`, `CHROM + CZT`, `ME-rPPG (stabilized + windowed)`, and `Hybrid v3 (ME-rPPG + PFD + BeatFormer-SCL + CZT)`) across the 42 UBFC subjects.

   * Save comparison table to `artifacts/ubfc/ubfc_ablation_summary.csv` and generate side-by-side Bland–Altman plots.

   * **Target Gate**:

     * `POS / CHROM + CZT` MAE $< 2.5\text{ BPM}$ on UBFC video frames.

     * `ME-rPPG (stabilized)` MAE $< 2.0\text{ BPM}$ on UBFC video frames (down from `5.91 BPM`).

     * `Hybrid v3` MAE $< 1.5\text{ BPM}$ on UBFC video frames and $< 2.5\text{ BPM}$ under `C1` ($\le 50\text{ lux}$).

2. **Component Ablation Matrix (`scripts/run_ablation_matrix.py`)**:

   * Automatically run and export `artifacts/ablation_matrix.md` and `artifacts/ablation_matrix.csv` isolating each paper contribution:

| Configuration | Spectral Estimator | BBox & Skin Filter | Denoiser / Attention | UBFC MAE | AiPEX (C2) MAE | C1 (10 lux) Error | C7 (Masked) Error | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| **Baseline v0** | Welch FFT | Raw Per-Frame | None (POS) | Report | Report | Report | Report | 
| **+ CZT (`BeatFormer`)** | 512-bin CZT | Raw Per-Frame | None (POS) | Report | Report | Report | Report | 
| **+ ROI Stabilization & Skin Filter (`frPPG`)** | 512-bin CZT | EMA BBox + YCbCr + SNR ROI | None (POS) | Report | Report | Report | Report | 
| **Pipeline v2 (`ME-rPPG`)** | 512-bin CZT | EMA BBox + RMA Warmup | `ME-rPPG` State-Space | Report | Report | Report | Report | 
| **+ `PFD` (`FreqPhys`)** | 512-bin CZT | EMA BBox + RMA Warmup | `ME-rPPG` + `PFD` (PBF+PSM+ASS) | Report | Report | Report | Report | 
| **Pipeline v3 (`Full Hybrid`)** | 512-bin CZT | EMA BBox + NMF/SNR ROI | `ME-rPPG` + `PFD` + `BeatFormer-SCL` | Report | Report | Report | Report | 

## 7. Step-by-Step Execution Checklist for the Agent

Execute the tasks in this exact order and run `PYTHONPATH=. pytest tests/` after each step:

* \[x\] **Step 1**: Update `preprocessing/roi_extractor.py` (EMA bbox stabilization, YCbCr/glare skin filtering, sub-ROI extraction, and automatic spectral SNR fusion) and `preprocessing/skin_tone.py` (percentile glare trimming before ITA).

* [x] **Step 2**: Update `models/hybrid/pfd_module.py` with harmonic-consistent `PSM` (`FreqPhys` Eqs. 3–5) and adaptive threshold `ASS` (`FreqPhys` Eqs. 7–9).

* [x] **Step 3**: Create `models/hybrid/beatformer.py` (`CZTLayer`, `ZOCA`, `ESD`, `ICZTLayer`) and update `models/hybrid/scl_pretrain.py` (`EMD` loss, `SCL` triplet hinge loss, `Orthonormal` loss, `RankContrast` resampling); run `scripts/train_beatformer_scl.py` to save `artifacts/beatformer_scl.pt`.

* [x] **Step 4**: Update `models/deep/merppg_engine.py` (45-frame RMA state pre-warming + sliding-window HR estimator with harmonic/jump outlier rejection) and `models/hybrid/hybrid_system.py` (dual-branch SNR-weighted fusion of `ME-rPPG + PFD` and `BeatFormer`).

* [x] **Step 5**: Add `C5` bitrate compression (`constraints/low_res.py`), `C6` aging-skin simulator (`constraints/aging_skin.py`), and `C3` multi-face Kalman/IoU tracker (`constraints/multi_face.py`).

* [x] **Step 6**: Fix `scripts/run_constraints.py` (`fs=30.0` vs `fs=15.0`, all 88 AiPEX subjects, real UBFC videos for `C1`–`C7`) and re-generate `artifacts/constraints/`.

* [x] **Step 7**: Fix `pipeline_packaging/exporter.py` (`bvp_signal` column) and `scripts/package_submission.py` (real video + sliding-window HR series so `pearson_r > 0.90` in `Round1_Baseline_None.mat`), then re-run `PYTHONPATH=. python scripts/package_submission.py`.

* [x] **Step 8**: Add unit tests in `tests/test_components.py` covering `BeatFormer`, upgraded `PFD`, `TargetFaceKalmanTracker` (`C3`), `apply_aging_skin_c6` (`C6`), and non-zero `pearson_r` in `tests/test_submission_packaging.py`; verify `PYTHONPATH=. pytest tests/` passes 100%.