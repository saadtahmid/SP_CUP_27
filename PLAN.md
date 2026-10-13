# Round 1 Battle Plan: HR Extraction Under Controlled + Uncontrolled Conditions

**Deadline: 30 November 2026 | Result: 5 December 2026 | Evaluated on: Innovation, Technical Feasibility, Clarity**

---

## 1. Decoding What Round 1 Actually Requires

Before any technical decision, here's what the competition document explicitly demands:

| Requirement | Source in Competition Doc | Implication for Plan |
|---|---|---|
| HR from RGB video, controlled **and** uncontrolled conditions | §2.3 Round 1 | Need both clean (UBFC/PURE) and hard (MMPD) datasets |
| **≥2 constraints from C1–C7 addressed and justified** "even at this baseline stage" | §2.3 Task 1 submission requirement | Constraint work is *mandatory*, not optional |
| More constraints = more grading credit | FAQ #2: "the total number of constraints considered will be taken into account" | Aim for 4–5 constraints, not the minimum 2 |
| Metrics: MAE, RMSE, Pearson r, **Bland–Altman limits of agreement** | §2.3 Task 1, Eval. Metrics | Bland–Altman is explicitly named — most student teams skip it |
| MATLAB simulation specs for each constraint (imnoise, imresize, insertShape, rgb2lab/ITA) | §2.2 Constraint specifications | The organizers told us *exactly* how they expect constraints to be emulated — follow their scripts verbatim |
| Datasets: organizer's clinical set, public sets (UBFC-RPPG, AiPEX rppg_biases listed in Annexure), or own IRB data | §3, Annexure 1 | Public datasets are the fastest sanctioned path |
| Submission: `Round1_Baseline_None.mat` (with facial_video, roi_data, reference_signal, processed_signal, metrics, **params incl. Fitzpatrick type, lux, age**), `process_round1.m`, CSVs, IEEE-format report | §4 Data organization | Metadata logging must be designed into the pipeline from Day 1 |

**Strategic reading:** Round 1 is not won by the lowest MAE alone. It's won by (a) a credible baseline, (b) *demonstrated* robustness strategy via constraints, and (c) a report that shows a clear innovation trajectory toward Rounds 2–3. The judges are selecting teams they believe can deliver the full three-parameter system.

---

## 2. Strategic Decisions at a Glance

| Decision | Choice | Primary Justification (Paper) |
|---|---|---|
| **Datasets** | UBFC-rPPG + PURE (clean), MMPD (stress), AiPEX rppg_biases (skin tone) | Debnath review (dataset table, benchmarks); AiPEX is organizer-endorsed (Annexure) |
| **Classical baselines** | POS, CHROM, GREEN, ICA | frPPG paper (community benchmark ranking: CHROM #1, POS #2, ICA #3) |
| **Pre/post-processing optimization** | Exhaustive permutation via frPPG framework | frPPG's core finding: **80% of MAE variance comes from pre/post-processing, not the core algorithm** |
| **Primary deep model** | ME-rPPG | Best cross-dataset MMPD (5.38 MAE), 580K params, 9.46 ms CPU inference (foreshadows Round 3 Pi demo) |
| **HR estimation upgrade** | Replace FFT with **CZT (Chirp-Z Transform)** in ALL pipelines | BeatFormer ablation: FFT 14.50 → CZT 9.15 MAE on MMPD (~5 bpm gain, zero parameters) |
| **Low-light robustness (C1)** | Extract + inject FreqPhys's **PFD module** into ME-rPPG | FreqPhys BUAA low-light MAE 1.13 vs. next-best 5.89 (5.2× better); PFD is plug-and-play (their Table 5) |
| **Constraint-aware training (C7)** | BeatFormer's **SCL** with spatial-occlusion augmentation | BeatFormer's occlusion transform *directly simulates* the competition's masked-face condition |
| **Core constraints** | **C2 (skin tone) + C4 (low resolution)** | Both have direct dataset support + trivial MATLAB simulation per spec |
| **Differentiator constraints** | **C1 (low light) + C7 (masks)**, bonus C5 (frame decimation, classical only) | C1 has FreqPhys evidence; C7 has BeatFormer SCL evidence; C5 is nearly free to test |
| **Deferred** | C3 (multi-face), C6 (aging), BP/SpO₂ | C3 needs the full Kalman tracking stack (Round 2/3); C6 needs elderly data we don't have yet |

---

## 3. The System Pipeline (Three Versions, Built Incrementally)

### Pipeline v1 — Classical Baseline (Week 2) 

```
Video → Face detection (MediaPipe FaceMesh) → ROI: forehead + bilateral cheeks
      → Spatial averaging → RGB traces
      → Temporal normalization (zero-mean, unit-variance per channel)
      → Core algorithm: {POS, CHROM, GREEN, ICA}   [frPPG implementations]
      → Butterworth bandpass 0.7–3.0 Hz             [competition's stated band]
      → HR estimation: FFT peak  →  then CZT zoomed peak   [BeatFormer]
      → Metrics: MAE, RMSE, r, Bland–Altman
```

**Why each stage (paper rationale):**

- **POS/CHROM as cores**: The frPPG paper's systematic review of 121 publications ranked CHROM and POS as the #1 and #2 most-used benchmarks — beating every deep method in frequency of use, and (per the MMPD results in both BeatFormer and FreqPhys tables) still outperforming most deep models under motion. The Saikevicius survey adds the theoretical reason: chrominance channels are "less influenced by melanin content compared to luminance" — your C2 justification.
- **frPPG for tuning**: Their headline finding — 80% of refined methods ranked in the top 25th percentile once pre/post-processing was optimized, 60% surpassing the best literature-reported results. Their repo (`Building-Robotics-Lab/flexible_rPPG`) contains faithful original implementations with the exact window sizes, strides, and filter specs (e.g., CHROM: 1.6 s window / 0.8 s stride; ICA: 30 s window, JADE, second component, 12 bpm outlier rejection).
- **CZT for HR estimation**: BeatFormer's ablation is unambiguous — CZT alone improved MMPD MAE from 14.50 to 9.15, because it concentrates all frequency bins in the HR band (~13× the resolution of FFT at 30 fps). This is a ~30-line change with no learned parameters. **Note:** widen BeatFormer's band from their 0.66–2.5 Hz to the competition's **0.7–3.0 Hz (42–180 bpm)** so you don't clip high heart rates.

### Pipeline v2 — Deep Learning Primary (Weeks 3–4)

ME-rPPG in **chunk mode** (offline inference is fine for Round 1; the single-frame "flow" mode is saved as the Round 3 headline).

- **Why ME-rPPG over PhysNet/TS-CAN/EfficientPhys:** Its cross-dataset MMPD MAE of 5.38 (trained on RLAP) is the best we've seen from any deployable architecture; PhysNet gets 11.20 and TS-CAN 10.19 on the same protocol (ME-rPPG Table 4). MMPD is the closest public analogue to your uncontrolled condition.
- **Why it fits the "Model and time complexity" judging criterion:** 580K params, 3.6 MB memory — you can honestly write "our primary model runs in 9.46 ms on a laptop CPU," which no team using PhysFormer (7.4M params, 2330 ms) can claim.
- **Adopt their data trick:** ME-rPPF generated +30% training data by time-scaling clips to cover 30–180 bpm, explicitly because datasets lack high/low-HR samples. Since your evaluation band is 42–180 bpm, this directly addresses a distribution gap. (The RePSS challenge winner independently used the same upsample/downsample property for their RankContrast loss — two independent confirmations.)
- **Known risk (from the paper itself):** ME-rPPG is implemented in **JAX**, and the authors admit this "may lead to some differences" vs. PyTorch. Mitigation below in §9.

### Pipeline v3 — Hybrid Innovation (Week 5)

**ME-rPPG backbone + FreqPhys PFD module + BeatFormer SCL pretraining.**

```
                    ┌─ SCL pretraining (unlabeled) ─┐
Constraint-         │  · HSV/LAB color conversion   │  ← C1: illumination invariance
simulated videos ──→│  · Temporal flipping          │  ← motion/phase disentangling
(MATLAB degradations)│  · Random spatial occlusion   │  ← C7: masked face!
                    └───────────────────────────────┘
                                  ↓
ME-rPPG state-space backbone (streaming temporal modeling)
                                  ↓
                    ┌─ PFD spectral refinement ─────┐
                    │  · PBF: hard mask [0.7–3.0 Hz]│  ← removes out-of-band noise
                    │  · PSM: learnable complex     │  ← enhances in-band harmonics
                    │    modulation                 │
                    │  · ASS: adaptive threshold    │  ← suppresses residual noise
                    └───────────────────────────────┘
                                  ↓
                    CZT-based HR estimation
```

**Why this hybrid is your innovation story (paper-by-paper):**

1. **PFD injection (FreqPhys):** Their Table 5 shows PFD improves *other* backbones as a plug-in (RhythmMamba: 4.30→4.15 on VIPL-HR; PhysDiff: 3.92→3.84). No one has yet combined it with a state-space streaming backbone. The BUAA low-light result (MAE 1.13, correlation 0.99 where the next-best method gets 5.89/0.48) is your evidence that frequency priors rescue signal extraction when SNR collapses — exactly the ≤50 lux C1 scenario. PFD is ~200 lines of PyTorch (their Equations 1–10 are fully specified; you don't need the diffusion wrapper).
2. **SCL with constraint-specific augmentations (BeatFormer):** Their four transformations map *one-to-one* onto competition constraints. The spatial-occlusion transform is literally a synthetic mask (C7). The color-space transforms preserve pulse while altering illumination statistics (C1). And SCL needs **no labels** — meaning you can pretrain on the organizer's own MATLAB-degraded videos without ground truth for the degraded versions.
3. **CZT (BeatFormer) as the final estimator** ties the frequency-domain narrative together end-to-end.

This gives your report a coherent thesis: *"Explicit physiological frequency priors + streaming state-space modeling = robust, edge-ready rPPG."* Two independent 2025–2026 papers (BeatFormer, FreqPhys) converge on the frequency-prior half; ME-rPPG establishes the streaming half; your contribution is the synthesis.

---

## 4. Dataset Plan

| Dataset | Role | Key Properties | Justification |
|---|---|---|---|
| **UBFC-rPPG** (42 videos) | Primary clean baseline; Round1_Baseline packaging | 30 fps, 640×480, CMS50E pulse-ox GT | Organizer-listed (Annexure); the single most-used benchmark in the frPPG review; native VGA resolution conveniently doubles as C4 evidence |
| **PURE** (60 videos, 10 subjects) | Motion robustness (6 head-movement protocols) | 30 fps, 60 Hz oximeter | frPPG's #2 dataset; its steady/talking/rotation/translation splits give you a free "uncontrolled" axis |
| **MMPD** (660 videos, 33 subjects) | **Uncontrolled condition testbed: C1 + C2** | Fitzpatrick 3–6, 4 lighting scenarios incl. **LED-low**, 4 activities incl. walking | Debnath review flags it as the skin-tone dataset; ME-rPPG and FreqPhys both use it as their hardest benchmark (FreqPhys: 4.20 MAE intra) |
| **AiPEX rppg_biases** (88 videos, India + Sierra Leone) | **C2 primary evaluation** | Ground-truth HR per video; demographically diverse; includes cultural artifacts (bindi) | **Organizer-listed in Annexure** — using it signals you read their materials; the repo ships `blandaltman.py` and hypothesis-testing scripts you can reuse |
| Sri Ramachandra clinical set | Request now, use in Rounds 2–3 | All 3 parameters with clinical GT | Agreement takes time; start the paperwork in Week 1 so it doesn't block Round 2 |
| VitalVideo (VV-EU-1-small) | Defer to Round 3 (BP) | 100 subjects, BP labels | Cloud-restriction clause in the agreement makes it awkward; HR-focused public sets suffice for Round 1 |

**Metadata obligation:** the `.mat` submission requires `params` with *Fitzpatrick type, lux, age, resolution, fps, distance*. Build a metadata manifest for every video on Day 1 (UBFC/PURE lack Fitzpatrick labels — document this limitation honestly; AiPEX and MMPD carry tone information you can map to types via the competition's own ITA procedure: `rgb2lab` → Individual Typology Angle).

---

## 5. Constraint Plan (Targeting 4–5 Constraints)

### Core pair (guaranteed deliverables)

**C2 — Skin Tone Generalization**
- **Data:** AiPEX dataset (stratified evaluation) + MMPD (Fitzpatrick 3–6 training/testing).
- **Procedure (per competition spec):** compute ITA from facial ROI pixels via `rgb2lab`; stratify accuracy per skin-tone stratum; **pre-specify the tolerance** in your report (e.g., per-type MAE within ±3 bpm or ±25% of population mean — the doc says "pre-specified tolerance" is yours to define, so define it *before* running results and state it).
- **Paper rationale:** Saikevicius — chrominance methods (POS/CHROM) are structurally less melanin-sensitive than luminance/green-only; Debnath — Project ICA "encounters challenges while extracting signals from dark skin variations," so don't lean on ICA for C2; ME-rPPG's MMPD leadership (5.58 intra / 5.38 cross) suggests state-space features transfer across tones.

**C4 — Low Resolution**
- **Procedure (per competition spec):** `imresize` baseline video to 640×480 (VGA) and 320×240 (QVGA) with bicubic interpolation; re-run the full pipeline; report degradation curves vs. the HD-equivalent baseline.
- **Paper rationale:** BeatFormer resizes to 96×96 and still achieves SOTA motion robustness — proof that ROI-averaged methods degrade gracefully; FreqPhys's MSTmap representation is similarly resolution-tolerant. Expect classical methods to hold up well (they only need enough pixels for a skin ROI), and note in the report where deep models start failing (spatial detail loss).

### Differentiator pair (innovation credit)

**C1 — Low Light (≤50 lux)**
- **Procedure (per competition spec, verbatim):** brightness attenuation (`im2double` × gain) → `imnoise(I,'poisson')` for shot noise → combined attenuation + `imnoise(I,'gaussian',0,var)`. Produce a lux-vs-MAE degradation curve.
- **Paper rationale:** FreqPhys's BUAA low-light result (MAE 1.13 vs. 5.89 next-best) is the strongest published evidence that frequency priors survive SNR collapse — your PFD-injected variant vs. plain ME-rPPG on this curve becomes your headline ablation figure. ME-rPPG's TN module (RMA detrending, removing DC/illumination drift) adds a second line of defense.

**C7 — Masked Face (≥40% lower-face occlusion)**
- **Procedure (per competition spec):** `insertShape` filled polygon aligned to facial landmarks over the lower face; verify the pipeline re-weights toward forehead/periocular ROI.
- **Paper rationale:** BeatFormer's SCL uses *random spatial occlusion as a training transformation* on exactly the assumption that "cardiac information can be recovered as long as a sufficient skin region is visible." Pretrain with occlusion augmentation → the model learns forehead-dominant extraction without labels.

### Bonus (near-zero cost)

**C5 — Frame-rate decimation** (classical methods only): decimate to 10 fps via `VideoReader/VideoWriter`, re-run POS/CHROM. The competition itself notes the Nyquist floor (6 fps for the 0.7–3 Hz band) — citing this correctly in the report costs nothing and shows signal-processing literacy. (Skip for deep models — they assume 30 fps frame stacks.)

**Deferred with stated rationale in the report:** C3 (needs the full MATLAB `faceDetector` + `vision.KalmanFilter` + `assignDetectionsToTracks` tracking stack — scheduled for Round 2), C6 (needs elderly subject data; synthetic wrinkle perturbation via `imfilter` is allowed as "supplementary stress-testing" per the doc, so include one such experiment if time permits).

---

## 6. Week-by-Week Execution Timeline

*(Anchored to: registration deadline 25 Oct 2026, Round 1 deadline 30 Nov 2026 ≈ 7 weeks)*

### Week 1 (Oct 12–18) — Foundation
- [ ] **Admin:** Team registration + SPS membership (≥3 undergrads) + signed T&C; download complimentary MATLAB license; email organizer re: Sri Ramachandra agreement; submit VitalVideos agreement (already drafted).
- [ ] **Environment:** Clone rPPG-Toolbox, frPPG, ME-rPPG-demo, FactorizePhys, BeatFormer, FreqPhys repos. One shared Conda env + one MATLAB R2024+ install.
- [ ] **Data:** Download UBFC, PURE, MMPD, AiPEX; verify loads; build the metadata manifest (subject ID, skin tone, fps, resolution, GT source).
- [ ] **Deliverable:** Repo skeleton matching the competition directory structure (`/Round1_Baseline/` etc.), metrics module (MAE/RMSE/r/Bland–Altman — port `blandaltman.py` from AiPEX repo).

### Week 2 (Oct 19–25) — Classical Baselines Complete ✅ *Gate 1*
- [ ] Run POS, CHROM, GREEN, ICA on UBFC + PURE via frPPG implementations (use their exact original parameters from their Table 3 — don't hand-roll).
- [ ] Run frPPG's permutation search on UBFC to find the best pre/post-processing combo per core method.
- [ ] Implement CZT HR estimator (MATLAB `czt()` and Python `scipy.signal.CZT`); A/B test FFT vs. CZT across all four cores.
- [ ] **Gate 1 (go/no-go):** POS/CHROM MAE < 5 bpm on UBFC. Literature says 2–4 is achievable (FreqPhys Table 1: CHROM 2.37); if you're above 5, your ROI/normalization has a bug — fix before proceeding.

### Week 3 (Oct 26–Nov 1) — Deep Model Online
- [ ] Reproduce ME-rPPG on UBFC+PURE (use released weights first, fine-tune after; chunk mode). If JAX friction exceeds 2 days → **fallback:** TS-CAN or EfficientPhys via rPPG-Toolbox (both are proven, and the toolbox standardizes preprocessing).
- [ ] Cross-dataset eval: train UBFC → test MMPD. This is your "uncontrolled condition" headline number.
- [ ] Apply the +30% time-scaling augmentation (30–180 bpm coverage) — ME-rPPG's own trick.
- [ ] **Gate 2:** ME-rPPG within ~1.5× of its reported numbers, or fallback model MAE < 3 on UBFC.

### Week 4 (Nov 2–8) — Constraint Harness Complete ✅ *Gate 3*
- [ ] MATLAB constraint simulator per competition specs: C4 (`imresize` VGA/QVGA), C1 (attenuation + poisson + gaussian), C7 (`insertShape` mask), C5 (decimation).
- [ ] **C2 evaluation:** ITA stratification on AiPEX + MMPD; per-skin-tone error tables with the pre-specified tolerance.
- [ ] Generate the degradation curves for classical + deep pipelines.
- [ ] **Gate 3:** complete metrics table for C2 + C4 across all models. *At this point you have a minimum viable submission — everything after is upside.*

### Week 5 (Nov 9–15) — Hybrid & Innovation
- [ ] Extract/implement PFD (PBF + PSM + ASS — FreqPhys Eqs. 1–10; ~200 lines) and inject post-ME-rPPG on the temporal window.
- [ ] SCL pretraining loop with the four BeatFormer transformations, applied to **the organizer's own degraded simulations**.
- [ ] Ablation matrix: {baseline, +CZT, +PFD, +SCL, full} × {UBFC, PURE, MMPD, AiPEX} × {C1, C2, C4, C7 conditions}.
- [ ] **Gate 4:** hybrid ≥ plain ME-rPPG on MMPD and on the C1 degradation curve. If PFD doesn't help, report the negative result honestly — a clean ablation with a null result beats a cherry-picked one.

### Week 6 (Nov 16–22) — Evaluation Freeze + Report v1
- [ ] Freeze all models; final Bland–Altman plots per dataset and per constraint.
- [ ] Report v1 (IEEE conference template), following the outline in §8.
- [ ] Package `Round1_Baseline_None.mat` (facial_video, roi_data, reference_signal, processed_signal, metrics, params — **including Fitzpatrick, lux, age metadata**) + `process_round1.m` + CSVs + one representative `target_video.mp4`.
- [ ] Internal review: supervisor + one member not on the codebase (fresh eyes on clarity).

### Week 7 (Nov 23–30) — Buffer & Submit
- [ ] Address review; verify directory structure and naming *character-by-character* against §4 of the doc ("Strict adherence... required for automated evaluation").
- [ ] Dry-run: can someone clone the repo and reproduce your numbers in one command?
- [ ] **Submit early (target: Nov 27–28)** — never the 30th.

---

## 7. Team Role Assignment (6-member template; scale as needed)

| Role | Members | Owns | Key tools/papers |
|---|---|---|---|
| **Team Lead / PM** | 1 | Timeline, gates, submission packaging, organizer liaison | Competition doc (naming conventions!) |
| **Classical SP** | 1–2 | frPPG baselines, CZT estimator, bandpass design, C5 | frPPG, BeatFormer, Dawoodjee specs |
| **Deep Learning** | 2 | ME-rPPG reproduction/training, time-scaling augmentation, fallback models | ME-rPPG, rPPG-Toolbox |
| **Constraint & Data Engineer** | 1 | MATLAB degradation suite, metadata manifest, AiPEX/MMPD pipelines, ITA computation | Competition §2.2, AiPEX repo |
| **Hybrid/Innovation** | 1 (can overlap with DL) | PFD module, SCL loop, ablation matrix | FreqPhys, BeatFormer, FactorizePhys |
| **Evaluation & Report** | 1 (can overlap with PM) | Metrics module, Bland–Altman, per-skin-tone tables, IEEE report | Debnath (metric defs), Saikevicius (taxonomy for related-work section) |

**MATLAB coverage is non-negotiable:** the constraint specs, the `.m` submission script, and (per the sponsor relationship) the Round 3 Raspberry Pi toolchain are all MATLAB-centric. Ensure ≥2 members are fluent.

---

## 8. Technical Report Outline (Mapped to Judging Criteria)

| Section | Content | Serves Criterion |
|---|---|---|
| 1. Introduction | Clinical motivation (telehealth, low-resource settings) — mirror the competition's framing | Clarity |
| 2. Related Work | Saikevicius taxonomy (color/motion/hybrid) + Debnath performance benchmarks as your landscape | Innovation (positions your gap) |
| 3. Method | Pipeline v1→v3; **CZT spectral estimation**; **PFD frequency-prior denoising**; **SCL constraint training**; state-space backbone. One paragraph per component citing its origin paper and your modification | **Innovation** |
| 4. Experimental Setup | Datasets table (adapt Debnath's), metrics (MAE/RMSE/r/Bland–Altman with formulas), constraint simulation specs **copied from the competition doc** | Technical feasibility |
| 5. Results | (a) Clean baselines UBFC/PURE vs. literature; (b) uncontrolled MMPD; (c) constraint results C1/C2/C4/C7 with degradation curves; (d) **ablation matrix** (CZT/PFD/SCL on-off); (e) Bland–Altman limits of agreement per condition | **Performance + feasibility** |
| 6. Discussion | Where it fails (motion walking split, C3 deferred), per-skin-tone fairness table with pre-specified tolerance, compute cost table (params/latency — cite ME-rPPG's CPU numbers) | Feasibility, fairness |
| 7. Conclusion & Roadmap | Round 2 (SpO₂ via ratio-of-ratios on R/B channels), Round 3 (BP via RePSS strategies, Pi deployment with ME-rPPG flow mode) | Shows the judges a 3-round trajectory |

**Length discipline:** 6–8 pages. Judges read dozens — every figure must earn its place. The two money figures: (1) the C1 lux-vs-MAE curve showing PFD rescue, (2) the per-skin-tone (C2) error table.

---

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| ME-rPPG JAX reproduction fails (authors themselves flag platform differences) | Medium | High | 2-day timebox → fallback to TS-CAN/EfficientPhys via rPPG-Toolbox (proven, standardized); CZT + PFD + SCL innovations are backbone-agnostic |
| FreqPhys code incomplete (ECCV 2026, "code will be released") | Medium | Medium | PFD is fully specified by Equations 1–10 — reimplement (~200 lines); you don't need their diffusion trainer, only the denoiser module |
| No GPU access for training | Medium | Medium | Use released pretrained weights (ME-rPPG ships models; exported ONNX ran on a laptop CPU in their user study); fine-tune only small models; classical methods need zero training. **Note:** public datasets have no cloud restriction, but any VitalVideos data must stay local per the agreement — for Round 1 you don't need it |
| Skin-tone tolerance failure on darker types (the documented failure mode of the field) | Medium | High | Chrominance cores (POS/CHROM) per Saikevicius's melanin argument; train on MMPD (types 3–6); report per-type honestly — the competition *rewards* explicit stratified reporting over a pooled number |
| UBFC is 640×480, below the 1280×720 "baseline" spec | Low | Low | Annexure explicitly sanctions UBFC; state in the report that baseline-condition footage is simulated per §2.2 where needed; PURE provides the complementary protocol |
| Team bandwidth (exams, holidays in Nov) | High | Medium | Week 4 Gate 3 = minimum viable submission; everything in Weeks 5–6 is stretch; PM enforces scope freeze after Week 5 |
| Bland–Altman done wrong (it's named in the metrics, most teams fumble it) | Low | Medium | Reuse `blandaltman.py` from the AiPEX organizer-endorsed repo; report both bias and limits of agreement per dataset and per constraint |

---

## 10. Pre-Submission Checklist (Week 7)

- [ ] `Round1_Baseline_None.mat` contains all six required variables, with `params` including resolution, fps, lux, camera distance, **Fitzpatrick type, age, constraints applied**
- [ ] `process_round1.m` runs end-to-end on `target_video.mp4` and emits `processed_output.csv` in one command
- [ ] Video files match the subject IDs in the `.mat` (explicit doc requirement)
- [ ] Directory structure: `/Round1_Baseline/` exactly as specified
- [ ] Report in IEEE conference format, ≤8 pages, all figures legible at 100% zoom
- [ ] ≥2 constraints addressed (target: C1, C2, C4, C7 + C5 bonus), each with its MATLAB simulation procedure documented
- [ ] Bland–Altman plots present
- [ ] Ablation table present (CZT / PFD / SCL contributions isolated)
- [ ] Ethics statement: all datasets public per Annexure or agreement-backed; no participant-collected data without IRB
- [ ] Submitted via the official registration system (only work submitted there counts) — **by Nov 27–28**

---

## 11. How This Round 1 Work Compounds Into Rounds 2–3

Every Round 1 investment was chosen because it carries forward:

- **ME-rPPG flow mode** (single-frame streaming) → your Round 3 Raspberry Pi live demo is already 80% built.
- **PFD frequency priors** → SpO₂ estimation (Round 2) is fundamentally a *spectral ratio* problem (pulsatile/non-pulsatile absorbance at red vs. blue/green) — the PSM/ASS machinery operates on exactly those components.
- **SCL unlabeled training** → lets you adapt to the Sri Ramachandra clinical dataset's conditions without exhaustively labeling degraded variants.
- **AiPEX + MMPD stratified evaluation harness** → drops directly into Round 2/3 fairness reporting.
- **CZT estimator** → BP estimation (Round 3) needs precise waveform morphology; RePSS showed BP is won or lost on BVP quality, and FreqPhys's HRV table shows frequency-prior methods produce the cleanest waveforms.

---

### One-Paragraph Summary for Your Team Sync

> Round 1 is a 7-week sprint built on four papers doing four jobs: **frPPG** gives us rigorous classical baselines and the insight that pipeline tuning beats algorithm-hopping; **ME-rPPG** gives us our primary model — accurate on hard data, tiny, CPU-real-time; **BeatFormer** gives us two weapons — the CZT estimator (a free ~5 bpm accuracy gain in every pipeline) and SCL training that turns the organizer's own constraint simulations into training signal without labels; and **FreqPhys** gives us the PFD module, whose low-light dominance is our answer to constraint C1. We submit classical + deep + hybrid results across UBFC/PURE/MMPD/AiPEX, demonstrate four constraints (C2 + C4 core, C1 + C7 differentiators, C5 bonus) using the organizer's verbatim MATLAB procedures, and package everything in the exact `.mat`/naming format the automated evaluator expects — by Nov 27, not Nov 30.
