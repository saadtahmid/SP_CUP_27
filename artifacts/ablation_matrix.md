# Multi-Subject Component Ablation Matrix (N = 10 UBFC Subjects — [0.7, 3.0] Hz)

| Configuration | Spectral Estimator | BBox & Skin Filter | Denoiser / Attention | Baseline MAE ± SD (BPM) | C1 (50 lux) MAE ± SD (BPM) | C1 (10 lux) MAE ± SD (BPM) | C7 (Masked) MAE ± SD (BPM) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Baseline v0 (POS + Welch FFT)** | Welch FFT | Raw Full BBox | None (POS) | 15.50 ± 20.13 | 28.77 ± 24.38 | 19.46 ± 18.31 | 18.64 ± 12.30 |
| **2. + CZT Zooming (POS + 512-bin CZT)** | 512-bin CZT | Raw Full BBox | None (POS) | 13.87 ± 19.13 | 25.17 ± 25.57 | 23.80 ± 21.18 | 21.33 ± 18.66 |
| **3. + Locked R4 Sub-ROI & Mask Gate (CHROM)** | 512-bin CZT | Locked R4 Sub-ROI + Mask Gate | Chrominance (CHROM) | 8.48 ± 14.25 | 19.93 ± 13.47 | 23.25 ± 18.09 | 9.81 ± 14.66 |
| **4. Pipeline v2 (ME-rPPG State-Space)** | 512-bin CZT | Locked BBox + 36x36 Crop | ME-rPPG State-Space | 2.46 ± 3.46 | 7.14 ± 6.85 | 15.72 ± 9.14 | 2.50 ± 2.65 |
| **5. + FreqPhys PFD (Fundamental-Verified)** | 512-bin CZT | Locked BBox + 36x36 Crop | ME-rPPG + PFD | 2.56 ± 4.16 | 7.19 ± 7.49 | 16.18 ± 10.37 | 1.78 ± 2.55 |
| **6. Pipeline v3/v4 (Full Consensus Hybrid)** | 512-bin CZT | Locked R4 Sub-ROI + Mask Gate | ME-rPPG + PFD + BeatFormer + CHROM | 3.26 ± 5.19 | 7.14 ± 7.59 | 20.22 ± 12.81 | 1.81 ± 2.54 |
