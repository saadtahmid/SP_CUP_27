# Multi-Subject Component Ablation Matrix (N = 10 UBFC Subjects — [0.7, 3.0] Hz)

| Configuration | Spectral Estimator | BBox & Skin Filter | Denoiser / Attention | Baseline MAE ± SD (BPM) | C1 (50 lux) MAE ± SD (BPM) | C1 (10 lux) MAE ± SD (BPM) | C7 (Masked) MAE ± SD (BPM) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Baseline v0 (POS + Welch FFT)** | Welch FFT | Raw Full BBox | None (POS) | 13.13 ± 16.31 | 17.24 ± 12.61 | 23.69 ± 18.71 | 20.38 ± 13.95 |
| **2. + CZT Zooming (POS + 512-bin CZT)** | 512-bin CZT | Raw Full BBox | None (POS) | 7.83 ± 11.26 | 15.54 ± 13.74 | 20.08 ± 12.03 | 17.16 ± 15.04 |
| **3. + Locked R4 Sub-ROI & Mask Gate (CHROM)** | 512-bin CZT | Locked R4 Sub-ROI + Mask Gate | Chrominance (CHROM) | 2.06 ± 2.98 | 20.25 ± 13.15 | 15.19 ± 9.09 | 2.62 ± 4.87 |
| **4. Pipeline v2 (ME-rPPG State-Space)** | 512-bin CZT | Locked BBox + 36x36 Crop | ME-rPPG State-Space | 2.15 ± 1.97 | 6.86 ± 9.38 | 16.33 ± 13.99 | 3.07 ± 4.12 |
| **5. + FreqPhys PFD (Fundamental-Verified)** | 512-bin CZT | Locked BBox + 36x36 Crop | ME-rPPG + PFD | 1.51 ± 2.49 | 4.85 ± 5.67 | 17.24 ± 13.87 | 1.43 ± 2.56 |
| **6. Pipeline v3/v4 (Full Consensus Hybrid)** | 512-bin CZT | Locked R4 Sub-ROI + Mask Gate | ME-rPPG + PFD + BeatFormer + CHROM | 1.55 ± 2.49 | 4.88 ± 5.60 | 17.21 ± 13.89 | 1.50 ± 2.54 |
