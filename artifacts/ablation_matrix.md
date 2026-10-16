# Multi-Subject Component Ablation Matrix (N = 10 UBFC Subjects — [0.7, 3.0] Hz)

| Configuration | Spectral Estimator | BBox & Skin Filter | Denoiser / Attention | Baseline MAE ± SD (BPM) | C1 (50 lux) MAE ± SD (BPM) | C1 (10 lux) MAE ± SD (BPM) | C7 (Masked) MAE ± SD (BPM) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Baseline v0 (POS + Welch FFT)** | Welch FFT | Raw Full BBox | None (POS) | 15.50 ± 20.13 | 16.56 ± 12.21 | 22.99 ± 18.24 | 18.64 ± 12.30 |
| **2. + CZT Zooming (POS + 512-bin CZT)** | 512-bin CZT | Raw Full BBox | None (POS) | 13.87 ± 19.13 | 18.05 ± 16.25 | 21.29 ± 14.40 | 21.33 ± 18.66 |
| **3. + Locked R4 Sub-ROI & Mask Gate (CHROM)** | 512-bin CZT | Locked R4 Sub-ROI + Mask Gate | Chrominance (CHROM) | 8.48 ± 14.25 | 20.25 ± 13.15 | 15.19 ± 9.09 | 9.81 ± 14.66 |
| **4. Pipeline v2 (ME-rPPG State-Space)** | 512-bin CZT | Locked BBox + 36x36 Crop | ME-rPPG State-Space | 2.46 ± 3.46 | 6.86 ± 9.38 | 16.33 ± 13.99 | 2.50 ± 2.65 |
| **5. + FreqPhys PFD (Fundamental-Verified)** | 512-bin CZT | Locked BBox + 36x36 Crop | ME-rPPG + PFD | 2.56 ± 4.16 | 4.85 ± 5.67 | 17.24 ± 13.87 | 1.78 ± 2.55 |
| **6. Pipeline v3/v4 (Full Consensus Hybrid)** | 512-bin CZT | Locked R4 Sub-ROI + Mask Gate | ME-rPPG + PFD + BeatFormer + CHROM | 1.96 ± 3.83 | 6.52 ± 6.14 | 17.10 ± 14.00 | 6.69 ± 12.75 |
