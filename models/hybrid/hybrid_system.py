"""Pipeline v3: Hybrid Innovation rPPG System.

Integrates:
1. Spatial-Temporal State-Space Backbone: ME-rPPG (Wang et al., 2025)
2. Complex Spectral Denoising: FreqPhys PFD (Qian et al., ECCV 2026)
3. Chrominance Spectral Transformer: BeatFormer (Comas & Sukno, ICCVW 2025; Martinez et al., 2025)
4. Dynamic In-Band SNR-Weighted Spectrum Fusion & CZT High-Resolution Estimation
"""

import os
from typing import Dict, Tuple, Optional, Union
import numpy as np
import torch

from models.deep.merppg_engine import MERPPGEngine
from models.hybrid.pfd_module import PhysiologicalFrequencyDenoiser
from models.hybrid.beatformer import BeatFormer
from models.classical.pos import pos_process
from estimation.czt_estimator import CZTHRTransformer


class HybridRPPGSystem:
    """End-to-End Hybrid rPPG System with dual-branch ME-rPPG + PFD and BeatFormer fusion."""

    def __init__(
        self,
        onnx_model_path: str = "third_party/ME-rPPG/model.onnx",
        state_json_path: str = "third_party/ME-rPPG/state.json",
        beatformer_weights_path: str = "artifacts/beatformer_scl.pt",
        fs: float = 30.0,
        enable_pfd: bool = True,
        enable_czt: bool = True,
        enable_beatformer: bool = True,
    ):
        self.fs = float(fs)
        self.enable_pfd = bool(enable_pfd)
        self.enable_czt = bool(enable_czt)
        self.enable_beatformer = bool(enable_beatformer)

        # 1. State-Space Deep Backbone (Spatial-Temporal Branch)
        self.backbone = MERPPGEngine(
            onnx_model_path=onnx_model_path,
            state_json_path=state_json_path,
            fs=self.fs,
        )

        # 2. PFD Spectral Denoiser (FreqPhys)
        self.pfd = PhysiologicalFrequencyDenoiser(fs=self.fs, low_hz=0.7, high_hz=3.0)

        # 3. CZT HR Estimator (512-bin high-resolution Chirp Z-Transform)
        self.czt = CZTHRTransformer(fs=self.fs, low_hz=0.7, high_hz=3.0, num_bins=512)

        # 4. BeatFormer Spectral Transformer (Chrominance Attention Branch)
        self.beatformer = BeatFormer(window_len=150, num_bins=128, fs=self.fs, low_hz=0.7, high_hz=3.0)
        if os.path.exists(beatformer_weights_path):
            try:
                self.beatformer.load_state_dict(torch.load(beatformer_weights_path, map_location="cpu"))
                self.beatformer.eval()
            except Exception:
                pass

    def process(
        self,
        cropped_faces: np.ndarray,
        rgb_signals: Optional[np.ndarray] = None,
        return_spectrum: bool = False,
    ) -> Dict[str, any]:
        """Process video through dual-branch hybrid pipeline.

        Args:
            cropped_faces: (T, 36, 36, 3) normalized float32 facial crops.
            rgb_signals: Optional (T, 3) spatial mean RGB color signals.
            return_spectrum: Whether to return fused power spectrum.

        Returns:
            dict containing raw BVP, cleaned BVP, estimated HR, and diagnostics.
        """
        t_len = len(cropped_faces)

        # Branch 1: Deep State-Space Backbone (ME-rPPG)
        bb_out = self.backbone.process_frames(cropped_faces, return_spectrum=False)
        raw_bvp = bb_out["bvp"]
        hr_series_me = bb_out.get("hr_series", np.full(t_len, bb_out["hr_bpm"]))

        # Step 2: PFD Spectral Denoiser on Branch 1
        if self.enable_pfd:
            cleaned_bvp, _, _ = self.pfd.denoise_spectrum(raw_bvp)
        else:
            cleaned_bvp = raw_bvp

        # Compute CZT spectrum for Branch 1 (freqs_czt is in BPM: [42.0, ..., 180.0])
        if t_len >= int(10.0 * self.fs):
            hr_me, hr_series_me = self.backbone.estimate_hr_windowed(cleaned_bvp)
            _, freqs_czt, psd_me = self.czt.estimate_hr(cleaned_bvp, return_spectrum=True)
        else:
            hr_me, freqs_czt, psd_me = self.czt.estimate_hr(cleaned_bvp, return_spectrum=True)
        snr_me = self.czt.compute_snr_db(psd_me, freqs_czt, hr_me)

        # Branch 2: BeatFormer Spectral Transformer + POS-PFD Chrominance Anchor
        if self.enable_beatformer:
            if rgb_signals is not None:
                rgb_trace = rgb_signals
            else:
                rgb_trace = np.mean(cropped_faces, axis=(1, 2)) * 255.0

            hr_bf, bvp_bf, psd_bf = self.beatformer.estimate_hr_from_trace(rgb_trace)
            # FIXED: Convert BeatFormer frequency axis from Hz [0.7, 3.0] to BPM [42.0, 180.0]
            freqs_bf_bpm = self.beatformer.czt.freq_axis.cpu().numpy() * 60.0

            # Also compute PFD-denoised POS chrominance spectrum for low-light/mask resilience
            pos_bvp = pos_process(rgb_trace, fs=self.fs)
            if self.enable_pfd:
                pos_bvp, _, _ = self.pfd.denoise_spectrum(pos_bvp)
            hr_pos, _, psd_pos = self.czt.estimate_hr(pos_bvp, return_spectrum=True)
            snr_pos = self.czt.compute_snr_db(psd_pos, freqs_czt, hr_pos)

            # Interpolate BeatFormer PSD to match 512-bin CZT BPM grid
            psd_bf_interp = np.interp(freqs_czt, freqs_bf_bpm, psd_bf)
            snr_bf = self.czt.compute_snr_db(psd_bf_interp, freqs_czt, hr_bf)

            # In-band SNR-weighted spectral fusion across ME-rPPG+PFD, BeatFormer-SCL, and POS+PFD
            p_me_norm = psd_me / (np.sum(psd_me) + 1e-12)
            p_bf_norm = psd_bf_interp / (np.sum(psd_bf_interp) + 1e-12)
            p_pos_norm = psd_pos / (np.sum(psd_pos) + 1e-12)

            w_me = float(np.clip(10.0 ** (snr_me / 10.0), 0.05, 50.0))
            w_bf = float(np.clip(10.0 ** (snr_bf / 10.0), 0.05, 50.0))
            w_pos = float(np.clip(10.0 ** (snr_pos / 10.0), 0.05, 50.0))
            tot_w = w_me + w_bf + w_pos
            weight_me = w_me / tot_w
            weight_bf = (w_bf + w_pos) / tot_w

            fused_psd = (w_me * p_me_norm + w_bf * p_bf_norm + w_pos * p_pos_norm) / tot_w
            fused_bvp = (w_me * cleaned_bvp + w_bf * bvp_bf[:t_len] + w_pos * pos_bvp[:t_len]) / tot_w

            peak_idx = int(np.argmax(fused_psd))
            # Parabolic sub-bin interpolation on fused spectrum
            final_hr = float(freqs_czt[peak_idx])
            if 0 < peak_idx < len(fused_psd) - 1:
                a_p, b_p, g_p = fused_psd[peak_idx - 1], fused_psd[peak_idx], fused_psd[peak_idx + 1]
                denom = a_p - 2.0 * b_p + g_p
                if abs(denom) > 1e-12:
                    delta = 0.5 * (a_p - g_p) / denom
                    dbpm = float(freqs_czt[1] - freqs_czt[0])
                    final_hr = float(freqs_czt[peak_idx] + delta * dbpm)
        else:
            final_hr = float(hr_me)
            fused_bvp = cleaned_bvp
            fused_psd = psd_me
            weight_me = 1.0
            weight_bf = 0.0

        res = {
            "model": "Hybrid v3 (ME-rPPG + PFD + BeatFormer-SCL + CZT)",
            "raw_bvp": raw_bvp,
            "processed_signal": fused_bvp,
            "hr_bpm": final_hr,
            "hr_series": hr_series_me,
            "branch_weights": {"merppg": weight_me, "beatformer": weight_bf},
            "fs": self.fs,
            "enable_pfd": self.enable_pfd,
            "enable_czt": self.enable_czt,
            "enable_beatformer": self.enable_beatformer,
        }
        if return_spectrum:
            res["spectrum_freqs"] = freqs_czt
            res["spectrum_power"] = fused_psd

        return res
