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

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

from models.deep.merppg_engine import MERPPGEngine
from models.hybrid.pfd_module import PhysiologicalFrequencyDenoiser
if _HAS_TORCH:
    from models.hybrid.beatformer import BeatFormer
else:
    BeatFormer = None
from models.classical.pos import pos_process
from models.classical.chrom import chrom_process
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
        if _HAS_TORCH and BeatFormer is not None:
            self.beatformer = BeatFormer(window_len=150, num_bins=128, fs=self.fs, low_hz=0.7, high_hz=3.0)
            if os.path.exists(beatformer_weights_path):
                try:
                    raw_sd = torch.load(beatformer_weights_path, map_location="cpu")
                    # Preserve exact fs-specific CZT/ICZT Vandermonde buffers when self.fs != 30.0
                    filtered_sd = {
                        k: v for k, v in raw_sd.items()
                        if not k.startswith(("czt.", "iczt."))
                    }
                    self.beatformer.load_state_dict(filtered_sd, strict=False)
                    self.beatformer.eval()
                except Exception:
                    pass
        else:
            self.beatformer = None

    def process(
        self,
        cropped_faces: np.ndarray,
        rgb_signals: Optional[np.ndarray] = None,
        return_spectrum: bool = False,
    ) -> Dict[str, any]:
        """Process video through dual-branch hybrid pipeline."""
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

        if rgb_signals is not None:
            rgb_trace = rgb_signals
        else:
            rgb_trace = np.mean(cropped_faces, axis=(1, 2)) * 255.0

        # Compute PFD-denoised POS and CHROM chrominance anchors
        pos_bvp = pos_process(rgb_trace, fs=self.fs)
        chrom_bvp = chrom_process(rgb_trace, fs=self.fs)
        if self.enable_pfd:
            pos_bvp, _, _ = self.pfd.denoise_spectrum(pos_bvp)
            chrom_bvp, _, _ = self.pfd.denoise_spectrum(chrom_bvp)
        hr_pos, _, psd_pos = self.czt.estimate_hr(pos_bvp, return_spectrum=True)
        hr_chrom, _, psd_chrom = self.czt.estimate_hr(chrom_bvp, return_spectrum=True)
        snr_pos = self.czt.compute_snr_db(psd_pos, freqs_czt, hr_pos)
        snr_chrom = self.czt.compute_snr_db(psd_chrom, freqs_czt, hr_chrom)

        # Branch 2: BeatFormer Spectral Transformer + Chrominance Anchor Fusion
        if self.enable_beatformer and self.beatformer is not None:
            hr_bf, bvp_bf, psd_bf = self.beatformer.estimate_hr_from_trace(rgb_trace)
            freqs_bf_bpm = self.beatformer.czt.freq_axis.cpu().numpy() * 60.0

            psd_bf_interp = np.interp(freqs_czt, freqs_bf_bpm, psd_bf)
            snr_bf = self.czt.compute_snr_db(psd_bf_interp, freqs_czt, hr_bf)

            p_me_norm = psd_me / (np.sum(psd_me) + 1e-12)
            p_bf_norm = psd_bf_interp / (np.sum(psd_bf_interp) + 1e-12)
            p_pos_norm = psd_pos / (np.sum(psd_pos) + 1e-12)
            p_chrom_norm = psd_chrom / (np.sum(psd_chrom) + 1e-12)

            w_me = float(np.clip(10.0 ** (snr_me / 10.0), 0.05, 50.0))
            w_bf = float(np.clip(10.0 ** (snr_bf / 10.0), 0.05, 50.0))
            w_pos = float(np.clip(10.0 ** (snr_pos / 10.0), 0.05, 50.0))
            w_chrom = float(np.clip(10.0 ** (snr_chrom / 10.0), 0.05, 50.0))
            tot_w = w_me + w_bf + w_pos + w_chrom
            weight_me = w_me / tot_w
            weight_bf = (w_bf + w_pos + w_chrom) / tot_w

            fused_psd = (
                w_me * p_me_norm + w_bf * p_bf_norm + w_pos * p_pos_norm + w_chrom * p_chrom_norm
            ) / tot_w
            fused_bvp = (
                w_me * cleaned_bvp
                + w_bf * bvp_bf[:t_len]
                + w_pos * pos_bvp[:t_len]
                + w_chrom * chrom_bvp[:t_len]
            ) / tot_w

            core_mask = (freqs_czt >= 60.0) & (freqs_czt <= 145.0)
            low_mask = freqs_czt < 60.0
            if np.any(core_mask) and np.any(low_mask):
                if np.max(fused_psd[core_mask]) >= 0.20 * np.max(fused_psd):
                    fused_psd[low_mask] *= 0.10

            peak_idx = int(np.argmax(fused_psd))
            final_hr = float(freqs_czt[peak_idx])
            if 0 < peak_idx < len(fused_psd) - 1:
                a_p, b_p, g_p = fused_psd[peak_idx - 1], fused_psd[peak_idx], fused_psd[peak_idx + 1]
                denom = a_p - 2.0 * b_p + g_p
                if abs(denom) > 1e-12:
                    delta = 0.5 * (a_p - g_p) / denom
                    dbpm = float(freqs_czt[1] - freqs_czt[0])
                    final_hr = float(freqs_czt[peak_idx] + delta * dbpm)
        elif rgb_signals is not None and self.enable_pfd:
            p_me_norm = psd_me / (np.sum(psd_me) + 1e-12)
            p_pos_norm = psd_pos / (np.sum(psd_pos) + 1e-12)
            p_chrom_norm = psd_chrom / (np.sum(psd_chrom) + 1e-12)
            w_me = float(np.clip(10.0 ** (snr_me / 10.0), 0.05, 50.0))
            w_pos = float(np.clip(10.0 ** (snr_pos / 10.0), 0.05, 50.0)) * 0.5
            w_chrom = float(np.clip(10.0 ** (snr_chrom / 10.0), 0.05, 50.0)) * 0.5
            tot_w = w_me + w_pos + w_chrom
            fused_psd = (w_me * p_me_norm + w_pos * p_pos_norm + w_chrom * p_chrom_norm) / tot_w
            fused_bvp = (w_me * cleaned_bvp + w_pos * pos_bvp[:t_len] + w_chrom * chrom_bvp[:t_len]) / tot_w
            peak_idx = int(np.argmax(fused_psd))
            final_hr = float(freqs_czt[peak_idx])
            if 0 < peak_idx < len(fused_psd) - 1:
                a_p, b_p, g_p = fused_psd[peak_idx - 1], fused_psd[peak_idx], fused_psd[peak_idx + 1]
                denom = a_p - 2.0 * b_p + g_p
                if abs(denom) > 1e-12:
                    delta = 0.5 * (a_p - g_p) / denom
                    dbpm = float(freqs_czt[1] - freqs_czt[0])
                    final_hr = float(freqs_czt[peak_idx] + delta * dbpm)
            weight_me = w_me / tot_w
            weight_bf = (w_pos + w_chrom) / tot_w
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

