"""Pipeline v3/v4: Hybrid Innovation rPPG System.

Implements the Stage 3C & Stage 4 verified Majority-Consensus Hybrid Architecture:
1. Spatial-Temporal State-Space Backbone: ME-rPPG (Wang et al., 2025)
2. Fundamental-Verified Spectral Denoising: FreqPhys PFD (Qian et al., ECCV 2026)
3. Chrominance & Plane-Orthogonal Anchors + Spectral Transformer: CHROM + POS + BeatFormer
4. Consensus-Verified Spectral Fusion & 512-Bin CZT Estimation
5. Nyquist-Shannon Cubic Temporal Sample-Rate Reconstruction for low-fps (e.g., C5 10 fps) streams
"""

import os
from typing import Dict, Tuple, Optional, Union
import numpy as np
from scipy.interpolate import interp1d

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
from models.classical.chrom import chrom_process
from models.classical.pos import pos_process
from estimation.czt_estimator import CZTHRTransformer


class HybridRPPGSystem:
    """End-to-End Hybrid rPPG System with Stage 3C/4 verified consensus spectral fusion."""

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
        if self.fs < 20.0:
            self._upsample_factor = max(2, int(round(30.0 / self.fs)))
        else:
            self._upsample_factor = 1
        self.eff_fs = self.fs * self._upsample_factor

        self.enable_pfd = bool(enable_pfd)
        self.enable_czt = bool(enable_czt)
        self.enable_beatformer = bool(enable_beatformer)

        self.backbone = MERPPGEngine(
            onnx_model_path=onnx_model_path,
            state_json_path=state_json_path,
            fs=self.eff_fs,
        )
        self.pfd = PhysiologicalFrequencyDenoiser(fs=self.eff_fs, low_hz=0.7, high_hz=3.0)
        self.czt = CZTHRTransformer(fs=self.eff_fs, low_hz=0.7, high_hz=3.0, num_bins=512)

        if _HAS_TORCH and BeatFormer is not None:
            self.beatformer = BeatFormer(window_len=150, num_bins=128, fs=self.eff_fs, low_hz=0.7, high_hz=3.0)
            self.beatformer.eval()
        else:
            self.beatformer = None

    def process(
        self,
        cropped_faces: np.ndarray,
        rgb_signals: Optional[np.ndarray] = None,
        return_spectrum: bool = False,
    ) -> Dict[str, any]:
        """Process video through Stage 3C/4 verified consensus hybrid pipeline."""
        orig_len = len(cropped_faces)

        if rgb_signals is not None:
            rgb_trace_in = np.asarray(rgb_signals, dtype=np.float64)
        else:
            rgb_trace_in = np.mean(cropped_faces, axis=(1, 2)).astype(np.float64) * 255.0

        if self._upsample_factor > 1 and orig_len >= 8:
            up_len = orig_len * self._upsample_factor
            t_in = np.arange(orig_len, dtype=np.float64) / self.fs
            t_up = np.arange(up_len, dtype=np.float64) / self.eff_fs
            kind = "cubic" if orig_len >= 16 else "linear"
            faces_proc = np.clip(
                interp1d(t_in, cropped_faces, axis=0, kind=kind, bounds_error=False, fill_value="extrapolate")(t_up),
                0.0,
                1.0,
            ).astype(np.float32)
            rgb_trace = interp1d(
                t_in, rgb_trace_in, axis=0, kind=kind, bounds_error=False, fill_value="extrapolate"
            )(t_up).astype(np.float64)
        else:
            faces_proc = cropped_faces
            rgb_trace = rgb_trace_in

        t_len = len(faces_proc)

        # Branch 1: Deep State-Space Backbone (ME-rPPG) + Fundamental-Verified PFD
        bb_out = self.backbone.process_frames(faces_proc, return_spectrum=False)
        raw_bvp = bb_out["bvp"]
        hr_series_me = bb_out.get("hr_series", np.full(t_len, bb_out["hr_bpm"]))

        if self.enable_pfd:
            cleaned_bvp, _, _ = self.pfd.denoise_spectrum(raw_bvp)
        else:
            cleaned_bvp = raw_bvp

        hr_me, freqs_czt, psd_me = self.czt.estimate_hr(cleaned_bvp, return_spectrum=True)

        # Branch 2: Stage 3C Verified Consensus Spectral Refinement with CHROM + POS + BeatFormer
        if self.enable_beatformer and self.beatformer is not None:
            bvp_chr = chrom_process(rgb_trace, fs=self.eff_fs)
            hr_chr, _, psd_chr = self.czt.estimate_hr(bvp_chr, return_spectrum=True)

            bvp_pos = pos_process(rgb_trace, fs=self.eff_fs)
            hr_pos, _, psd_pos = self.czt.estimate_hr(bvp_pos, return_spectrum=True)

            hr_bf, bvp_bf, psd_bf_raw = self.beatformer.estimate_hr_from_trace(rgb_trace)
            freqs_bf_bpm = self.beatformer.czt.freq_axis.cpu().numpy() * 60.0
            psd_bf = np.interp(freqs_czt, freqs_bf_bpm, psd_bf_raw)
            hr_bf = float(freqs_czt[int(np.argmax(psd_bf))])

            p_me_norm = psd_me / (np.max(psd_me) + 1e-12)
            p_chr_norm = psd_chr / (np.max(psd_chr) + 1e-12)
            p_pos_norm = psd_pos / (np.max(psd_pos) + 1e-12)
            p_bf_norm = psd_bf / (np.max(psd_bf) + 1e-12)

            w_me = 1.0
            w_chr = 0.25 if abs(hr_chr - hr_me) <= 3.5 else 0.0
            w_pos = 0.06 if abs(hr_pos - hr_me) <= 3.5 else 0.0
            w_bf = 0.15 if abs(hr_bf - hr_me) <= 3.5 else 0.0

            tot_w = w_me + w_chr + w_pos + w_bf
            weight_me = float(w_me / tot_w)
            weight_bf = float((w_chr + w_pos + w_bf) / tot_w)

            fused_psd = (
                w_me * p_me_norm
                + w_chr * p_chr_norm
                + w_pos * p_pos_norm
                + w_bf * p_bf_norm
            ) / tot_w
            fused_bvp = (
                w_me * cleaned_bvp
                + w_chr * bvp_chr[:t_len]
                + w_pos * bvp_pos[:t_len]
                + w_bf * bvp_bf[:t_len]
            ) / tot_w

            peak_idx = int(np.argmax(fused_psd))
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

        if self._upsample_factor > 1 and len(fused_bvp) != orig_len:
            raw_bvp = raw_bvp[::self._upsample_factor][:orig_len]
            fused_bvp = fused_bvp[::self._upsample_factor][:orig_len]
            hr_series_me = hr_series_me[::self._upsample_factor][:orig_len]

        res = {
            "model": "Hybrid v4 (ME-rPPG + PFD + BeatFormer-SCL + CHROM + CZT)",
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
