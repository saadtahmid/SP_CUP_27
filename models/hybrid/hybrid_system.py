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
        verbose: bool = False,
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
        self.verbose = bool(verbose)

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
        hr_me_raw, freqs_czt, psd_raw = self.czt.estimate_hr(raw_bvp, return_spectrum=True)

        defection_fired = False
        pfd_cap_fired = False

        if self.enable_pfd:
            cleaned_bvp, _, _ = self.pfd.denoise_spectrum(raw_bvp)
            hr_pfd = float(self.czt.estimate_hr(cleaned_bvp))
            shift = abs(hr_pfd - hr_me_raw)
            ratio = hr_pfd / max(1e-6, hr_me_raw)

            # Detect integer sub-harmonic or harmonic comb artifacts:
            # Covers halving (0.5x), doubling (2.0x), 2/3 comb, and 3/2 comb
            is_harmonic_jump = (
                (0.45 <= ratio <= 0.55) or
                (1.80 <= ratio <= 2.20) or
                (0.62 <= ratio <= 0.72) or
                (1.40 <= ratio <= 1.60)
            )

            # Unit-proof raw anchor concentration calculation (supports Hz or BPM freqs):
            is_hz = (freqs_czt[-1] < 10.0)
            bin_hz = float(freqs_czt[1] - freqs_czt[0]) if is_hz else float(freqs_czt[1] - freqs_czt[0]) / 60.0
            bin_rad = max(1, int(round((6.0 / 60.0) / bin_hz)))
            raw_peak_idx = int(np.argmax(psd_raw))
            i_lo = max(0, raw_peak_idx - bin_rad)
            i_hi = min(len(psd_raw), raw_peak_idx + bin_rad + 1)
            raw_conc = float(np.sum(psd_raw[i_lo:i_hi])) / (float(np.sum(psd_raw)) + 1e-12)

            pfd_reject_reason = None
            if is_harmonic_jump:
                accept_pfd = False
                pfd_reject_reason = f"harmonic_jump (ratio={ratio:.2f})"
            elif shift <= 8.0:
                accept_pfd = True
            elif shift <= 18.0 and raw_conc < 0.35:
                accept_pfd = True
            else:
                accept_pfd = False
                if shift > 18.0:
                    pfd_reject_reason = f"shift_too_large (|Δ|={shift:.1f} > 18 BPM)"
                else:
                    pfd_reject_reason = f"anchor_confident (|Δ|={shift:.1f} > 8 BPM and raw_conc={raw_conc:.2f} >= 0.35)"

            if accept_pfd:
                hr_me = hr_pfd
                psd_me = self.czt.compute_spectrum(cleaned_bvp)[1]
            else:
                pfd_cap_fired = True
                if self.verbose:
                    print(
                        f"  [PFD CAP FIRED] Harmonic shift capped: raw_me={hr_me_raw:.1f}, "
                        f"pfd={hr_pfd:.1f} (|Δ|={shift:.1f}, ratio={ratio:.2f}, raw_conc={raw_conc:.2f})"
                    )
                hr_me = hr_me_raw
                cleaned_bvp = raw_bvp
                psd_me = psd_raw
        else:
            cleaned_bvp = raw_bvp
            hr_me = hr_me_raw
            hr_pfd = hr_me_raw
            shift = 0.0
            accept_pfd = False
            pfd_reject_reason = "pfd_disabled"
            raw_conc = None
            psd_me = psd_raw

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

            # 1. Pairwise Independent-Branch Defection (CHROM ∧ BeatFormer):
            pairwise_agree = abs(hr_chr - hr_bf) <= 4.0
            consensus_hr = 0.5 * (hr_chr + hr_bf)

            # Comprehensive Harmonic Exclusion Guard (1/2, 2x, 2/3, 3/2):
            harmonic_of_anchor = (
                abs(consensus_hr - 0.5 * hr_me) <= 3.0 or
                abs(consensus_hr - 2.0 * hr_me) <= 3.0 or
                abs(consensus_hr - (2.0 / 3.0) * hr_me) <= 3.0 or
                abs(consensus_hr - 1.5 * hr_me) <= 3.0
            )
            if harmonic_of_anchor:
                pairwise_agree = False

            anchor_far = abs(hr_me - consensus_hr) > 3.5

            # Dual Guard: Anchor confirmation by POS prevents false defection when anchor is valid
            pos_confirms_anchor = (abs(hr_pos - hr_me) <= 3.5)

            # 2. In-band spectral concentration quality gate with luminance calibration:
            chr_peak_idx = int(np.argmax(psd_chr))
            bin_radius = int(round(6.0 / (freqs_czt[1] - freqs_czt[0])))
            i_lo_chr = max(0, chr_peak_idx - bin_radius)
            i_hi_chr = min(len(psd_chr), chr_peak_idx + bin_radius + 1)
            conc_chr = float(np.sum(psd_chr[i_lo_chr:i_hi_chr])) / (float(np.sum(psd_chr)) + 1e-12)

            bf_peak_idx = int(np.argmax(psd_bf))
            i_lo_bf = max(0, bf_peak_idx - bin_radius)
            i_hi_bf = min(len(psd_bf), bf_peak_idx + bin_radius + 1)
            conc_bf = float(np.sum(psd_bf[i_lo_bf:i_hi_bf])) / (float(np.sum(psd_bf)) + 1e-12)

            skin_adc = float(np.mean(rgb_trace)) if rgb_trace is not None else 100.0
            conf_thresh = 0.30 if skin_adc < 35.0 else 0.15
            secondary_conf = max(conc_chr, conc_bf) >= conf_thresh

            defection_reason = None
            if pairwise_agree and anchor_far and secondary_conf and not pos_confirms_anchor:
                defection_fired = True
                defection_reason = "pairwise_chrom_beatformer_consensus"
                if self.verbose:
                    print(f"  [DEFECTION FIRING] Secondary consensus accepted: CHROM={hr_chr:.1f}, BF={hr_bf:.1f} vs Anchor ME={hr_me:.1f} (Conc={max(conc_chr, conc_bf):.2f})")
                w_me = 0.10
                w_chr = 0.65
                w_bf = 0.25
                w_pos = 0.00
                final_hr = float(w_me * hr_me + w_chr * hr_chr + w_bf * hr_bf)
                # BVP waveform synthesized from concurring secondary pair to maintain BVP CZT peak consistency
                fused_bvp = (0.70 * bvp_chr[:t_len] + 0.30 * bvp_bf[:t_len])
                fused_psd = (0.70 * p_chr_norm + 0.30 * p_bf_norm)
            else:
                if pairwise_agree and anchor_far and pos_confirms_anchor:
                    defection_reason = f"pos_confirms_anchor (|hr_pos - hr_me|={abs(hr_pos - hr_me):.2f} <= 3.5)"
                elif pairwise_agree and anchor_far and not secondary_conf:
                    defection_reason = f"secondary_too_diffuse (max_conc={max(conc_chr, conc_bf):.2f} < {conf_thresh:.2f})"
                elif pairwise_agree and not anchor_far:
                    defection_reason = f"anchor_not_isolated (|hr_me - consensus_hr|={abs(hr_me - consensus_hr):.2f} <= 3.5)"
                elif not pairwise_agree and harmonic_of_anchor:
                    defection_reason = f"harmonic_of_anchor (consensus={consensus_hr:.1f}, anchor={hr_me:.1f})"
                elif not pairwise_agree and anchor_far:
                    defection_reason = f"no_pairwise_agree (|chr-bf|={abs(hr_chr-hr_bf):.1f} > 4.0)"
                else:
                    defection_reason = "anchor_concurrence"
                w_me = 1.00
                w_chr = 0.25 if abs(hr_chr - hr_me) <= 3.5 else 0.0
                w_pos = 0.06 if abs(hr_pos - hr_me) <= 3.5 else 0.0
                w_bf = 0.15 if abs(hr_bf - hr_me) <= 3.5 else 0.0

                tot_w = w_me + w_chr + w_pos + w_bf
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

            tot_w = w_me + w_chr + w_pos + w_bf
            weight_me = float(w_me / tot_w)
            weight_bf = float((w_chr + w_pos + w_bf) / tot_w)
        else:
            final_hr = float(hr_me)
            fused_bvp = cleaned_bvp
            fused_psd = psd_me
            weight_me = 1.0
            weight_bf = 0.0
            hr_chr = None
            hr_pos = None
            hr_bf = None
            w_me = 1.0
            w_chr = 0.0
            w_pos = 0.0
            w_bf = 0.0
            consensus_hr = None
            pairwise_agree = False
            anchor_far = False
            pos_confirms_anchor = False
            secondary_conf = False
            conc_chr = None
            conc_bf = None
            defection_reason = "beatformer_disabled"

        if self._upsample_factor > 1 and len(fused_bvp) != orig_len:
            raw_bvp = raw_bvp[::self._upsample_factor][:orig_len]
            fused_bvp = fused_bvp[::self._upsample_factor][:orig_len]
            hr_series_me = hr_series_me[::self._upsample_factor][:orig_len]

        decision_log = {
            "hr_me_raw": float(hr_me_raw),
            "hr_me_pfd": float(hr_pfd),
            "pfd_shift": float(hr_pfd - hr_me_raw),
            "pfd_accepted": bool(accept_pfd),
            "pfd_reject_reason": pfd_reject_reason,
            "raw_conc": float(raw_conc) if raw_conc is not None else None,
            "hr_chrom": float(hr_chr) if hr_chr is not None else None,
            "hr_pos": float(hr_pos) if hr_pos is not None else None,
            "hr_bf": float(hr_bf) if hr_bf is not None else None,
            "consensus_hr": float(consensus_hr) if consensus_hr is not None else None,
            "branch_weights": {
                "me": float(w_me),
                "chrom": float(w_chr),
                "pos": float(w_pos),
                "bf": float(w_bf),
            },
            "branch_admission": {
                "chrom": bool(w_chr > 0),
                "pos": bool(w_pos > 0),
                "bf": bool(w_bf > 0),
            },
            "defection_fired": bool(defection_fired),
            "defection_reason": defection_reason,
            "pairwise_agree": bool(pairwise_agree),
            "anchor_isolated": bool(anchor_far),
            "pos_confirms_anchor": bool(pos_confirms_anchor),
            "confident_secondary": bool(secondary_conf),
            "conc_chr": float(conc_chr) if conc_chr is not None else None,
            "conc_bf": float(conc_bf) if conc_bf is not None else None,
            "final_hr": float(final_hr),
        }

        res = {
            "model": "Hybrid v4 (ME-rPPG + PFD + BeatFormer-SCL + CHROM + CZT)",
            "raw_bvp": raw_bvp,
            "processed_signal": fused_bvp,
            "hr_bpm": final_hr,
            "hr_anchor_bpm": float(hr_me),
            "hr_me_raw_bpm": float(hr_me_raw),
            "hr_series": hr_series_me,
            "branch_weights": {"merppg": weight_me, "beatformer": weight_bf},
            "fs": self.fs,
            "enable_pfd": self.enable_pfd,
            "enable_czt": self.enable_czt,
            "enable_beatformer": self.enable_beatformer,
            "defection_fired": defection_fired,
            "pfd_cap_fired": pfd_cap_fired,
            "decision_log": decision_log,
        }
        if return_spectrum:
            res["spectrum_freqs"] = freqs_czt
            res["spectrum_power"] = fused_psd

        return res
