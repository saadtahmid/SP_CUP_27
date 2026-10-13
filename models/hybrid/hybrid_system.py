"""Pipeline v3: Hybrid Innovation rPPG System.

Glues:
1. Backbone: ME-rPPG Temporal-Spatial State Space Duality backbone
2. Spectral Refinement: FreqPhys Physiological Frequency-prior Denoiser (PFD)
3. HR Estimation: BeatFormer Chirp-Z Transform (CZT) Zoomed Peak Detector
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np

from models.deep.merppg_engine import MERPPGEngine
from models.hybrid.pfd_module import PhysiologicalFrequencyDenoiser
from estimation.czt_estimator import CZTHRTransformer


class HybridRPPGSystem:
    """End-to-End Hybrid rPPG System (ME-rPPG + PFD + CZT)."""

    def __init__(
        self,
        onnx_model_path: str = "third_party/ME-rPPG/model.onnx",
        state_json_path: str = "third_party/ME-rPPG/state.json",
        fs: float = 30.0,
        enable_pfd: bool = True,
        enable_czt: bool = True,
    ):
        self.fs = float(fs)
        self.enable_pfd = bool(enable_pfd)
        self.enable_czt = bool(enable_czt)

        # 1. State-Space Deep Backbone
        self.backbone = MERPPGEngine(
            onnx_model_path=onnx_model_path,
            state_json_path=state_json_path,
            fs=self.fs,
        )

        # 2. PFD Spectral Denoiser
        self.pfd = PhysiologicalFrequencyDenoiser(fs=self.fs, low_hz=0.7, high_hz=3.0)

        # 3. CZT HR Estimator
        self.czt = CZTHRTransformer(fs=self.fs, low_hz=0.7, high_hz=3.0)

    def process(
        self,
        cropped_faces: np.ndarray,
        return_spectrum: bool = False,
    ) -> Dict[str, any]:
        """Process cropped face sequence through hybrid pipeline.

        Args:
            cropped_faces: (T, 36, 36, 3) normalized float32 facial frames.
            return_spectrum: whether to return power spectrum.

        Returns:
            dict containing raw BVP, cleaned BVP, estimated HR, and diagnostics.
        """
        # Step 1: Deep Backbone inference
        bb_out = self.backbone.process_frames(cropped_faces, return_spectrum=False)
        raw_bvp = bb_out["bvp"]

        # Step 2: PFD Spectral Denoiser
        if self.enable_pfd:
            cleaned_bvp, _, _ = self.pfd.denoise_spectrum(raw_bvp)
        else:
            cleaned_bvp = raw_bvp

        # Step 3: CZT HR Estimation
        if self.enable_czt:
            if return_spectrum:
                hr_bpm, freqs, power = self.czt.estimate_hr(cleaned_bvp, return_spectrum=True)
            else:
                hr_bpm = self.czt.estimate_hr(cleaned_bvp, return_spectrum=False)
                freqs, power = None, None
        else:
            # Baseline Welch FFT
            from estimation.fft_estimator import estimate_hr_welch
            hr_bpm = estimate_hr_welch(cleaned_bvp, fs=self.fs)
            freqs, power = None, None

        res = {
            "model": "Hybrid (ME-rPPG + PFD + CZT)",
            "raw_bvp": raw_bvp,
            "processed_signal": cleaned_bvp,
            "hr_bpm": float(hr_bpm),
            "fs": self.fs,
            "enable_pfd": self.enable_pfd,
            "enable_czt": self.enable_czt,
        }
        if return_spectrum:
            res["spectrum_freqs"] = freqs
            res["spectrum_power"] = power

        return res
