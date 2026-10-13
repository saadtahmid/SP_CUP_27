"""Pipeline v2: Memory-Efficient State-Space rPPG (ME-rPPG) Deep Learning Engine.

Wang, K., Tang, J., Fan, Y., et al. (2025). Memory-efficient Low-latency
Remote Photoplethysmography through Temporal-Spatial State Space Duality.

Properties:
- 580K parameters (vs PhysFormer 7.4M)
- 9.46 ms CPU inference latency
- Supports both chunk mode and single-frame streaming flow mode with internal state
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import onnxruntime as ort

from estimation.czt_estimator import CZTHRTransformer


class MERPPGEngine:
    """ME-rPPG deep learning inference engine using ONNX Runtime."""

    def __init__(
        self,
        onnx_model_path: str = "third_party/ME-rPPG/model.onnx",
        state_json_path: str = "third_party/ME-rPPG/state.json",
        fs: float = 30.0,
        device: str = "cpu",
    ):
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ME-rPPG model not found at {onnx_model_path}")
        if not os.path.exists(state_json_path):
            raise FileNotFoundError(f"ME-rPPG state not found at {state_json_path}")

        self.fs = float(fs)
        self.dt = float(1.0 / self.fs)

        # Configure ONNX Runtime execution provider
        providers = ["CPUExecutionProvider"]
        if device.lower() in ["cuda", "gpu"] and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(onnx_model_path, providers=providers)

        with open(state_json_path, "r") as f:
            self.base_state = json.load(f)

        self.current_state = {k: np.array(v, dtype=np.float32) for k, v in self.base_state.items()}
        self.state_keys = list(self.base_state.keys())

        self.czt_estimator = CZTHRTransformer(fs=self.fs, low_hz=0.7, high_hz=3.0)

    def reset_state(self):
        """Reset internal recurrence state to baseline."""
        self.current_state = {k: np.array(v, dtype=np.float32) for k, v in self.base_state.items()}

    def step(self, face_crop_36x36: np.ndarray) -> float:
        """Run single-frame streaming inference.

        Args:
            face_crop_36x36: (36, 36, 3) normalized float32 RGB face image [0, 1].

        Returns:
            bvp_val: Instantaneous scalar BVP output.
        """
        img = face_crop_36x36.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

        # Input shape expected: (1, 1, 36, 36, 3)
        inp_tensor = img[None, None]
        dt_tensor = np.array([self.dt], dtype=np.float32)

        feed_dict = {
            "arg_0.1": inp_tensor,
            "onnx::Mul_37": dt_tensor,
            **self.current_state,
        }

        outputs = self.session.run(None, feed_dict)
        bvp_val = float(outputs[0][0, 0])

        # Update recurrence state
        new_states = outputs[1:]
        self.current_state = dict(zip(self.state_keys, new_states))

        return bvp_val

    def process_frames(
        self,
        cropped_faces: np.ndarray,
        return_spectrum: bool = False,
    ) -> Dict[str, any]:
        """Process a sequence of cropped face frames (T, 36, 36, 3).

        Args:
            cropped_faces: (T, 36, 36, 3) normalized float32 RGB faces.
            return_spectrum: Whether to return CZT spectrum arrays.

        Returns:
            dict containing:
            - bvp: 1D array of extracted BVP values
            - hr_bpm: estimated HR via CZT
            - spectrum_freqs (optional)
            - spectrum_power (optional)
        """
        self.reset_state()
        t_len = len(cropped_faces)
        bvp_series = np.empty(t_len, dtype=np.float64)

        for i in range(t_len):
            bvp_series[i] = self.step(cropped_faces[i])

        # Estimate HR via CZT
        if return_spectrum:
            hr_bpm, freqs, power = self.czt_estimator.estimate_hr(bvp_series, return_spectrum=True)
        else:
            hr_bpm = self.czt_estimator.estimate_hr(bvp_series, return_spectrum=False)
            freqs, power = None, None

        res = {
            "model": "ME-rPPG",
            "bvp": bvp_series,
            "hr_bpm": float(hr_bpm),
            "fs": self.fs,
        }
        if return_spectrum:
            res["spectrum_freqs"] = freqs
            res["spectrum_power"] = power

        return res
