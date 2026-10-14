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

    def estimate_hr_windowed(
        self,
        bvp_series: np.ndarray,
        win_sec: float = 10.0,
        stride_sec: float = 1.0,
    ) -> Tuple[float, np.ndarray]:
        """Compute windowed HR series with harmonic and jump outlier rejection.

        Synthesized from frPPG (Section III-B3) and BeatFormer.

        Args:
            bvp_series: 1D temporal BVP signal.
            win_sec: Window length in seconds (default 10s).
            stride_sec: Stride in seconds (default 1s).

        Returns:
            global_hr: Robust global median HR in BPM.
            hr_series: Frame-level interpolated HR series (T,).
        """
        win_len = int(round(win_sec * self.fs))
        stride_len = int(round(stride_sec * self.fs))
        t_len = len(bvp_series)

        if t_len < win_len:
            single_hr = float(self.czt_estimator.estimate_hr(bvp_series))
            return single_hr, np.full(t_len, single_hr, dtype=np.float64)

        starts = list(range(0, t_len - win_len + 1, stride_len))
        if not starts:
            starts = [0]

        window_hrs = []
        window_snrs = []
        window_centers = []

        for s in starts:
            segment = bvp_series[s : s + win_len]
            hr, freqs, pxx = self.czt_estimator.estimate_hr(segment, return_spectrum=True)
            snr_db = self.czt_estimator.compute_snr_db(pxx, freqs, hr)

            window_hrs.append(hr)
            window_snrs.append(snr_db)
            window_centers.append(s + win_len // 2)

        window_hrs = np.array(window_hrs, dtype=np.float64)
        window_snrs = np.array(window_snrs, dtype=np.float64)

        running_med = float(np.median(window_hrs))
        valid_hrs = []
        valid_centers = []

        snr_thresh = np.percentile(window_snrs, 15) if len(window_snrs) >= 6 else -999.0

        for idx, (hr_cand, snr_cand, center) in enumerate(zip(window_hrs, window_snrs, window_centers)):
            if snr_cand < snr_thresh and len(window_hrs) >= 6:
                continue

            # Check jump from running median
            if abs(hr_cand - running_med) > 15.0:
                half_hr = hr_cand / 2.0
                double_hr = hr_cand * 2.0
                if abs(half_hr - running_med) < 8.0 and half_hr >= 42.0:
                    hr_cand = half_hr
                elif abs(double_hr - running_med) < 8.0 and double_hr <= 180.0:
                    hr_cand = double_hr

            valid_hrs.append(hr_cand)
            valid_centers.append(center)

        if not valid_hrs:
            valid_hrs = window_hrs.tolist()
            valid_centers = window_centers

        global_hr = float(np.median(valid_hrs))

        if len(valid_centers) > 1:
            hr_series = np.interp(np.arange(t_len), valid_centers, valid_hrs)
        else:
            hr_series = np.full(t_len, global_hr, dtype=np.float64)

        return global_hr, hr_series

    def process_frames(
        self,
        cropped_faces: np.ndarray,
        return_spectrum: bool = False,
    ) -> Dict[str, any]:
        """Process a sequence of cropped face frames (T, 36, 36, 3).

        Applies 45-frame RMA state pre-warming, linear detrending, and sliding-window HR estimation.

        Args:
            cropped_faces: (T, 36, 36, 3) normalized float32 RGB faces.
            return_spectrum: Whether to return CZT spectrum arrays.

        Returns:
            dict containing:
            - bvp: 1D array of extracted BVP values
            - hr_bpm: estimated robust global HR via CZT + windowing
            - hr_series: frame-level estimated HR series
            - spectrum_freqs (optional)
            - spectrum_power (optional)
        """
        self.reset_state()
        t_len = len(cropped_faces)
        bvp_series = np.empty(t_len, dtype=np.float64)

        # Pre-warm recurrent state on initial frames to discard RMA warm-up transient
        warmup = min(45, t_len)
        for i in range(warmup):
            _ = self.step(cropped_faces[i])

        # Execute full frame sequence
        for i in range(t_len):
            bvp_series[i] = self.step(cropped_faces[i])

        # Smooth initial warmup transient to avoid CZT spectrum pollution
        if warmup < t_len:
            baseline_val = float(np.mean(bvp_series[warmup : min(warmup + 30, t_len)]))
            bvp_series[:warmup] = baseline_val

        # Sliding-window HR estimation with outlier rejection
        if t_len >= int(10.0 * self.fs):
            hr_bpm, hr_series = self.estimate_hr_windowed(bvp_series)
        else:
            hr_bpm = float(self.czt_estimator.estimate_hr(bvp_series))
            hr_series = np.full(t_len, hr_bpm, dtype=np.float64)

        # Full-window spectrum for diagnostic output
        if return_spectrum:
            _, freqs, power = self.czt_estimator.estimate_hr(bvp_series, return_spectrum=True)
        else:
            freqs, power = None, None

        res = {
            "model": "ME-rPPG",
            "bvp": bvp_series,
            "hr_bpm": float(hr_bpm),
            "hr_series": hr_series,
            "fs": self.fs,
        }
        if return_spectrum:
            res["spectrum_freqs"] = freqs
            res["spectrum_power"] = power

        return res
