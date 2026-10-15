"""Dataset Loader for PURE (Pulse Rate Detection) Dataset.

Reference:
Stricker, R., Müller, S., Gross, H.-M. (2014).
Non-contact video-based pulse rate measurement on a mobile service robot.
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS).

Dataset Structure:
- 10 subjects (01 to 10)
- 6 protocols per subject:
    * 01: Steady (neutral frontal pose)
    * 02: Talking (speech / jaw and mouth motion)
    * 03: Slow translation (lateral / depth translation)
    * 04: Fast translation (rapid head movement)
    * 05: Small rotation (pitch / yaw rotation <= 20 deg)
    * 06: Medium rotation (pitch / yaw rotation <= 35 deg)
- Image sequences: PNG files (640x480 @ 30 fps) or compressed video (.avi / .mp4)
- Ground truth: JSON file with synchronized Pulox CMS50E pulse oximeter reference (60 Hz)
"""

import os
import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from preprocessing.roi_extractor import ROIExtractor, filter_skin_pixels
from preprocessing.skin_tone import analyze_skin_tone
from estimation.czt_estimator import CZTHRTransformer


PURE_PROTOCOLS = {
    "01": "Steady",
    "02": "Talking",
    "03": "Slow Translation",
    "04": "Fast Translation",
    "05": "Small Rotation",
    "06": "Medium Rotation",
}


class PUREDataLoader:
    """Loader for the PURE rPPG dataset."""

    def __init__(self, data_root: str = "data/PURE"):
        self.data_root = Path(data_root)
        self.sessions = self._discover_sessions()

    def _discover_sessions(self) -> List[Dict[str, Any]]:
        """Discover all valid PURE sessions in the dataset root."""
        if not self.data_root.exists():
            return []

        sessions = []
        # Find all directories matching session patterns (e.g. '01-01', '02-05')
        candidate_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir() and not d.name.startswith(".")])

        for s_dir in candidate_dirs:
            name = s_dir.name
            parts = name.split("-")
            if len(parts) >= 2:
                subj_id, protocol_code = parts[0], parts[1]
            else:
                subj_id, protocol_code = name, "01"

            protocol_name = PURE_PROTOCOLS.get(protocol_code, f"Protocol_{protocol_code}")

            # Locate JSON ground-truth file
            json_file = s_dir / f"{name}.json"
            if not json_file.exists():
                json_file = self.data_root / f"{name}.json"
            if not json_file.exists():
                alt_jsons = list(s_dir.glob("*.json"))
                json_file = alt_jsons[0] if alt_jsons else None

            # Locate video or image directory
            # PURE can have images in s_dir directly or in s_dir/name or a video file
            img_dir = s_dir / name if (s_dir / name).is_dir() else s_dir
            png_files = sorted(list(img_dir.glob("*.png")))
            if not png_files:
                png_files = sorted(list(s_dir.glob("*.png")))

            video_file = None
            for ext in [".avi", ".mp4", ".mkv"]:
                cand_vids = list(s_dir.glob(f"*{ext}"))
                if cand_vids:
                    video_file = cand_vids[0]
                    break

            has_data = len(png_files) > 0 or (video_file is not None and video_file.exists())
            has_gt = json_file is not None and json_file.exists()

            if has_gt or has_data:
                sessions.append({
                    "session_id": name,
                    "subject_id": subj_id,
                    "protocol_code": protocol_code,
                    "protocol_name": protocol_name,
                    "session_dir": s_dir,
                    "json_path": json_file,
                    "image_dir": img_dir,
                    "png_files": png_files,
                    "video_file": video_file,
                    "is_video": video_file is not None and len(png_files) == 0,
                })

        return sessions

    @staticmethod
    def parse_pure_json(
        json_path: Path,
        time_window: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """Parse PURE ground truth JSON file with optional timestamp windowing."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        hr_values = []
        wave_values = []
        timestamps = []

        start_t = time_window[0] if time_window else -float("inf")
        end_t = time_window[1] if time_window else float("inf")

        # Case 0: Official PURE /FullPackage structure
        if isinstance(data, dict) and "/FullPackage" in data and isinstance(data["/FullPackage"], list):
            for rec in data["/FullPackage"]:
                if isinstance(rec, dict):
                    ts = rec.get("Timestamp")
                    if ts is not None:
                        ts_val = float(ts)
                        if start_t <= ts_val <= end_t:
                            timestamps.append(ts_val)
                            val = rec.get("Value", {})
                            if isinstance(val, dict):
                                pr = val.get("pulseRate") or val.get("PulseRate")
                                if pr is not None and float(pr) > 30.0:
                                    hr_values.append(float(pr))
                                wf = val.get("waveform") or val.get("Waveform")
                                if wf is not None:
                                    wave_values.append(float(wf))

        # Case 1: dictionary containing image/timestamp records
        # E.g. {"/Image": [{"/Timestamp": ..., "/PulseRate": ...}]}
        image_records = None
        for key in ["/Image", "Image", "images", "frames", "data"]:
            if key in data and isinstance(data[key], list):
                image_records = data[key]
                break

        if len(hr_values) == 0 and image_records is not None:
            for rec in image_records:
                if isinstance(rec, dict):
                    ts = rec.get("/Timestamp") or rec.get("Timestamp") or rec.get("time")
                    ts_val = float(ts) if ts is not None else None
                    if ts_val is None or (start_t <= ts_val <= end_t):
                        if ts_val is not None:
                            timestamps.append(ts_val)
                        pr = rec.get("/PulseRate") or rec.get("PulseRate") or rec.get("pulse_rate") or rec.get("hr")
                        if pr is not None and float(pr) > 30.0:
                            hr_values.append(float(pr))
                        wf = rec.get("/Waveform") or rec.get("waveform") or rec.get("ppg") or rec.get("bvp")
                        if wf is not None:
                            wave_values.append(float(wf))

        # Case 2: Direct list of records
        elif len(hr_values) == 0 and isinstance(data, list):
            for rec in data:
                if isinstance(rec, dict):
                    ts = rec.get("Timestamp") or rec.get("time")
                    ts_val = float(ts) if ts is not None else None
                    if ts_val is None or (start_t <= ts_val <= end_t):
                        if ts_val is not None:
                            timestamps.append(ts_val)
                        pr = rec.get("PulseRate") or rec.get("pulse_rate") or rec.get("hr")
                        if pr is not None and float(pr) > 30.0:
                            hr_values.append(float(pr))
                        wf = rec.get("Waveform") or rec.get("waveform") or rec.get("ppg")
                        if wf is not None:
                            wave_values.append(float(wf))
                elif isinstance(rec, (int, float)):
                    if float(rec) > 30.0:
                        hr_values.append(float(rec))

        # Case 3: Flat keys
        elif len(hr_values) == 0 and isinstance(data, dict):
            for k in ["pulse_rate_bpm", "PulseRate", "mean_hr", "hr"]:
                if k in data and isinstance(data[k], (int, float)):
                    hr_values = [float(data[k])]
                    break
                elif k in data and isinstance(data[k], list):
                    hr_values = [float(x) for x in data[k] if float(x) > 30.0]
                    break

            for k in ["waveform", "ppg", "bvp", "Waveform"]:
                if k in data and isinstance(data[k], list):
                    wave_values = [float(x) for x in data[k]]
                    break

        hr_array = np.array(hr_values, dtype=np.float64)
        valid_hr = hr_array[hr_array > 35.0]
        mean_hr_bpm = float(np.mean(valid_hr)) if len(valid_hr) > 0 else 75.0

        ts_array = np.array(timestamps, dtype=np.float64)
        if len(ts_array) > 1:
            duration_s = (ts_array[-1] - ts_array[0])
            if duration_s > 1e6:
                duration_s /= 1e9
            fps = float(len(ts_array) - 1) / duration_s if duration_s > 0 else 30.0
        else:
            fps = 30.0

        wave_array = np.array(wave_values, dtype=np.float64)
        if len(wave_array) >= 128:
            czt = CZTHRTransformer(fs=fps, low_hz=0.7, high_hz=3.0, num_bins=512)
            czt_hr_bpm = float(czt.estimate_hr(wave_array))
        else:
            czt_hr_bpm = mean_hr_bpm

        return {
            "hr_series": hr_array,
            "mean_hr_bpm": mean_hr_bpm,
            "czt_hr_bpm": czt_hr_bpm,
            "waveform": wave_array,
            "timestamps": ts_array,
            "fps": fps,
        }

    def load_session_frames(
        self,
        session_info: Dict[str, Any],
        max_frames: Optional[int] = 450,
        return_time_window: bool = False,
    ):
        """Load BGR frames for a PURE session and return (frames, fps) or (frames, fps, time_window)."""
        frames = []
        fps = 30.0
        time_window = None

        if session_info.get("is_video") and session_info.get("video_file"):
            cap = cv2.VideoCapture(str(session_info["video_file"]))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            idx = 0
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret or (max_frames is not None and idx >= max_frames):
                    break
                frames.append(frame_bgr)
                idx += 1
            cap.release()
        else:
            png_files = session_info["png_files"]
            if max_frames is not None:
                png_files = png_files[:max_frames]

            if len(png_files) > 0:
                try:
                    ts_start = float(png_files[0].stem.replace("Image", ""))
                    ts_end = float(png_files[-1].stem.replace("Image", ""))
                    time_window = (ts_start, ts_end)
                except Exception:
                    time_window = None

            for pf in png_files:
                frame_bgr = cv2.imread(str(pf))
                if frame_bgr is not None:
                    frames.append(frame_bgr)

        if len(frames) == 0:
            frames_arr = np.zeros((0, 480, 640, 3), dtype=np.uint8)
        else:
            frames_arr = np.array(frames, dtype=np.uint8)

        if return_time_window:
            return frames_arr, float(fps), time_window
        return frames_arr, float(fps)
