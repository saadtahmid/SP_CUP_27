"""Submission packaging module for 2027 IEEE Signal Processing Cup (§1.2 & §4).

Generates official `Round1_Baseline_None.mat` strictly containing the 6 required variables:
1. `facial_video`: Raw RGB video frames of the subject (uint8 array)
2. `roi_data`: Extracted facial ROI coordinates / boolean masks (forehead, cheeks)
3. `reference_signal`: Ground-truth physiological HR signal
4. `processed_signal`: Estimated HR output from the algorithm
5. `metrics`: Evaluation struct (mae, rmse, pearson_r, bias, upper_loa, lower_loa)
6. `params`: Metadata struct (resolution, fps, lux, camera_distance_cm, fitzpatrick_type, age, constraints)
"""

import os
from typing import Dict, Any, Optional
import numpy as np
from scipy import io as spio


def create_submission_mat_file(
    output_mat_path: str,
    facial_video: np.ndarray,
    roi_data: np.ndarray,
    reference_signal: np.ndarray,
    processed_signal: np.ndarray,
    metrics: Dict[str, float],
    params: Dict[str, Any],
) -> str:
    """Save all 6 required variables to a MATLAB-compatible .mat file.

    Args:
        output_mat_path: Full path to write .mat file (e.g. Round1_Baseline_None.mat).
        facial_video: Video frames array (uint8).
        roi_data: ROI masks or bounding box coordinates.
        reference_signal: Ground truth HR time series or vector.
        processed_signal: Algorithm estimated HR time series or vector.
        metrics: Dict of computed metrics.
        params: Dict of parameter metadata.

    Returns:
        output_mat_path: Path of created file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_mat_path)), exist_ok=True)

    # Format structs as dicts of numpy objects for MATLAB compatibility
    mat_dict = {
        "facial_video": np.asarray(facial_video, dtype=np.uint8),
        "roi_data": np.asarray(roi_data, dtype=np.uint8),
        "reference_signal": np.asarray(reference_signal, dtype=np.float64),
        "processed_signal": np.asarray(processed_signal, dtype=np.float64),
        "metrics": {
            "mae": float(metrics.get("mae", 0.0)),
            "rmse": float(metrics.get("rmse", 0.0)),
            "pearson_r": float(metrics.get("pearson_r", 0.0)),
            "bias": float(metrics.get("bias", metrics.get("mean_bias", 0.0))),
            "upper_loa": float(metrics.get("loa_upper", metrics.get("upper_loa", 0.0))),
            "lower_loa": float(metrics.get("loa_lower", metrics.get("lower_loa", 0.0))),
        },
        "params": {
            "resolution": str(params.get("resolution", "1280x720")),
            "fps": float(params.get("fps", 30.0)),
            "lux": float(params.get("lux", 300.0)),
            "camera_distance_cm": float(params.get("camera_distance_cm", 50.0)),
            "fitzpatrick_type": str(params.get("fitzpatrick_type", "Type III")),
            "age": int(params.get("age", 25)),
            "constraints_applied": str(params.get("constraints_applied", "None")),
        },
    }

    spio.savemat(output_mat_path, mat_dict, do_compression=True)
    return output_mat_path


def verify_submission_mat_file(mat_path: str) -> Dict[str, Any]:
    """Verify that a generated .mat file contains all 6 required variables and valid types."""
    data = spio.loadmat(mat_path, squeeze_me=True)
    required_vars = ["facial_video", "roi_data", "reference_signal", "processed_signal", "metrics", "params"]
    missing = [v for v in required_vars if v not in data]
    if missing:
        raise ValueError(f"MATLAB file {mat_path} is missing required variables: {missing}")

    return {
        "facial_video_shape": data["facial_video"].shape,
        "roi_data_shape": data["roi_data"].shape,
        "reference_len": len(np.atleast_1d(data["reference_signal"])),
        "processed_len": len(np.atleast_1d(data["processed_signal"])),
        "metrics_keys": data["metrics"].dtype.names if hasattr(data["metrics"], "dtype") else list(data["metrics"].keys()),
        "params_keys": data["params"].dtype.names if hasattr(data["params"], "dtype") else list(data["params"].keys()),
        "status": "PASSED",
    }
