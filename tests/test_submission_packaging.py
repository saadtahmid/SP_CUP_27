"""Integration and Verification Tests for Official Submission Packaging (§1.2 & §4)."""

import os
import shutil
import pytest
import numpy as np
from scipy import io as spio

from pipeline_packaging.mat_packager import create_submission_mat_file, verify_submission_mat_file
from pipeline_packaging.exporter import export_submission_bundle


def test_submission_mat_structure(tmp_path):
    mat_path = str(tmp_path / "Round1_Baseline_None.mat")

    facial_video = np.zeros((30, 100, 100, 3), dtype=np.uint8)
    roi_data = np.ones((30, 100, 100), dtype=np.uint8)
    ref_signal = np.full(30, 75.0)
    proc_signal = np.full(30, 74.8)
    metrics = {"mae": 0.2, "rmse": 0.2, "pearson_r": 0.99, "bias": -0.2, "loa_upper": 0.1, "loa_lower": -0.5}
    params = {
        "resolution": "1280x720",
        "fps": 30.0,
        "lux": 300.0,
        "camera_distance_cm": 50.0,
        "fitzpatrick_type": "Type III",
        "age": 25,
        "constraints_applied": "None",
    }

    create_submission_mat_file(
        output_mat_path=mat_path,
        facial_video=facial_video,
        roi_data=roi_data,
        reference_signal=ref_signal,
        processed_signal=proc_signal,
        metrics=metrics,
        params=params,
    )

    assert os.path.exists(mat_path)
    res = verify_submission_mat_file(mat_path)
    assert res["status"] == "PASSED"
    assert res["facial_video_shape"] == (30, 100, 100, 3)
    assert res["reference_len"] == 30


def test_submission_bundle_export(tmp_path):
    sub_dir = str(tmp_path / "Round1_Baseline")
    dummy_vid = str(tmp_path / "dummy.mp4")
    dummy_csv = str(tmp_path / "dummy_gt.csv")

    # Create dummy video and csv
    with open(dummy_vid, "wb") as f:
        f.write(b"dummy video data")
    with open(dummy_csv, "w") as f:
        f.write("timestamp_sec,hr_bpm\n0.0,75.0\n")

    frames = np.zeros((10, 50, 50, 3), dtype=np.uint8)
    masks = np.ones((10, 50, 50), dtype=np.uint8)
    ref = np.full(10, 75.0)
    proc = np.full(10, 75.0)
    metrics = {"mae": 0.0, "rmse": 0.0, "pearson_r": 1.0}
    params = {"resolution": "640x480", "fps": 30.0, "lux": 300.0, "fitzpatrick_type": "Type III"}

    export_submission_bundle(
        output_dir=sub_dir,
        video_path=dummy_vid,
        gt_csv_path=dummy_csv,
        facial_frames=frames,
        roi_masks=masks,
        reference_hr_series=ref,
        estimated_hr_series=proc,
        metrics=metrics,
        params=params,
    )

    # Verify all 5 files exist
    assert os.path.exists(os.path.join(sub_dir, "Round1_Baseline_None.mat"))
    assert os.path.exists(os.path.join(sub_dir, "process_round1.m"))
    assert os.path.exists(os.path.join(sub_dir, "target_video.mp4"))
    assert os.path.exists(os.path.join(sub_dir, "reference_signal.csv"))
    assert os.path.exists(os.path.join(sub_dir, "processed_output.csv"))
