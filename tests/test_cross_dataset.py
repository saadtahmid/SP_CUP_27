"""Tests for Cross-Dataset Loaders and Generalization Harness (PURE and MMPD)."""

import os
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import scipy.io as sio
import pytest

from datasets.pure_loader import PUREDataLoader, PURE_PROTOCOLS
from datasets.mmpd_loader import MMPDDataLoader
from scripts.run_cross_dataset import evaluate_pure, generate_synthetic_pure_session


@pytest.fixture
def mock_pure_dir(tmp_path):
    """Create a mock PURE dataset folder with 2 subjects and 2 protocols."""
    pure_root = tmp_path / "PURE"
    pure_root.mkdir()

    for s_idx in ["01", "02"]:
        for p_code in ["01", "02"]:
            sess_name = f"{s_idx}-{p_code}"
            sess_dir = pure_root / sess_name
            sess_dir.mkdir()

            # Create mock JSON with /Image structure
            json_path = sess_dir / f"{sess_name}.json"
            mock_json_content = {
                "/Image": [
                    {"/Timestamp": 1000000000 + i * 33333333, "/PulseRate": 72.5 + i * 0.05, "/Waveform": np.sin(i * 0.2)}
                    for i in range(30)
                ]
            }
            with open(json_path, "w") as f:
                json.dump(mock_json_content, f)

            # Create mock dummy PNG frames
            img_sub = sess_dir / sess_name
            img_sub.mkdir()
            for i in range(10):
                dummy_img = (np.ones((60, 80, 3)) * 128).astype(np.uint8)
                import cv2
                cv2.imwrite(str(img_sub / f"Image{i:04d}.png"), dummy_img)

    return pure_root


@pytest.fixture
def mock_mmpd_dir(tmp_path):
    """Create a mock MMPD dataset folder with .mat files."""
    mmpd_root = tmp_path / "MMPD"
    mmpd_root.mkdir()

    subj_dir = mmpd_root / "subject1"
    subj_dir.mkdir()

    mat_path = subj_dir / "subject1_mini.mat"
    mock_data = {
        "video": (np.random.randint(100, 200, (15, 60, 80, 3))).astype(np.uint8),
        "d_ppg": np.sin(np.linspace(0, 10, 150)),
        "gt_HR": np.array([72.0]),
        "fs": 30.0,
        "light": np.array(["LED-high"]),
        "motion": np.array(["Stationary"]),
        "skin_color": np.array(["Type III"]),
    }
    sio.savemat(str(mat_path), mock_data)

    return mmpd_root


def test_pure_loader_discovery_and_parsing(mock_pure_dir):
    """Verify that PUREDataLoader correctly identifies sessions and parses JSON."""
    loader = PUREDataLoader(data_root=str(mock_pure_dir))
    sessions = loader.sessions

    assert len(sessions) == 4, f"Expected 4 sessions, found {len(sessions)}"

    s0 = sessions[0]
    assert s0["subject_id"] == "01"
    assert s0["protocol_name"] in ["Steady", "Talking"]

    gt_info = loader.parse_pure_json(s0["json_path"])
    assert "mean_hr_bpm" in gt_info
    assert gt_info["mean_hr_bpm"] > 50.0
    assert len(gt_info["hr_series"]) == 30

    frames, fps = loader.load_session_frames(s0, max_frames=5)
    assert len(frames) == 5
    assert fps == 30.0


def test_mmpd_loader_parsing(mock_mmpd_dir):
    """Verify that MMPDDataLoader correctly parses .mat signals and metadata tags."""
    loader = MMPDDataLoader(data_root=str(mock_mmpd_dir))
    recordings = loader.recordings

    assert len(recordings) == 1

    rec = recordings[0]
    parsed = loader.parse_mmpd_mat(rec["mat_path"])

    assert parsed["gt_hr_bpm"] == pytest.approx(72.0, abs=1.0)
    assert parsed["light"] == "LED-high"
    assert parsed["motion"] == "Stationary"
    assert parsed["skin"] == "Type III"

    frames, fps, gt_hr = loader.load_recording_frames(rec, max_frames=10)
    assert len(frames) == 10
    assert frames.shape[1:] == (60, 80, 3)
    assert gt_hr == pytest.approx(72.0, abs=1.0)


def test_synthetic_pure_session_generator():
    """Verify synthetic PURE session generator produces correct shapes and motion annotations."""
    sess = generate_synthetic_pure_session("01", "04", fps=30.0, duration_sec=1.0)
    assert sess["protocol_name"] == "Fast Translation"
    assert len(sess["frames"]) == 30
    assert sess["frames"].shape[1:] == (480, 640, 3)
    assert sess["gt_hr"] > 40.0


def test_run_cross_dataset_pure_execution(mock_pure_dir, tmp_path):
    """Verify end-to-end execution of PURE benchmark on mock data without overwriting real artifacts."""
    out_dir = str(tmp_path / "mock_pure_artifacts")
    df = evaluate_pure(data_dir=str(mock_pure_dir), output_dir=out_dir, max_subjects=1, max_frames=30)
    assert isinstance(df, object)
    assert len(df) > 0
    assert "hybrid_v4_bpm" in df.columns
    assert "pos_error_bpm" in df.columns
    assert "chrom_error_bpm" in df.columns
    assert (Path(out_dir) / "pure_results.csv").exists()
    assert (Path(out_dir) / "pure_protocol_summary.csv").exists()
