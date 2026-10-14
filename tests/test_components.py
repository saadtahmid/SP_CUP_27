"""Automated Unit Tests for rPPG System Components."""

import pytest
import numpy as np

import os
import cv2
import torch

from preprocessing.skin_tone import compute_ita, classify_fitzpatrick, analyze_skin_tone
from estimation.czt_estimator import CZTHRTransformer
from estimation.fft_estimator import estimate_hr_welch
from models.classical.pipeline import ClassicalRPPGPipeline
from models.hybrid.pfd_module import PhysiologicalFrequencyDenoiser, PFDBlock
from models.hybrid.beatformer import BeatFormer, CZTLayer, ICZTLayer, ZOCABlock, ESDBlock
from constraints.low_light import apply_low_light_c1
from constraints.low_res import apply_low_resolution_c4, apply_frame_decimation_c5, simulate_bitrate_compression_frames
from constraints.mask_occlusion import apply_mask_occlusion_c7
from constraints.multi_face import TargetFaceKalmanTracker, compute_iou, simulate_multi_person_video
from constraints.aging_skin import apply_aging_skin_c6, simulate_aging_skin_video
from evaluation.metrics import compute_all_metrics


def test_skin_tone_ita_and_fitzpatrick():
    # Light skin swatch
    light_rgb = np.array([250, 220, 200], dtype=np.uint8)
    ita_light, _, _ = compute_ita(light_rgb.reshape(1, 1, 3))
    type_light = classify_fitzpatrick(ita_light)
    assert type_light in ["Type I", "Type II"], f"Expected light type, got {type_light}"

    # Dark skin swatch
    dark_rgb = np.array([60, 40, 25], dtype=np.uint8)
    ita_dark, _, _ = compute_ita(dark_rgb.reshape(1, 1, 3))
    type_dark = classify_fitzpatrick(ita_dark)
    assert type_dark in ["Type V", "Type VI"], f"Expected dark type, got {type_dark}"


def test_czt_frequency_resolution():
    fs = 30.0
    duration = 10.0
    t = np.arange(int(duration * fs)) / fs

    # Synthetic signal at exactly 72.0 bpm (1.20 Hz)
    target_bpm = 72.0
    sig = np.sin(2.0 * np.pi * (target_bpm / 60.0) * t)

    czt_estimator = CZTHRTransformer(fs=fs, low_hz=0.7, high_hz=3.0, num_bins=512)
    est_bpm = czt_estimator.estimate_hr(sig)

    assert abs(est_bpm - target_bpm) < 0.2, f"CZT error too large: {est_bpm} vs {target_bpm}"


def test_classical_pipeline_execution():
    fs = 30.0
    t = np.arange(300) / fs
    target_hr = 75.0
    pulse = np.sin(2.0 * np.pi * (target_hr / 60.0) * t)

    # Synthesize RGB traces: Green modulated by cardiac pulse
    r = 200.0 * np.ones_like(t)
    g = 180.0 * (1.0 - 0.02 * pulse)
    b = 150.0 * np.ones_like(t)
    rgb = np.column_stack([r, g, b])

    for method in ["POS", "CHROM", "GREEN"]:
        pipe = ClassicalRPPGPipeline(method=method, fs=fs, use_czt=True)
        out = pipe.process(rgb)
        assert "hr_bpm" in out
        assert 42.0 <= out["hr_bpm"] <= 180.0
        assert abs(out["hr_bpm"] - target_hr) < 2.0, f"Method {method} error: {out['hr_bpm']}"


def test_pfd_denoiser():
    fs = 30.0
    t = np.arange(300) / fs
    # Clean pulse (1.25 Hz = 75 bpm) + out-of-band high-frequency noise (5 Hz)
    clean = np.sin(2.0 * np.pi * 1.25 * t)
    noise = 0.5 * np.sin(2.0 * np.pi * 5.0 * t)
    sig = clean + noise

    pfd = PhysiologicalFrequencyDenoiser(fs=fs)
    cleaned, freqs, psd = pfd.denoise_spectrum(sig)

    assert len(cleaned) == len(sig)
    # Check that out-of-band energy (> 3.5 Hz) in cleaned spectrum is negligible
    out_band = freqs > 3.5
    if np.any(out_band):
        assert np.max(psd[out_band]) < 1e-4


def test_constraint_simulators():
    frame = np.full((100, 100, 3), 180, dtype=np.uint8)

    # C1: Low light
    dim = apply_low_light_c1(frame, target_lux=30.0, baseline_lux=300.0)
    assert np.mean(dim) < np.mean(frame)

    # C4: Resolution
    vga = apply_low_resolution_c4(frame, target_resolution="VGA")
    assert vga.shape == (480, 640, 3)

    # C5: Frame decimation
    frames = np.zeros((30, 50, 50, 3), dtype=np.uint8)
    dec, eff_fps = apply_frame_decimation_c5(frames, original_fps=30.0, target_fps=10.0)
    assert len(dec) == 10
    assert eff_fps == 10.0

    # C7: Mask occlusion
    masked = apply_mask_occlusion_c7(frame)
    assert masked.shape == frame.shape


def test_beatformer_architecture_and_hr():
    """Verify BeatFormer forward pass, CZT/ICZT reconstruction, and HR estimation."""
    fs = 30.0
    model = BeatFormer(fs=fs, window_len=150, num_bins=128)
    model.eval()

    # Load SCL pretrained checkpoint if available
    ckpt_path = "artifacts/beatformer_scl.pt"
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)

    # Synthetic RGB trace with 75.0 BPM (1.25 Hz) modulation
    t = np.arange(300) / fs
    target_hr = 75.0
    pulse = np.sin(2.0 * np.pi * (target_hr / 60.0) * t)
    r = 200.0 * np.ones_like(t)
    g = 180.0 * (1.0 - 0.02 * pulse)
    b = 150.0 * np.ones_like(t)
    rgb_np = np.column_stack([r, g, b])

    # 1. Heart rate estimation
    est_hr = model.estimate_hr(rgb_np)
    assert 42.0 <= est_hr <= 180.0
    assert abs(est_hr - target_hr) < 8.0, f"BeatFormer estimated HR error: {est_hr} vs {target_hr}"

    # 2. Forward pass output on window
    x_win = torch.from_numpy(rgb_np[:150]).float().unsqueeze(0)
    out = model.forward(x_win)
    assert "bvp_wave" in out
    assert out["bvp_wave"].shape == (1, 150, 1)
    assert not torch.isnan(out["bvp_wave"]).any()

    # 3. Full sequence BVP reconstruction
    hr_bpm, bvp_full, psd = model.estimate_hr_from_trace(rgb_np)
    assert len(bvp_full) == 300
    assert not np.isnan(bvp_full).any()


def test_czt_and_iczt_differentiability():
    """Verify CZTLayer and ICZTLayer gradient flow."""
    czt = CZTLayer(length=150, num_bins=128, fs=30.0)
    iczt = ICZTLayer(length=150, num_bins=128)

    x = torch.randn(2, 150, 3, requires_grad=True)
    f_re, f_im = czt(x)
    assert f_re.shape == (2, 128, 3)
    assert f_im.shape == (2, 128, 3)

    recon = iczt(f_re, f_im)
    assert recon.shape == (2, 150, 3)

    loss = recon.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_pfd_block_gradients():
    """Verify PyTorch PFDBlock complex forward and backward gradient flow."""
    pfd_block = PFDBlock(in_features=3, out_features=3)

    z_re = torch.randn(2, 128, 3, requires_grad=True)
    z_im = torch.randn(2, 128, 3, requires_grad=True)

    out_re, out_im = pfd_block(z_re, z_im)
    assert out_re.shape == (2, 128, 3)
    assert out_im.shape == (2, 128, 3)

    loss = out_re.pow(2).sum() + out_im.pow(2).sum()
    loss.backward()

    assert z_re.grad is not None
    assert pfd_block.W_re.grad is not None


def test_target_face_kalman_tracker_c3():
    """Verify C3 TargetFaceKalmanTracker locks onto primary subject and rejects distractor."""
    tracker = TargetFaceKalmanTracker(dt=1.0 / 30.0)
    frame_shape = (480, 640)

    # Primary face centered: [200, 150, 440, 390] (size 240x240)
    # Distractor face in corner: [500, 20, 600, 120] (size 100x100)
    primary_box = np.array([200, 150, 440, 390])
    distractor_box = np.array([500, 20, 600, 120])

    # Frame 0: initialize
    tracked = tracker.track([primary_box, distractor_box], frame_shape)
    assert compute_iou(tracked, primary_box) > 0.80

    # Frames 1-10: smooth movement of primary face
    for i in range(1, 10):
        moved_primary = primary_box + np.array([i * 2, i * 1, i * 2, i * 1])
        tracked = tracker.track([distractor_box, moved_primary], frame_shape)
        assert compute_iou(tracked, moved_primary) > 0.65, f"Lost track at frame {i}"


def test_aging_skin_c6():
    """Verify C6 aging-skin micro-texture perturbation simulator."""
    frame = np.full((120, 120, 3), 180, dtype=np.uint8)
    cv2.circle(frame, (60, 60), 35, (220, 200, 190), -1)
    aged = apply_aging_skin_c6(frame, wrinkle_strength=0.35)

    assert aged.shape == frame.shape
    assert aged.dtype == np.uint8
    # Wrinkle micro-shadows should alter pixel values
    assert not np.array_equal(aged, frame)

    video = np.full((10, 120, 120, 3), 180, dtype=np.uint8)
    for v in video:
        cv2.circle(v, (60, 60), 35, (220, 200, 190), -1)
    aged_video = simulate_aging_skin_video(video, wrinkle_strength=0.35)
    assert aged_video.shape == video.shape
    assert aged_video.dtype == np.uint8


def test_bitrate_compression_c5():
    """Verify C5 bitrate compression simulation."""
    frames = np.random.randint(50, 220, size=(15, 64, 64, 3), dtype=np.uint8)
    compressed = simulate_bitrate_compression_frames(frames, target_kbps=400)

    assert compressed.shape == frames.shape
    assert compressed.dtype == np.uint8

