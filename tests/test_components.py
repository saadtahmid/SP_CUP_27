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


def test_pairwise_defection_and_diagnostic_cases():
    """Verify Pairwise CHROM ∧ BeatFormer defection, harmonic exclusion guard, and BVP CZT peak consistency."""
    from unittest.mock import patch
    from models.hybrid.hybrid_system import HybridRPPGSystem

    hyb = HybridRPPGSystem(fs=30.0, enable_pfd=False, enable_czt=True, enable_beatformer=True)
    crops = np.zeros((450, 36, 36, 3), dtype=np.uint8)
    rgb = np.zeros((450, 3), dtype=np.float64)

    freqs = np.linspace(42, 180, 512)
    sigma = 1.5
    t = np.arange(450) / 30.0
    freqs_bf = hyb.beatformer.czt.freq_axis.cpu().numpy() * 60.0

    # -------------------------------------------------------------------------
    # Case 1: subject32 @ 300 lx signature (POS subharmonic veto overcome)
    # -------------------------------------------------------------------------
    bvp_bf_1 = np.sin(2 * np.pi * (88.72 / 60.0) * t)
    bvp_me_1 = np.sin(2 * np.pi * (75.08 / 60.0) * t)
    psd_bf_1 = np.exp(-0.5 * ((freqs_bf - 88.72) / sigma)**2)

    with patch.object(hyb.backbone, 'process_frames', return_value={
        'bvp': bvp_me_1, 'hr_bpm': 75.08, 'hr_series': np.full(450, 75.08)
    }), patch.object(hyb.czt, 'estimate_hr', side_effect=[
        (75.08, freqs, np.exp(-0.5 * ((freqs - 75.08) / sigma)**2)),  # raw ME
        (89.87, 0.0, np.exp(-0.5 * ((freqs - 89.87) / sigma)**2)),    # CHROM
        (44.41, 0.0, np.exp(-0.5 * ((freqs - 44.41) / sigma)**2)),    # POS
    ]), patch.object(hyb.beatformer, 'estimate_hr_from_trace', return_value=(
        88.72, bvp_bf_1, psd_bf_1
    )):
        res1 = hyb.process(crops, rgb_signals=rgb)
        log1 = res1['decision_log']
        assert log1['defection_fired'] is True
        assert log1['pairwise_agree'] is True
        assert log1['anchor_isolated'] is True
        assert res1['hr_bpm'] == pytest.approx(88.10, abs=0.5)

        # Edit 2: Shipped BVP CZT peak must agree with final_hr
        czt = CZTHRTransformer(fs=30.0, low_hz=0.7, high_hz=3.0, num_bins=512)
        bvp_peak1 = czt.estimate_hr(res1['processed_signal'])
        assert abs(bvp_peak1 - res1['hr_bpm']) <= 1.5, f"BVP peak ({bvp_peak1}) disagrees with final HR ({res1['hr_bpm']})"

    # -------------------------------------------------------------------------
    # Case 2: subject3 @ 300 lx signature (Mechanism check on dual-peak ambiguity)
    # -------------------------------------------------------------------------
    bvp_bf_2 = np.sin(2 * np.pi * (101.68 / 60.0) * t)
    bvp_me_2 = np.sin(2 * np.pi * (82.00 / 60.0) * t)
    psd_bf_2 = np.exp(-0.5 * ((freqs_bf - 101.68) / sigma)**2)

    with patch.object(hyb.backbone, 'process_frames', return_value={
        'bvp': bvp_me_2, 'hr_bpm': 82.00, 'hr_series': np.full(450, 82.00)
    }), patch.object(hyb.czt, 'estimate_hr', side_effect=[
        (82.00, freqs, np.exp(-0.5 * ((freqs - 82.00) / sigma)**2)),
        (105.20, 0.0, np.exp(-0.5 * ((freqs - 105.20) / sigma)**2)),
        (104.84, 0.0, np.exp(-0.5 * ((freqs - 104.84) / sigma)**2)),
    ]), patch.object(hyb.beatformer, 'estimate_hr_from_trace', return_value=(
        101.68, bvp_bf_2, psd_bf_2
    )):
        res2 = hyb.process(crops, rgb_signals=rgb)
        assert res2['decision_log']['defection_fired'] is True
        assert res2['hr_bpm'] == pytest.approx(102.00, abs=1.0)

    # -------------------------------------------------------------------------
    # Case 3: subject3 @ 50 lx signature (Secondary dispersion, anchor protected)
    # -------------------------------------------------------------------------
    bvp_bf_3 = np.sin(2 * np.pi * (68.20 / 60.0) * t)
    bvp_me_3 = np.sin(2 * np.pi * (98.96 / 60.0) * t)
    psd_bf_3 = np.exp(-0.5 * ((freqs_bf - 68.20) / sigma)**2)

    with patch.object(hyb.backbone, 'process_frames', return_value={
        'bvp': bvp_me_3, 'hr_bpm': 98.96, 'hr_series': np.full(450, 98.96)
    }), patch.object(hyb.czt, 'estimate_hr', side_effect=[
        (98.96, freqs, np.exp(-0.5 * ((freqs - 98.96) / sigma)**2)),
        (74.39, 0.0, np.exp(-0.5 * ((freqs - 74.39) / sigma)**2)),
        (73.68, 0.0, np.exp(-0.5 * ((freqs - 73.68) / sigma)**2)),
    ]), patch.object(hyb.beatformer, 'estimate_hr_from_trace', return_value=(
        68.20, bvp_bf_3, psd_bf_3
    )):
        res3 = hyb.process(crops, rgb_signals=rgb)
        assert res3['decision_log']['defection_fired'] is False
        assert abs(res3['hr_bpm'] - 98.96) < 1.0

    # -------------------------------------------------------------------------
    # Case 4: subject32 @ 50 lx signature (Anchor not far, co-located consensus)
    # -------------------------------------------------------------------------
    bvp_bf_4 = np.sin(2 * np.pi * (75.49 / 60.0) * t)
    bvp_me_4 = np.sin(2 * np.pi * (77.69 / 60.0) * t)
    psd_bf_4 = np.exp(-0.5 * ((freqs_bf - 75.49) / sigma)**2)

    with patch.object(hyb.backbone, 'process_frames', return_value={
        'bvp': bvp_me_4, 'hr_bpm': 77.69, 'hr_series': np.full(450, 77.69)
    }), patch.object(hyb.czt, 'estimate_hr', side_effect=[
        (77.69, freqs, np.exp(-0.5 * ((freqs - 77.69) / sigma)**2)),
        (76.37, 0.0, np.exp(-0.5 * ((freqs - 76.37) / sigma)**2)),
        (77.19, 0.0, np.exp(-0.5 * ((freqs - 77.19) / sigma)**2)),
    ]), patch.object(hyb.beatformer, 'estimate_hr_from_trace', return_value=(
        75.49, bvp_bf_4, psd_bf_4
    )):
        res4 = hyb.process(crops, rgb_signals=rgb)
        assert res4['decision_log']['defection_fired'] is False
        assert abs(res4['hr_bpm'] - 77.69) < 2.0

    # -------------------------------------------------------------------------
    # Case 5: Harmonic exclusion guard (1/2 and 3/2 ratios rejected)
    # -------------------------------------------------------------------------
    for harm_chr, harm_bf in [(40.2, 39.8), (120.1, 119.9)]:
        bvp_bf_h = np.sin(2 * np.pi * (harm_bf / 60.0) * t)
        bvp_me_h = np.sin(2 * np.pi * (80.00 / 60.0) * t)
        psd_bf_h = np.exp(-0.5 * ((freqs_bf - harm_bf) / sigma)**2)

        with patch.object(hyb.backbone, 'process_frames', return_value={
            'bvp': bvp_me_h, 'hr_bpm': 80.00, 'hr_series': np.full(450, 80.00)
        }), patch.object(hyb.czt, 'estimate_hr', side_effect=[
            (80.00, freqs, np.exp(-0.5 * ((freqs - 80.00) / sigma)**2)),
            (harm_chr, 0.0, np.exp(-0.5 * ((freqs - harm_chr) / sigma)**2)),
            (80.00, 0.0, np.exp(-0.5 * ((freqs - 80.00) / sigma)**2)),
        ]), patch.object(hyb.beatformer, 'estimate_hr_from_trace', return_value=(
            harm_bf, bvp_bf_h, psd_bf_h
        )):
            res5 = hyb.process(crops, rgb_signals=rgb)
            assert res5['decision_log']['defection_fired'] is False
            assert "harmonic_of_anchor" in res5['decision_log']['defection_reason']

    # -------------------------------------------------------------------------
    # Case 6: subject15 @ 300 lx signature (POS confirms anchor, defection blocked)
    # -------------------------------------------------------------------------
    bvp_bf_6 = np.sin(2 * np.pi * (72.50 / 60.0) * t)
    bvp_me_6 = np.sin(2 * np.pi * (121.80 / 60.0) * t)
    psd_bf_6 = np.exp(-0.5 * ((freqs_bf - 72.50) / sigma)**2)

    with patch.object(hyb.backbone, 'process_frames', return_value={
        'bvp': bvp_me_6, 'hr_bpm': 121.80, 'hr_series': np.full(450, 121.80)
    }), patch.object(hyb.czt, 'estimate_hr', side_effect=[
        (121.80, freqs, np.exp(-0.5 * ((freqs - 121.80) / sigma)**2)),
        (72.00, 0.0, np.exp(-0.5 * ((freqs - 72.00) / sigma)**2)),
        (120.92, 0.0, np.exp(-0.5 * ((freqs - 120.92) / sigma)**2)),
    ]), patch.object(hyb.beatformer, 'estimate_hr_from_trace', return_value=(
        72.50, bvp_bf_6, psd_bf_6
    )):
        res6 = hyb.process(crops, rgb_signals=rgb)
        assert res6['decision_log']['defection_fired'] is False
        assert res6['decision_log']['pos_confirms_anchor'] is True
        assert "pos_confirms_anchor" in res6['decision_log']['defection_reason']
        assert abs(res6['hr_bpm'] - 121.80) < 1.0



