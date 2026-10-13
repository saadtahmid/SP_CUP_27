"""Automated Unit Tests for rPPG System Components."""

import pytest
import numpy as np

from preprocessing.skin_tone import compute_ita, classify_fitzpatrick, analyze_skin_tone
from estimation.czt_estimator import CZTHRTransformer
from estimation.fft_estimator import estimate_hr_welch
from models.classical.pipeline import ClassicalRPPGPipeline
from models.hybrid.pfd_module import PhysiologicalFrequencyDenoiser
from constraints.low_light import apply_low_light_c1
from constraints.low_res import apply_low_resolution_c4, apply_frame_decimation_c5
from constraints.mask_occlusion import apply_mask_occlusion_c7
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
