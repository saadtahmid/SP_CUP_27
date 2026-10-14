"""Constraint C1: Low-Light Operation Simulation (§2.2 C1).

Implements the MathWorks competition specifications verbatim:
1. Brightness attenuation: Scale pixel intensities using im2double * gain to emulate lux <= 50.
2. Sensor noise injection: Poisson photon shot noise dominant at low illuminance.
3. Combined degradation: Brightness attenuation + Gaussian read noise.
"""

from typing import Tuple, Union
import numpy as np


def apply_low_light_c1(
    frame: np.ndarray,
    target_lux: float = 30.0,
    baseline_lux: float = 300.0,
    noise_mode: str = "combined",  # 'attenuation', 'poisson', 'combined'
    gaussian_var: float = 1e-5,
) -> np.ndarray:
    """Degrade a well-lit video frame to low-light conditions (<= 50 lux).

    Args:
        frame: RGB image (uint8 [0, 255] or float [0, 1]).
        target_lux: Simulated ambient illuminance in lux (<= 50.0).
        baseline_lux: Original illuminance of baseline recording (nominally 300.0 lux).
        noise_mode: 'attenuation', 'poisson', or 'combined' (MathWorks specs).
        gaussian_var: Variance for Gaussian read noise.

    Returns:
        degraded: Degraded frame in uint8 format [0, 255].
    """
    img = frame.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0  # equivalent to MATLAB im2double

    # 1. Multiplicative gain factor for brightness attenuation
    gain = max(0.01, min(1.0, target_lux / baseline_lux))
    attenuated = img * gain

    if noise_mode == "attenuation":
        degraded = attenuated
    elif noise_mode == "poisson":
        # Poisson photon shot noise: scaled by photon count
        peak_photons = 255.0  # attenuated is already scaled by gain = target_lux / baseline_lux
        noisy = np.random.poisson(np.clip(attenuated * peak_photons, 0, None)) / peak_photons
        degraded = np.clip(noisy, 0.0, 1.0)
    elif noise_mode == "combined":
        # Combined Poisson shot noise + Gaussian read/thermal noise
        peak_photons = 255.0  # attenuated is already scaled by gain = target_lux / baseline_lux
        noisy_shot = np.random.poisson(np.clip(attenuated * peak_photons, 0, None)) / peak_photons
        # Gaussian read noise: zero mean, specified variance
        sigma = np.sqrt(gaussian_var)
        read_noise = np.random.normal(0.0, sigma, frame.shape)
        degraded = np.clip(noisy_shot + read_noise, 0.0, 1.0)
    else:
        degraded = attenuated

    return np.clip(degraded * 255.0, 0, 255).astype(np.uint8)


def simulate_low_light_video(
    frames: np.ndarray,
    target_lux: float = 30.0,
    baseline_lux: float = 300.0,
    noise_mode: str = "combined",
) -> np.ndarray:
    """Apply C1 low-light degradation to an entire video sequence of shape (T, H, W, 3)."""
    degraded = np.empty_like(frames, dtype=np.uint8)
    for i in range(len(frames)):
        degraded[i] = apply_low_light_c1(
            frames[i],
            target_lux=target_lux,
            baseline_lux=baseline_lux,
            noise_mode=noise_mode,
        )
    return degraded
