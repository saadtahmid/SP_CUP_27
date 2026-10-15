"""Low-light (C1) constraint simulation following 2027 IEEE SP Cup / MathWorks guidelines (§2.2 C1).

Implements physically grounded linear illuminance attenuation and CMOS sensor noise:
1. Linear brightness attenuation: `gain = target_lux / baseline_lux` (at 10 lux vs 300 lux,
   gain = 1/30 = 0.0333, reducing skin pixels from ~180 ADC counts to ~6 ADC counts).
2. Sensor photon shot noise: Poisson process scaled to 8-bit full-well equivalent (`peak_photons = 255.0`).
3. Electronic read noise: Zero-mean Gaussian read floor (`gaussian_var = 1e-4`, sigma ~ 2.5 ADC counts).
4. 8-bit ADC quantization (`uint8`), permanently enforcing the physical sub-LSB quantization limit at 10 lux.
"""

from typing import Tuple, Union, Optional
import numpy as np


def apply_low_light_c1(
    frame: np.ndarray,
    target_lux: float = 30.0,
    baseline_lux: float = 300.0,
    noise_mode: str = "combined",  # 'attenuation', 'poisson', 'combined'
    gaussian_var: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Degrade a well-lit video frame to physical low-light conditions (<= 50 lux).

    Args:
        frame: RGB/BGR image (uint8 [0, 255] or float [0, 1]).
        target_lux: Simulated ambient illuminance in lux (e.g., 50.0, 25.0, 10.0).
        baseline_lux: Original illuminance of baseline recording (nominally 300.0 lux).
        noise_mode: 'attenuation', 'poisson', or 'combined' (MathWorks specs).
        gaussian_var: Variance for Gaussian read noise in normalized [0, 1] units.
        rng: Optional NumPy random Generator for reproducible noise draws.

    Returns:
        degraded: Degraded frame in uint8 format [0, 255].
    """
    if float(target_lux) >= float(baseline_lux):
        if frame.dtype == np.uint8:
            return frame.copy()
        return np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)

    img = frame.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0  # equivalent to MATLAB im2double

    # 1. Strict physical linear illuminance scaling: gain = target_lux / baseline_lux
    # At 10 lux / 300 lux, gain = 0.0333 -> 180 ADC skin pixels drop to ~6.0 ADC counts
    gain = max(0.005, min(1.0, float(target_lux) / float(baseline_lux)))
    attenuated = img * gain

    if rng is None:
        rng = np.random.default_rng(42)

    # 2. 8-bit CMOS photon shot noise + electronic read floor
    peak_photons = 255.0

    if noise_mode == "attenuation":
        degraded = attenuated
    elif noise_mode == "poisson":
        lam = np.clip(attenuated * peak_photons, 0.0, None)
        noisy = rng.poisson(lam).astype(np.float32) / peak_photons
        degraded = np.clip(noisy, 0.0, 1.0)
    elif noise_mode == "combined":
        lam = np.clip(attenuated * peak_photons, 0.0, None)
        noisy_shot = rng.poisson(lam).astype(np.float32) / peak_photons
        sigma = float(np.sqrt(gaussian_var))
        read_noise = rng.normal(0.0, sigma, frame.shape).astype(np.float32)
        degraded = np.clip(noisy_shot + read_noise, 0.0, 1.0)
    else:
        degraded = attenuated

    return np.clip(np.round(degraded * 255.0), 0.0, 255.0).astype(np.uint8)


def simulate_low_light_video(
    frames: np.ndarray,
    target_lux: float = 30.0,
    baseline_lux: float = 300.0,
    noise_mode: str = "combined",
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Apply physical C1 low-light degradation to an entire video sequence of shape (T, H, W, 3)."""
    if float(target_lux) >= float(baseline_lux):
        return frames.copy()

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    degraded = np.empty_like(frames, dtype=np.uint8)
    for i in range(len(frames)):
        degraded[i] = apply_low_light_c1(
            frames[i],
            target_lux=target_lux,
            baseline_lux=baseline_lux,
            noise_mode=noise_mode,
            rng=rng,
        )
    return degraded
