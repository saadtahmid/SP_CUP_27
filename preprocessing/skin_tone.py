"""Skin tone classification and Individual Typology Angle (ITA) computation.

Complies with 2027 IEEE Signal Processing Cup specifications (Section 2.2, C2).
Converts facial ROI pixels from RGB to CIE 1976 L*a*b* space, computes ITA:
    ITA = arctan((L* - 50) / b*) * (180 / pi)
and stratifies skin tone across Fitzpatrick Skin Types I-VI.
"""

from typing import Dict, Tuple, Union, Optional
import numpy as np


# Standard ITA-to-Fitzpatrick classification thresholds
FITZPATRICK_THRESHOLDS = {
    "Type I": (55.0, float("inf")),       # Very Light (> 55°)
    "Type II": (41.0, 55.0),              # Light (41° to 55°)
    "Type III": (28.0, 41.0),             # Intermediate (28° to 41°)
    "Type IV": (10.0, 28.0),              # Tan (10° to 28°)
    "Type V": (-30.0, 10.0),              # Brown (-30° to 10°)
    "Type VI": (float("-inf"), -30.0),    # Dark (<= -30°)
}


def rgb_to_lab_approx(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image/pixels to CIE L*a*b* color space.

    Args:
        rgb: Float numpy array of shape (..., 3) normalized to [0, 1] or uint8 [0, 255].

    Returns:
        lab: Float numpy array of shape (..., 3) with L in [0, 100], a in [-128, 127], b in [-128, 127].
    """
    arr = rgb.astype(np.float64)
    if arr.max() > 1.0:
        arr /= 255.0

    # sRGB gamma correction to linear RGB
    mask = arr > 0.04045
    linear = np.where(mask, np.power((arr + 0.055) / 1.055, 2.4), arr / 12.92)

    # Convert linear RGB to CIE XYZ (standard D65 illuminant matrix)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float64)

    xyz = np.tensordot(linear, M.T, axes=(-1, -1))

    # Reference white point D65
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    xyz[..., 0] /= Xn
    xyz[..., 1] /= Yn
    xyz[..., 2] /= Zn

    # Non-linear transformation to L*a*b*
    epsilon = 0.008856
    kappa = 903.3

    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1)


def compute_ita(
    roi_pixels: np.ndarray,
    reference_white: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Compute Individual Typology Angle (ITA) from RGB facial ROI pixels.

    Args:
        roi_pixels: RGB image or 2D array of skin pixels (H, W, 3) or (N, 3).
        reference_white: Optional reference white RGB for illumination normalization.

    Returns:
        ita: Individual Typology Angle in degrees.
        mean_l: Average L* value.
        mean_b: Average b* value.
    """
    pixels = np.asarray(roi_pixels)
    if pixels.ndim > 2:
        pixels = pixels.reshape(-1, 3)

    # Filter out black or pure white padding if masked
    valid_mask = (pixels.sum(axis=-1) > 10) & (pixels.sum(axis=-1) < 750)
    if np.any(valid_mask):
        pixels = pixels[valid_mask]

    # Glare and extreme shadow trimming (10th to 85th luminance percentile)
    if len(pixels) > 20:
        if reference_white is not None:
            ref = np.asarray(reference_white, dtype=np.float64)
            if ref.ndim > 1:
                ref = np.mean(ref.reshape(-1, 3), axis=0)
            scale = np.mean(ref) / (ref + 1e-6)
            pixels = np.clip(pixels.astype(np.float64) * scale, 0.0, 255.0).astype(np.uint8)

        # Compute luminance (Rec. 601)
        lum = 0.299 * pixels[:, 0] + 0.587 * pixels[:, 1] + 0.114 * pixels[:, 2]
        p10 = np.percentile(lum, 10)
        p85 = np.percentile(lum, 85)
        in_range = (lum >= p10) & (lum <= p85)
        if np.sum(in_range) > 5:
            pixels = pixels[in_range]

    lab = rgb_to_lab_approx(pixels)
    mean_l = float(np.mean(lab[:, 0]))
    mean_b = float(np.mean(lab[:, 2]))

    # Prevent division by zero
    b_val = mean_b if abs(mean_b) > 1e-4 else 1e-4

    # Formula: ITA = arctan((L* - 50) / b*) * (180 / pi)
    ita_deg = float(np.arctan2(mean_l - 50.0, b_val) * (180.0 / np.pi))
    return ita_deg, mean_l, mean_b


def classify_fitzpatrick(ita_deg: float) -> str:
    """Classify skin tone into Fitzpatrick Types I-VI based on ITA degree."""
    for skin_type, (low, high) in FITZPATRICK_THRESHOLDS.items():
        if low <= ita_deg < high:
            return skin_type
    return "Type VI" if ita_deg <= -30.0 else "Type I"


def analyze_skin_tone(
    roi_pixels: np.ndarray,
    reference_white: Optional[np.ndarray] = None,
) -> Dict[str, Union[float, str]]:
    """Complete skin tone analysis returning ITA and Fitzpatrick classification."""
    ita, mean_l, mean_b = compute_ita(roi_pixels, reference_white=reference_white)
    fitzpatrick = classify_fitzpatrick(ita)
    return {
        "ita_deg": round(ita, 2),
        "mean_L": round(mean_l, 2),
        "mean_b": round(mean_b, 2),
        "fitzpatrick_type": fitzpatrick,
    }
