"""Constraint C6: Aging-Skin Micro-Texture Perturbation Simulator.

Complies with 2027 IEEE Signal Processing Cup specifications (Section 2.2, C6):
'Apply localized micro-texture/shadow perturbation via imfilter with edge-emphasizing
kernels to approximate wrinkle-induced reflectance variation.'
"""

from typing import Optional
import cv2
import numpy as np


def apply_aging_skin_c6(
    frame_bgr: np.ndarray,
    wrinkle_strength: float = 0.35,
    roi_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply aging skin micro-texture and shadow perturbation (IEEE SP Cup C6).

    Simulates epidermis elasticity loss and micro-shadowing from wrinkles using
    directional edge-emphasizing filtering and Difference-of-Gaussians (DoG).

    Args:
        frame_bgr: Input BGR image (H, W, 3) in uint8.
        wrinkle_strength: Perturbation intensity scaling (default 0.35).
        roi_mask: Optional binary mask (H, W) indicating facial skin region.

    Returns:
        aged_frame: Image with localized micro-wrinkle shadows.
    """
    frame_f = frame_bgr.astype(np.float32)

    # 1. Directional edge-emphasizing kernel (horizontal/sub-horizontal forehead creases)
    k_dir = np.array([
        [-1.0, -2.0, -1.0],
        [ 2.0,  4.0,  2.0],
        [-1.0, -2.0, -1.0],
    ], dtype=np.float32) / 4.0

    dir_edges = cv2.filter2D(frame_f, -1, k_dir)

    # 2. Difference-of-Gaussians (DoG) micro-relief texture
    g1 = cv2.GaussianBlur(frame_f, (3, 3), 1.0)
    g2 = cv2.GaussianBlur(frame_f, (7, 7), 2.5)
    dog = g1 - g2

    # Combined micro-shadow perturbation
    perturbation = (0.6 * dir_edges + 0.4 * dog) * float(wrinkle_strength)

    # Apply perturbation (localized to skin region if mask is provided)
    if roi_mask is not None:
        mask_norm = (roi_mask.astype(np.float32) / 255.0)[..., None]
        aged_f = frame_f - perturbation * mask_norm
    else:
        # Default: apply predominantly to upper 75% of image (forehead & cheeks)
        h, w = frame_bgr.shape[:2]
        spatial_weight = np.ones((h, w, 1), dtype=np.float32)
        spatial_weight[int(0.85 * h):] = 0.2
        aged_f = frame_f - perturbation * spatial_weight

    return np.clip(aged_f, 0.0, 255.0).astype(np.uint8)


def simulate_aging_skin_video(
    frames: np.ndarray,
    wrinkle_strength: float = 0.35,
    roi_masks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply aging skin micro-texture stress test to an entire video stack.

    Args:
        frames: Array of frames (T, H, W, 3) in uint8.
        wrinkle_strength: Strength of wrinkle perturbation.
        roi_masks: Optional array of masks (T, H, W).

    Returns:
        aged_frames: Perturbed video frames (T, H, W, 3).
    """
    t_len = len(frames)
    aged = np.empty_like(frames)
    for i in range(t_len):
        mask = roi_masks[i] if roi_masks is not None else None
        aged[i] = apply_aging_skin_c6(frames[i], wrinkle_strength=wrinkle_strength, roi_mask=mask)
    return aged
