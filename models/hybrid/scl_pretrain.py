"""Self-Supervised Contrastive Learning (SCL) Transformations and Losses for rPPG.

Synthesized from BeatFormer (Comas & Sukno, ICCVW 2025; Martinez et al., 2025)
and RePSS (Zhang et al., 2024).

Features:
1. SCLAugmentations: Pulse-preserving photometric/spatial transforms (HSV/Lab, occlusion, temporal flipping).
2. RankContrast Temporal Resampling: Frequency-shifting hard negative generation.
3. Earth Mover's Distance (EMD) Loss: Differentiable 1D Wasserstein metric on normalized spectral CDFs.
4. SCL Triplet Hinge Loss: Encourages spectral invariance under noise while repelling resampled negatives.
5. Spectral Sparsity Loss: Encourages cardiac energy concentration in a single dominant peak.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SCLAugmentations:
    """Constraint-aware contrastive data augmentation transforms."""

    def __init__(self, p_occlude: float = 0.5, p_color_jitter: float = 0.5):
        self.p_occlude = p_occlude
        self.p_color_jitter = p_color_jitter

    def random_spatial_occlusion(self, frame: np.ndarray) -> np.ndarray:
        """Apply random rectangular occlusion (simulates partial mask / object occlusion)."""
        h, w = frame.shape[:2]
        out = frame.copy()
        if np.random.rand() < self.p_occlude:
            occ_h = int(h * np.random.uniform(0.3, 0.5))
            occ_w = int(w * np.random.uniform(0.5, 0.9))
            y1 = np.random.randint(h // 2, h - occ_h + 1) if h - occ_h > h // 2 else h // 2
            x1 = np.random.randint(0, max(1, w - occ_w))
            out[y1:y1 + occ_h, x1:x1 + occ_w] = np.random.randint(50, 200, (occ_h, occ_w, 3), dtype=np.uint8)
        return out

    def color_jitter(self, frame: np.ndarray) -> np.ndarray:
        """Color space jittering preserving physiological pulse variation."""
        if np.random.rand() < self.p_color_jitter:
            return self.hsv_transform(frame)
        return frame

    def hsv_transform(self, frame: np.ndarray) -> np.ndarray:
        """Perturb hue and brightness while retaining chrominance dynamics."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.7, 1.3), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def lab_transform(self, frame: np.ndarray) -> np.ndarray:
        """Perturb L* luminance while preserving physiological a* and b* pulse vectors."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * np.random.uniform(0.75, 1.25), 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def temporal_flip(self, video_frames: np.ndarray) -> np.ndarray:
        """Temporal reverse video sequence (preserves cardiac frequency, reverses dicrotic notch)."""
        return video_frames[::-1].copy()

    def transform_sequence(self, video_frames: np.ndarray) -> np.ndarray:
        """Transform entire frame stack."""
        t_len = len(video_frames)
        augmented = np.empty_like(video_frames)
        for i in range(t_len):
            frame = self.random_spatial_occlusion(video_frames[i])
            frame = self.color_jitter(frame)
            augmented[i] = frame
        return augmented


def temporal_resample(rgb_trace: np.ndarray, factor: float) -> np.ndarray:
    """Temporal resampling of RGB trace to alter pulse rate (RePSS RankContrast negative generator).

    Args:
        rgb_trace: (L, 3) numpy array.
        factor: Resampling ratio (e.g. 0.8 to speed up or 1.25 to slow down heart rate).

    Returns:
        resampled_trace: (L, 3) numpy array with modified heart rate.
    """
    L, C = rgb_trace.shape
    orig_t = np.linspace(0.0, 1.0, L)
    # Scaled time coordinates
    scaled_t = np.linspace(0.0, 1.0 / factor, L)
    scaled_t = np.clip(scaled_t, 0.0, 1.0)

    resampled = np.zeros_like(rgb_trace)
    for c in range(C):
        resampled[:, c] = np.interp(scaled_t, orig_t, rgb_trace[:, c])
    return resampled


def emd_loss(psd_p: torch.Tensor, psd_t: torch.Tensor) -> torch.Tensor:
    """Earth Mover's Distance (EMD / 1D Wasserstein) loss between two power spectral densities.

    Implements BeatFormer Eq. 13:
        L_EMD(p, t) = (1 / M) * sum_i (CDF_i(p) - CDF_i(t))^2

    Args:
        psd_p: Power spectrum tensor of shape (B, M) or (M,).
        psd_t: Power spectrum tensor of shape (B, M) or (M,).

    Returns:
        Scalar EMD loss.
    """
    if psd_p.dim() == 1:
        psd_p = psd_p.unsqueeze(0)
    if psd_t.dim() == 1:
        psd_t = psd_t.unsqueeze(0)

    p = psd_p / (torch.sum(psd_p, dim=-1, keepdim=True) + 1e-8)
    t = psd_t / (torch.sum(psd_t, dim=-1, keepdim=True) + 1e-8)

    cdf_p = torch.cumsum(p, dim=-1)
    cdf_t = torch.cumsum(t, dim=-1)

    loss_per_sample = torch.mean((cdf_p - cdf_t) ** 2, dim=-1)
    return torch.mean(loss_per_sample)


def scl_triplet_loss(
    anchor_psd: torch.Tensor,
    pos_psd: torch.Tensor,
    neg_psd: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Spectral Contrastive Learning Triplet Loss (BeatFormer Eqs. 14-16).

    Args:
        anchor_psd: Anchor spectrum (B, M).
        pos_psd: Augmented positive view spectrum (B, M).
        neg_psd: Resampled hard negative spectrum (B, M).
        gamma: Margin parameter.

    Returns:
        Scalar hinge loss: max(0, L_pos - L_neg + gamma).
    """
    l_pos = emd_loss(anchor_psd, pos_psd)
    l_neg = emd_loss(anchor_psd, neg_psd)
    return torch.relu(l_pos - l_neg + gamma)


def spectral_sparsity_loss(psd: torch.Tensor) -> torch.Tensor:
    """Spectral Sparsity regularizer (RePSS / HFUT-VUT, Zhang et al. 2024).

    Penalizes spectral dispersion, encouraging the network to concentrate
    energy around the primary cardiac harmonic.

    Args:
        psd: (B, M) power spectrum tensor.

    Returns:
        Scalar entropy loss.
    """
    if psd.dim() == 1:
        psd = psd.unsqueeze(0)
    prob = psd / (torch.sum(psd, dim=-1, keepdim=True) + 1e-8)
    entropy = -torch.sum(prob * torch.log(prob + 1e-12), dim=-1)
    return torch.mean(entropy)
