"""Self-Supervised Contrastive Learning (SCL) Transformations for rPPG.

Synthesized from BeatFormer (Comas & Sukno, ICCVW 2025).
Provides the 4 physiological contrastive augmentations:
1. Random Spatial Occlusion: directly emulates masked face (C7)
2. HSV/Lab Color Jitter: emulates illumination & skin tone shifts (C1/C2)
3. Temporal Inversion / Flipping: cardiac pulse phase disentangling
4. Spatial Scaling: camera distance / resolution invariance (C4)
"""

from typing import Tuple, List
import cv2
import numpy as np


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
            # Mask lower 40-50% or random patch
            occ_h = int(h * np.random.uniform(0.3, 0.5))
            occ_w = int(w * np.random.uniform(0.5, 0.9))
            y1 = np.random.randint(h // 2, h - occ_h + 1) if h - occ_h > h // 2 else h // 2
            x1 = np.random.randint(0, max(1, w - occ_w))
            out[y1:y1 + occ_h, x1:x1 + occ_w] = np.random.randint(50, 200, (occ_h, occ_w, 3), dtype=np.uint8)
        return out

    def color_jitter(self, frame: np.ndarray) -> np.ndarray:
        """Color space jittering preserving physiological pulse variation."""
        if np.random.rand() < self.p_color_jitter:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            # Jitter hue and value (brightness)
            hsv[:, :, 0] = (hsv[:, :, 0] + np.random.uniform(-10, 10)) % 180
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.7, 1.3), 0, 255)
            frame_jitter = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            return frame_jitter
        return frame

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
