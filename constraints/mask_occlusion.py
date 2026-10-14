"""Constraint C7: Masked-Face Operation Simulation (§2.2 C7).

Implements the MathWorks competition specifications verbatim:
- Overlays a synthetic occlusion patch on the lower face region of a baseline video
  covering >= 40% of the lower face (e.g. surgical mask), aligned to facial landmarks.
- Enables downstream pipelines to dynamically re-weight towards forehead/periocular ROIs.
"""

from typing import Tuple, List, Optional
import cv2
import numpy as np


def apply_mask_occlusion_c7(
    frame_bgr: np.ndarray,
    lower_face_pts: Optional[np.ndarray] = None,
    mask_color: Tuple[int, int, int] = (195, 175, 95),  # surgical cyan (BGR)
) -> np.ndarray:
    """Overlay a synthetic surgical mask covering >= 40% of the lower face (cheeks, nose, mouth, chin).

    Args:
        frame_bgr: Image array in BGR format of shape (H, W, 3).
        lower_face_pts: Optional (N, 2) facial landmark points covering mid-nose and cheeks down to chin.
        mask_color: BGR color of the surgical mask.

    Returns:
        masked_frame: Image with realistic surgical mask polygon applied.
    """
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    if lower_face_pts is not None and len(lower_face_pts) >= 4:
        hull = cv2.convexHull(lower_face_pts.astype(np.int32))
        cv2.fillConvexPoly(out, hull, mask_color)
        cv2.polylines(out, [hull], isClosed=True, color=(160, 140, 70), thickness=2)
    else:
        # Landmark-free geometric fallback: lower 50% of face oval (mid-nose & cheeks to chin)
        cx, cy = w // 2, int(h * 0.5)
        rx, ry = int(w * 0.25), int(h * 0.35)

        poly_pts = np.array([
            [cx - int(rx * 0.96), int(cy - ry * 0.08)],
            [cx,                  int(cy - ry * 0.16)],
            [cx + int(rx * 0.96), int(cy - ry * 0.08)],
            [cx + int(rx * 0.80), int(cy + ry * 0.95)],
            [cx - int(rx * 0.80), int(cy + ry * 0.95)],
        ], dtype=np.int32)

        cv2.fillPoly(out, [poly_pts], mask_color)
        cv2.polylines(out, [poly_pts], isClosed=True, color=(160, 140, 70), thickness=2)

    return out


def simulate_masked_video(
    frames: np.ndarray,
    tracker: Optional[any] = None,
) -> np.ndarray:
    """Apply synthetic mask occlusion to an entire video sequence with stabilized mask anchoring."""
    t_len = len(frames)
    masked_frames = np.empty_like(frames)
    anchor_pts = None

    for i in range(t_len):
        pts = None
        if tracker is not None:
            res = tracker.process_frame(frames[i])
            cand_pts = res.get("lower_face_pts")
            if cand_pts is not None and len(cand_pts) >= 4:
                if anchor_pts is None or len(anchor_pts) != len(cand_pts):
                    anchor_pts = cand_pts.astype(np.float64)
                else:
                    shift = float(np.linalg.norm(np.mean(cand_pts, axis=0) - np.mean(anchor_pts, axis=0)))
                    face_w = max(1.0, float(np.max(cand_pts[:, 0]) - np.min(cand_pts[:, 0])))
                    if shift > 0.20 * face_w:
                        anchor_pts = 0.85 * anchor_pts + 0.15 * cand_pts.astype(np.float64)
                pts = np.round(anchor_pts).astype(np.int32)
        masked_frames[i] = apply_mask_occlusion_c7(frames[i], lower_face_pts=pts)

    return masked_frames

