"""Facial occlusion / surgical mask simulator for Constraint C7 (§2.2 C7).

Simulates realistic lower-face surgical mask occlusion covering >= 40% of the face
(nose bridge, cheeks, mouth, and chin), leaving only the forehead and periorbital region visible.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def apply_mask_occlusion_c7(
    frame_bgr: np.ndarray,
    lower_face_pts: Optional[np.ndarray] = None,
    mask_color: Tuple[int, int, int] = (195, 175, 95),  # surgical cyan (BGR)
) -> np.ndarray:
    """Overlay a synthetic surgical mask covering >= 40% of the lower face.

    Args:
        frame_bgr: Image array in BGR format of shape (H, W, 3).
        lower_face_pts: Optional (N, 2) facial landmark points covering mid-nose down to chin.
        mask_color: BGR color of the surgical mask.

    Returns:
        masked_frame: Frame with lower face occluded.
    """
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    if lower_face_pts is not None and len(lower_face_pts) >= 4:
        hull = cv2.convexHull(lower_face_pts.astype(np.int32))
        cv2.fillConvexPoly(out, hull, mask_color)
        # Add subtle shading/border to emulate cloth edge
        cv2.polylines(out, [hull], isClosed=True, color=(160, 140, 70), thickness=2)
    else:
        # Landmark-free geometric fallback: lower 45% of face oval
        cx, cy = w // 2, int(h * 0.5)
        rx, ry = int(w * 0.25), int(h * 0.35)

        # Polygon from mid-nose level down to chin boundary
        poly_pts = np.array([
            [cx - int(rx * 0.95), int(cy + ry * 0.05)],
            [cx + int(rx * 0.95), int(cy + ry * 0.05)],
            [cx + int(rx * 0.70), int(cy + ry * 0.95)],
            [cx - int(rx * 0.70), int(cy + ry * 0.95)],
        ], dtype=np.int32)

        cv2.fillPoly(out, [poly_pts], mask_color)
        cv2.polylines(out, [poly_pts], isClosed=True, color=(160, 140, 70), thickness=2)

    return out


def simulate_masked_video(
    frames: np.ndarray,
    tracker: Optional[any] = None,
) -> np.ndarray:
    """Apply synthetic mask occlusion to an entire video sequence."""
    if tracker is not None and hasattr(tracker, "reset"):
        tracker.reset()

    t_len = len(frames)
    masked_frames = np.empty_like(frames)

    for i in range(t_len):
        pts = None
        if tracker is not None:
            res = tracker.process_frame(frames[i])
            pts = res.get("lower_face_pts")
        masked_frames[i] = apply_mask_occlusion_c7(frames[i], lower_face_pts=pts)

    return masked_frames
