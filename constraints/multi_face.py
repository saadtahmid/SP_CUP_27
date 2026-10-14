"""Constraint C3: Multi-Person Frame Isolation & Kalman Target Face Tracker.

Complies with 2027 IEEE Signal Processing Cup specifications (Section 2.2, C3):
'Isolate primary subject using vision.KalmanFilter + assignDetectionsToTracks to reject
bystander faces pulsing at distinct heart rates.'
"""

from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


class TargetFaceKalmanTracker:
    """Tracks primary subject across video frames rejecting bystander distractor faces."""

    def __init__(self, dt: float = 1.0 / 30.0):
        self.dt = dt
        # State: [cx, cy, w, h, vx, vy]^T
        self.state = None
        self.covariance = np.eye(6, dtype=np.float64) * 100.0

        # State transition matrix F
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 4] = dt
        self.F[1, 5] = dt

        # Measurement matrix H: measures [cx, cy, w, h]
        self.H = np.zeros((4, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Covariance noise matrices
        self.Q = np.eye(6, dtype=np.float64) * 1.0
        self.R = np.eye(4, dtype=np.float64) * 5.0

        self.initialized = False
        self.primary_target_box = None

    def initialize_primary_target(self, candidate_boxes: List[np.ndarray], frame_shape: Tuple[int, int]):
        """Select largest central face in frame 0 as primary subject target."""
        if not candidate_boxes:
            return

        h, w = frame_shape[:2]
        center_x, center_y = w / 2.0, h / 2.0

        best_score = -1e9
        best_box = candidate_boxes[0]

        for box in candidate_boxes:
            xmin, ymin, xmax, ymax = box
            area = (xmax - xmin) * (ymax - ymin)
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            dist_to_center = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)

            score = area - 50.0 * dist_to_center
            if score > best_score:
                best_score = score
                best_box = box

        xmin, ymin, xmax, ymax = best_box
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        bw = float(xmax - xmin)
        bh = float(ymax - ymin)

        self.state = np.array([cx, cy, bw, bh, 0.0, 0.0], dtype=np.float64)
        self.primary_target_box = np.array(best_box, dtype=np.int32)
        self.initialized = True

    def predict(self) -> np.ndarray:
        """Kalman state prediction step."""
        if not self.initialized:
            return None
        self.state = self.F @ self.state
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        cx, cy, bw, bh = self.state[:4]
        return np.array([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], dtype=np.float64)

    def update(self, detected_box: np.ndarray):
        """Kalman measurement update step."""
        xmin, ymin, xmax, ymax = detected_box
        z = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, xmax - xmin, ymax - ymin], dtype=np.float64)

        y = z - self.H @ self.state
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.covariance = (np.eye(6) - K @ self.H) @ self.covariance

        cx, cy, bw, bh = self.state[:4]
        self.primary_target_box = np.array([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], dtype=np.int32)

    def track(self, candidate_boxes: List[np.ndarray], frame_shape: Tuple[int, int]) -> np.ndarray:
        """Associate detections to target track and return primary subject box."""
        if not candidate_boxes:
            if self.initialized:
                pred_box = self.predict()
                return np.round(pred_box).astype(np.int32)
            return np.array([0, 0, frame_shape[1], frame_shape[0]], dtype=np.int32)

        if not self.initialized:
            self.initialize_primary_target(candidate_boxes, frame_shape)
            return self.primary_target_box

        pred_box = self.predict()

        best_iou = -1.0
        best_match = None

        for box in candidate_boxes:
            iou = compute_iou(pred_box, box)
            if iou > best_iou:
                best_iou = iou
                best_match = box

        if best_match is not None and best_iou > 0.15:
            self.update(best_match)
        else:
            cx, cy, bw, bh = self.state[:4]
            self.primary_target_box = np.array([cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0], dtype=np.int32)

        return self.primary_target_box


def simulate_multi_person_video(
    target_frames: np.ndarray,
    distractor_hr_bpm: float = 115.0,
    distractor_scale: float = 0.35,
    fs: float = 30.0,
) -> np.ndarray:
    """Embed a secondary bystander face into the frame pulsing at a conflicting heart rate (C3)."""
    t_len, h, w, _ = target_frames.shape
    composite = np.copy(target_frames)

    f_dist = distractor_hr_bpm / 60.0
    dh = int(h * distractor_scale)
    dw = int(dh * 0.85)

    # Position in upper right corner (x >= 0.72 * w)
    x1 = w - dw - int(0.05 * w)
    y1 = int(0.08 * h)
    x2 = x1 + dw
    y2 = y1 + dh

    for t in range(t_len):
        pulse = np.sin(2.0 * np.pi * f_dist * (t / fs))
        bystander = np.zeros((dh, dw, 3), dtype=np.uint8)

        r_val = 195 + int(3.0 * np.sin(2.0 * np.pi * 0.2 * t))
        g_val = int(np.clip(150.0 - 15.0 * pulse, 0, 255))
        b_val = 125

        cv2.ellipse(bystander, (dw // 2, dh // 2), (dw // 2 - 4, dh // 2 - 4), 0, 0, 360, (b_val, g_val, r_val), -1)
        cv2.circle(bystander, (dw // 3, dh // 3), 4, (50, 40, 30), -1)
        cv2.circle(bystander, (2 * dw // 3, dh // 3), 4, (50, 40, 30), -1)
        cv2.line(bystander, (dw // 3, 2 * dh // 3), (2 * dw // 3, 2 * dh // 3), (70, 50, 60), 2)

        mask = (bystander.sum(axis=-1) > 0)[..., None]
        composite[t, y1:y2, x1:x2] = np.where(mask, bystander, composite[t, y1:y2, x1:x2])

    return composite

