"""Face detection and landmark extraction module using MediaPipe FaceMesh.

Extracts facial landmarks and defines physiological Regions of Interest (ROIs):
- Forehead
- Left cheek
- Right cheek
- Lower face boundary (for C7 mask occlusion detection)
"""

from typing import Dict, List, Tuple, Optional, Union
import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    mp = None
    _HAS_MEDIAPIPE = False


class FaceMeshTracker:
    """MediaPipe FaceMesh facial landmark and ROI extractor with OpenCV fallback."""

    def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = True):
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        if _HAS_MEDIAPIPE and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=max_num_faces,
                refine_landmarks=refine_landmarks,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._cascade = None
        else:
            self.face_mesh = None
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)

        # Standard MediaPipe landmark indices for physiological ROIs
        self.forehead_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
        ]
        self.forehead_roi_pts = [10, 67, 69, 108, 109, 10, 338, 297, 299, 337]
        self.left_cheek_pts = [117, 118, 100, 126, 209, 49, 129, 203, 205, 50]
        self.right_cheek_pts = [346, 347, 329, 355, 429, 279, 358, 423, 425, 280]
        self.lower_face_pts = [
            164, 167, 165, 92, 186, 57, 43, 106, 182, 83, 18,
            313, 406, 335, 273, 287, 410, 322, 391, 393, 152,
        ]

    def reset(self):
        """No-op reset since static_image_mode=True is stateless across frames."""
        pass

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        return_all_faces: bool = False,
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """Process a single frame and extract landmark coordinates."""
        h, w = frame_bgr.shape[:2]

        def _make_box_dict(xmin: int, ymin: int, xmax: int, ymax: int):
            bw = max(1, xmax - xmin)
            bh = max(1, ymax - ymin)
            fh_pts = np.array([
                [xmin + int(0.25 * bw), ymin + int(0.12 * bh)],
                [xmin + int(0.75 * bw), ymin + int(0.12 * bh)],
                [xmin + int(0.75 * bw), ymin + int(0.28 * bh)],
                [xmin + int(0.25 * bw), ymin + int(0.28 * bh)],
            ], dtype=np.int32)
            lc_pts = np.array([
                [xmin + int(0.18 * bw), ymin + int(0.50 * bh)],
                [xmin + int(0.42 * bw), ymin + int(0.50 * bh)],
                [xmin + int(0.42 * bw), ymin + int(0.75 * bh)],
                [xmin + int(0.18 * bw), ymin + int(0.75 * bh)],
            ], dtype=np.int32)
            rc_pts = np.array([
                [xmin + int(0.58 * bw), ymin + int(0.50 * bh)],
                [xmin + int(0.82 * bw), ymin + int(0.50 * bh)],
                [xmin + int(0.82 * bw), ymin + int(0.75 * bh)],
                [xmin + int(0.58 * bw), ymin + int(0.75 * bh)],
            ], dtype=np.int32)
            lower_pts = np.array([
                [xmin + int(0.18 * bw), ymin + int(0.58 * bh)],
                [xmin + int(0.50 * bw), ymin + int(0.54 * bh)],
                [xmin + int(0.82 * bw), ymin + int(0.58 * bh)],
                [xmin + int(0.78 * bw), ymin + int(0.92 * bh)],
                [xmin + int(0.50 * bw), ymin + int(0.98 * bh)],
                [xmin + int(0.22 * bw), ymin + int(0.92 * bh)],
            ], dtype=np.int32)
            return {
                "bbox": np.array([xmin, ymin, xmax, ymax], dtype=np.int32),
                "forehead_pts": fh_pts,
                "left_cheek_pts": lc_pts,
                "right_cheek_pts": rc_pts,
                "lower_face_pts": lower_pts,
                "landmarks": None,
            }

        def _fallback():
            if self._cascade is not None:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                det = self._cascade.detectMultiScale(gray, 1.1, 4)
                if len(det) > 0:
                    boxes = []
                    for (x, y, bw_d, bh_d) in det[:self.max_num_faces]:
                        boxes.append(_make_box_dict(int(x), int(y), int(x + bw_d), int(y + bh_d)))
                    return boxes if return_all_faces else boxes[0]
            cx, cy = w // 2, int(h * 0.5)
            rx, ry = int(w * 0.25), int(h * 0.35)
            xmin, xmax = max(0, cx - rx), min(w, cx + rx)
            ymin, ymax = max(0, cy - ry), min(h, cy + ry)
            fb = _make_box_dict(xmin, ymin, xmax, ymax)
            return [fb] if return_all_faces else fb

        if self.face_mesh is None:
            return _fallback()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return _fallback()

        extracted_faces = []
        for face_landmarks in results.multi_face_landmarks:
            coords = np.array([(int(pt.x * w), int(pt.y * h)) for pt in face_landmarks.landmark], dtype=np.int32)
            xmin = max(0, int(np.min(coords[:, 0])))
            xmax = min(w, int(np.max(coords[:, 0])))
            ymin = max(0, int(np.min(coords[:, 1])))
            ymax = min(h, int(np.max(coords[:, 1])))

            forehead_pts = coords[self.forehead_roi_pts]
            left_cheek = coords[self.left_cheek_pts]
            right_cheek = coords[self.right_cheek_pts]
            lower_face = coords[self.lower_face_pts]

            extracted_faces.append({
                "landmarks": coords,
                "bbox": np.array([xmin, ymin, xmax, ymax], dtype=np.int32),
                "forehead_pts": forehead_pts,
                "left_cheek_pts": left_cheek,
                "right_cheek_pts": right_cheek,
                "lower_face_pts": lower_face,
            })

        if return_all_faces:
            return extracted_faces
        return extracted_faces[0]

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()
