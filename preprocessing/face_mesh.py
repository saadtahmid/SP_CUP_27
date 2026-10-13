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
import mediapipe as mp


class FaceMeshTracker:
    """MediaPipe FaceMesh facial landmark and ROI extractor."""

    def __init__(self, max_num_faces: int = 1, refine_landmarks: bool = True):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Standard MediaPipe landmark indices for physiological ROIs
        # Forehead landmark polygon
        self.forehead_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        # Specific key forehead center points
        self.forehead_roi_pts = [10, 67, 69, 108, 109, 10, 338, 297, 299, 337]
        # Left cheek points (viewer's left or subject's right)
        self.left_cheek_pts = [117, 118, 100, 126, 209, 49, 129, 203, 205, 50]
        # Right cheek points (viewer's right or subject's left)
        self.right_cheek_pts = [346, 347, 329, 355, 429, 279, 358, 423, 425, 280]
        # Lower face points below nose tip (for C7 mask occlusion)
        self.lower_face_pts = [164, 167, 165, 92, 186, 57, 43, 106, 182, 83, 18, 313, 406, 335, 273, 287, 410, 322, 391, 393, 152]

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        return_all_faces: bool = False,
    ) -> Union[Dict[str, np.ndarray], List[Dict[str, np.ndarray]]]:
        """Process a single frame and extract landmark coordinates.

        Args:
            frame_bgr: Image in BGR uint8 format.
            return_all_faces: If True, return list of dicts for all detected faces.

        Returns:
            Single face dict (if return_all_faces is False) or list of face dicts.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        def _fallback():
            cx, cy = w // 2, int(h * 0.5)
            rx, ry = int(w * 0.25), int(h * 0.35)
            xmin, xmax = max(0, cx - rx), min(w, cx + rx)
            ymin, ymax = max(0, cy - ry), min(h, cy + ry)
            return {
                "bbox": np.array([xmin, ymin, xmax, ymax], dtype=np.int32),
                "forehead_pts": np.array([[cx - rx // 2, cy - ry // 2], [cx + rx // 2, cy - ry // 2], [cx + rx // 2, cy - ry // 4], [cx - rx // 2, cy - ry // 4]], dtype=np.int32),
                "left_cheek_pts": np.array([[cx - rx // 2, cy], [cx - rx // 4, cy], [cx - rx // 4, cy + ry // 4], [cx - rx // 2, cy + ry // 4]], dtype=np.int32),
                "right_cheek_pts": np.array([[cx + rx // 4, cy], [cx + rx // 2, cy], [cx + rx // 2, cy + ry // 4], [cx + rx // 4, cy + ry // 4]], dtype=np.int32),
                "lower_face_pts": None,
                "landmarks": None,
            }

        if not results.multi_face_landmarks:
            fb = _fallback()
            return [fb] if return_all_faces else fb

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
        self.face_mesh.close()
