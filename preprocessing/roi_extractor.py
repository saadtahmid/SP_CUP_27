"""ROI Extraction and Spatial Averaging Engine.

Extracts physiological Regions of Interest (forehead, bilateral cheeks) per frame,
computes spatial mean RGB color traces, and formats cropped face inputs for ME-rPPG.
Supports dynamic re-weighting when lower face occlusion (C7 mask) is present.
"""

from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from preprocessing.face_mesh import FaceMeshTracker


class ROIExtractor:
    """Extracts physiological skin regions and spatial RGB color traces."""

    def __init__(self, target_crop_size: Tuple[int, int] = (36, 36)):
        self.tracker = FaceMeshTracker()
        self.crop_size = target_crop_size

    def extract_from_frame(
        self,
        frame_bgr: np.ndarray,
        mask_occlusion_active: bool = False,
    ) -> Dict[str, any]:
        """Extract spatial mean RGB values and cropped face from a single frame."""
        h, w = frame_bgr.shape[:2]
        res = self.tracker.process_frame(frame_bgr)
        bbox = res["bbox"]
        xmin, ymin, xmax, ymax = bbox

        # Cropped face for deep learning (ME-rPPG input: 36x36 normalized float32)
        face_crop = frame_bgr[max(0, ymin):min(h, ymax), max(0, xmin):min(w, xmax)]
        if face_crop.size == 0:
            face_crop = frame_bgr
        face_resized = cv2.resize(face_crop, self.crop_size, interpolation=cv2.INTER_AREA)
        face_rgb_norm = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Construct ROI mask
        roi_mask = np.zeros((h, w), dtype=np.uint8)

        # 1. Forehead ROI
        fh_pts = res.get("forehead_pts")
        if fh_pts is not None and len(fh_pts) >= 3:
            hull_fh = cv2.convexHull(fh_pts)
            cv2.fillConvexPoly(roi_mask, hull_fh, 255)

        # 2. Left & Right Cheeks (if mask occlusion C7 is NOT active, or weighted lower)
        if not mask_occlusion_active:
            l_pts = res.get("left_cheek_pts")
            r_pts = res.get("right_cheek_pts")
            if l_pts is not None and len(l_pts) >= 3:
                cv2.fillConvexPoly(roi_mask, cv2.convexHull(l_pts), 255)
            if r_pts is not None and len(r_pts) >= 3:
                cv2.fillConvexPoly(roi_mask, cv2.convexHull(r_pts), 255)

        # If mask is empty, fallback to central forehead box
        if np.sum(roi_mask) == 0:
            fh_y1 = max(0, ymin)
            fh_y2 = min(h, ymin + (ymax - ymin) // 3)
            fh_x1 = max(0, xmin + (xmax - xmin) // 4)
            fh_x2 = min(w, xmax - (xmax - xmin) // 4)
            roi_mask[fh_y1:fh_y2, fh_x1:fh_x2] = 255

        # Compute spatial mean RGB over ROI
        mask_bool = roi_mask > 0
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pixels = frame_rgb[mask_bool]

        if len(pixels) > 0:
            mean_rgb = np.mean(pixels, axis=0)  # [R, G, B]
        else:
            mean_rgb = np.mean(frame_rgb, axis=(0, 1))

        return {
            "mean_rgb": mean_rgb.astype(np.float64),
            "roi_mask": roi_mask,
            "face_crop_36x36": face_rgb_norm,
            "bbox": bbox,
            "skin_pixels": pixels,
        }

    def process_video(
        self,
        video_path_or_frames: any,
        mask_occlusion_active: bool = False,
    ) -> Dict[str, any]:
        """Process an entire video sequence, extracting temporal RGB traces and cropped faces."""
        if isinstance(video_path_or_frames, str):
            cap = cv2.VideoCapture(video_path_or_frames)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()
            frames = np.array(frames)
        else:
            frames = video_path_or_frames

        t_len = len(frames)
        rgb_signals = np.zeros((t_len, 3), dtype=np.float64)
        cropped_faces = np.zeros((t_len, self.crop_size[1], self.crop_size[0], 3), dtype=np.float32)
        roi_masks = np.zeros((t_len, frames.shape[1], frames.shape[2]), dtype=np.uint8)
        all_skin_pixels = []

        for t in range(t_len):
            out = self.extract_from_frame(frames[t], mask_occlusion_active=mask_occlusion_active)
            rgb_signals[t] = out["mean_rgb"]
            cropped_faces[t] = out["face_crop_36x36"]
            roi_masks[t] = out["roi_mask"]
            if t % 10 == 0 and len(out["skin_pixels"]) > 0:
                all_skin_pixels.append(out["skin_pixels"][:100])

        skin_sample = np.vstack(all_skin_pixels) if all_skin_pixels else frames[0].reshape(-1, 3)

        return {
            "rgb_signals": rgb_signals,
            "cropped_faces": cropped_faces,
            "roi_masks": roi_masks,
            "skin_sample": skin_sample,
            "frames": frames,
        }

    def close(self):
        self.tracker.close()
