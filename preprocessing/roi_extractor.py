"""ROI Extraction and Spatial Averaging Engine.

Extracts physiological Regions of Interest (forehead, bilateral cheeks) per frame,
computes spatial mean RGB color traces with specular/glare and YCbCr skin filtering,
stabilizes cropped face inputs via EMA bounding boxes for ME-rPPG, and performs
automatic dynamic spectral SNR weighting across sub-ROIs.
"""

from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from preprocessing.face_mesh import FaceMeshTracker
from estimation.czt_estimator import CZTHRTransformer


def filter_skin_pixels(pixels_rgb: np.ndarray) -> np.ndarray:
    """Filter candidate skin pixels to exclude specular glare, shadows, and non-skin.

    Applies:
    1. Intensity bounds: 20 < R, G, B < 245
    2. Chrominance skin bounds in YCbCr: 77 <= Cb <= 135 and 130 <= Cr <= 180

    Args:
        pixels_rgb: Array of RGB pixels of shape (N, 3), uint8 or float.

    Returns:
        Filtered array of skin pixels (M, 3) where M <= N.
    """
    if len(pixels_rgb) == 0:
        return pixels_rgb

    pixels_u8 = np.clip(pixels_rgb, 0, 255).astype(np.uint8)

    # 1. Intensity bounds (exclude extreme shadows and specular reflections)
    int_mask = (
        (pixels_u8[:, 0] > 20) & (pixels_u8[:, 0] < 245) &
        (pixels_u8[:, 1] > 20) & (pixels_u8[:, 1] < 245) &
        (pixels_u8[:, 2] > 20) & (pixels_u8[:, 2] < 245)
    )
    cand_pixels = pixels_u8[int_mask] if np.sum(int_mask) > 0 else pixels_u8

    # 2. Chrominance skin bounds in YCbCr (OpenCV returns [Y, Cr, Cb])
    ycbcr = cv2.cvtColor(cand_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    cr = ycbcr[:, 1]
    cb = ycbcr[:, 2]
    skin_mask = (cb >= 77) & (cb <= 135) & (cr >= 130) & (cr <= 180)

    # If >25% of candidates fall in physiological skin ellipse, retain them
    if np.sum(skin_mask) > 0.25 * len(pixels_rgb):
        return cand_pixels[skin_mask]

    return cand_pixels


def fuse_sub_rois_by_snr(
    sub_roi_rgb: Dict[str, np.ndarray],
    fs: float = 30.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fuse sub-ROI RGB traces using dynamic spectral SNR weighting.

    Autonomously down-weights occluded or noisy facial regions (e.g., C7 masks)
    based on temporal variance and CZT cardiac in-band peak-to-average power ratio.

    Args:
        sub_roi_rgb: Dict of RGB traces {'forehead': (T, 3), 'left_cheek': (T, 3), 'right_cheek': (T, 3)}.
        fs: Video sampling frequency in Hz.

    Returns:
        fused_rgb: Fused RGB trace (T, 3).
        weights: Dictionary of assigned weights {k: w_k}.
    """
    snrs = {}
    for name, rgb in sub_roi_rgb.items():
        t_len = len(rgb)
        if t_len < 30:
            snrs[name] = 1.0
            continue

        # Check temporal variance of green channel & mean intensity
        g_var = float(np.var(rgb[:, 1]))
        g_mean = float(np.mean(rgb[:, 1]))
        if g_var < 1e-4 or g_mean < 5.0:
            snrs[name] = 0.0
            continue

        # Quick POS projection
        mean_c = np.mean(rgb, axis=0, keepdims=True) + 1e-8
        cn = (rgb / mean_c) - 1.0
        s1 = cn[:, 1] - cn[:, 2]
        s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]
        std1 = float(np.std(s1)) + 1e-8
        std2 = float(np.std(s2)) + 1e-8
        h = s1 + (std1 / std2) * s2

        # CZT power spectrum in [0.7, 3.0] Hz
        czt = CZTHRTransformer(fs=fs, low_hz=0.7, high_hz=3.0, num_bins=128)
        _, pxx = czt.compute_spectrum(h)
        max_p = float(np.max(pxx))
        sum_p = float(np.sum(pxx)) + 1e-12
        snrs[name] = max_p / sum_p

    snr_sq = {k: v**2 for k, v in snrs.items()}
    tot_sq = sum(snr_sq.values())
    if tot_sq > 1e-12:
        weights = {k: float(v / tot_sq) for k, v in snr_sq.items()}
    else:
        weights = {k: 1.0 / len(sub_roi_rgb) for k in sub_roi_rgb}

    fused_rgb = np.zeros_like(next(iter(sub_roi_rgb.values())), dtype=np.float64)
    for k, w in weights.items():
        fused_rgb += w * sub_roi_rgb[k]

    return fused_rgb, weights


class ROIExtractor:
    """Extracts physiological skin regions, stabilized crops, and spatial RGB color traces."""

    def __init__(
        self,
        target_crop_size: Tuple[int, int] = (36, 36),
        bbox_ema_alpha: float = 0.15,
        crop_margin: float = 1.25,
    ):
        self.tracker = FaceMeshTracker()
        self.crop_size = target_crop_size
        self.bbox_ema_alpha = bbox_ema_alpha
        self.crop_margin = crop_margin
        self._ema_bbox: Optional[np.ndarray] = None

    def reset(self):
        """Reset temporal state such as EMA bounding box."""
        self._ema_bbox = None

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

        # Temporal Bounding-Box Stabilization (Crucial for ME-rPPG)
        bw = max(1.0, float(xmax - xmin))
        bh = max(1.0, float(ymax - ymin))
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        side = max(bw, bh) * self.crop_margin
        b_square = np.array([cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0], dtype=np.float32)

        if self._ema_bbox is None:
            self._ema_bbox = b_square.copy()
        else:
            old_cx = (self._ema_bbox[0] + self._ema_bbox[2]) / 2.0
            old_cy = (self._ema_bbox[1] + self._ema_bbox[3]) / 2.0
            dist = np.sqrt((cx - old_cx) ** 2 + (cy - old_cy) ** 2)
            if dist < 0.5 * bw:
                self._ema_bbox = (1.0 - self.bbox_ema_alpha) * self._ema_bbox + self.bbox_ema_alpha * b_square
            else:
                self._ema_bbox = b_square.copy()

        sx1 = max(0, int(np.round(self._ema_bbox[0])))
        sy1 = max(0, int(np.round(self._ema_bbox[1])))
        sx2 = min(w, int(np.round(self._ema_bbox[2])))
        sy2 = min(h, int(np.round(self._ema_bbox[3])))

        face_crop = frame_bgr[sy1:sy2, sx1:sx2]
        if face_crop.size == 0:
            face_crop = frame_bgr[max(0, ymin):min(h, ymax), max(0, xmin):min(w, xmax)]
            if face_crop.size == 0:
                face_crop = frame_bgr

        face_resized = cv2.resize(face_crop, self.crop_size, interpolation=cv2.INTER_AREA)
        face_rgb_norm = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Construct sub-ROI masks
        mask_fh = np.zeros((h, w), dtype=np.uint8)
        mask_lc = np.zeros((h, w), dtype=np.uint8)
        mask_rc = np.zeros((h, w), dtype=np.uint8)

        # 1. Forehead ROI
        fh_pts = res.get("forehead_pts")
        if fh_pts is not None and len(fh_pts) >= 3:
            cv2.fillConvexPoly(mask_fh, cv2.convexHull(fh_pts), 255)
        else:
            fh_y1 = max(0, ymin)
            fh_y2 = min(h, ymin + int(bh // 3))
            fh_x1 = max(0, xmin + int(bw // 4))
            fh_x2 = min(w, xmax - int(bw // 4))
            mask_fh[fh_y1:fh_y2, fh_x1:fh_x2] = 255

        # 2. Left & Right Cheeks
        l_pts = res.get("left_cheek_pts")
        if l_pts is not None and len(l_pts) >= 3:
            cv2.fillConvexPoly(mask_lc, cv2.convexHull(l_pts), 255)

        r_pts = res.get("right_cheek_pts")
        if r_pts is not None and len(r_pts) >= 3:
            cv2.fillConvexPoly(mask_rc, cv2.convexHull(r_pts), 255)

        # Combined ROI mask
        if mask_occlusion_active:
            roi_mask = mask_fh.copy()
        else:
            roi_mask = mask_fh | mask_lc | mask_rc

        if np.sum(roi_mask) == 0:
            roi_mask = mask_fh.copy()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Extract & filter pixels for sub-ROIs
        def _get_mean_rgb(mask):
            m_bool = mask > 0
            if not np.any(m_bool):
                return np.mean(frame_rgb, axis=(0, 1))
            pix = frame_rgb[m_bool]
            filt_pix = filter_skin_pixels(pix)
            if len(filt_pix) > 0:
                return np.mean(filt_pix, axis=0)
            return np.mean(pix, axis=0)

        fh_mean = _get_mean_rgb(mask_fh)
        lc_mean = _get_mean_rgb(mask_lc)
        rc_mean = _get_mean_rgb(mask_rc)

        # Active combined ROI mean
        comb_pix = frame_rgb[roi_mask > 0]
        filt_comb_pix = filter_skin_pixels(comb_pix)
        if len(filt_comb_pix) > 0:
            mean_rgb = np.mean(filt_comb_pix, axis=0)
        elif len(comb_pix) > 0:
            mean_rgb = np.mean(comb_pix, axis=0)
        else:
            mean_rgb = np.mean(frame_rgb, axis=(0, 1))

        return {
            "mean_rgb": mean_rgb.astype(np.float64),
            "roi_mask": roi_mask,
            "face_crop_36x36": face_rgb_norm,
            "bbox": bbox,
            "skin_pixels": filt_comb_pix if len(filt_comb_pix) > 0 else comb_pix,
            "sub_rois": {
                "forehead": fh_mean.astype(np.float64),
                "left_cheek": lc_mean.astype(np.float64),
                "right_cheek": rc_mean.astype(np.float64),
            },
        }

    def process_video(
        self,
        video_path_or_frames: any,
        mask_occlusion_active: bool = False,
    ) -> Dict[str, any]:
        """Process an entire video sequence, extracting temporal RGB traces and cropped faces."""
        self.reset()

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
        if t_len == 0:
            raise ValueError(f"No valid frames found in input: {video_path_or_frames}")
        rgb_signals = np.zeros((t_len, 3), dtype=np.float64)
        sub_roi_rgb = {
            "forehead": np.zeros((t_len, 3), dtype=np.float64),
            "left_cheek": np.zeros((t_len, 3), dtype=np.float64),
            "right_cheek": np.zeros((t_len, 3), dtype=np.float64),
        }
        cropped_faces = np.zeros((t_len, self.crop_size[1], self.crop_size[0], 3), dtype=np.float32)
        roi_masks = np.zeros((t_len, frames.shape[1], frames.shape[2]), dtype=np.uint8)
        all_skin_pixels = []

        for t in range(t_len):
            out = self.extract_from_frame(frames[t], mask_occlusion_active=mask_occlusion_active)
            rgb_signals[t] = out["mean_rgb"]
            cropped_faces[t] = out["face_crop_36x36"]
            roi_masks[t] = out["roi_mask"]
            for k in sub_roi_rgb:
                sub_roi_rgb[k][t] = out["sub_rois"][k]

            if t % 10 == 0 and len(out["skin_pixels"]) > 0:
                all_skin_pixels.append(out["skin_pixels"][:100])

        skin_sample = np.vstack(all_skin_pixels) if all_skin_pixels else frames[0].reshape(-1, 3)

        # Dynamic spectral SNR weighting across sub-ROIs
        if mask_occlusion_active:
            sub_roi_weights = {"forehead": 1.0, "left_cheek": 0.0, "right_cheek": 0.0}
        elif t_len >= 30:
            fused_rgb, sub_roi_weights = fuse_sub_rois_by_snr(sub_roi_rgb, fs=30.0)
            rgb_signals = fused_rgb
        else:
            sub_roi_weights = {"forehead": 0.333, "left_cheek": 0.333, "right_cheek": 0.333}

        return {
            "rgb_signals": rgb_signals,
            "cropped_faces": cropped_faces,
            "roi_masks": roi_masks,
            "skin_sample": skin_sample,
            "frames": frames,
            "sub_roi_rgb": sub_roi_rgb,
            "sub_roi_weights": sub_roi_weights,
        }

    def close(self):
        self.tracker.close()
