"""ROI Extraction, NMF Spatial-Temporal Attention, and Spatial Averaging Engine.

Implements the Stage 1B verified R4 Locked Sub-ROI & Crop Extraction:
1. Locks the primary face bounding box [x1, y1, x2, y2] and 1.25x square crop on Frame 0
   to eliminate per-frame bounding-box and threshold-switching jitter.
2. Extracts continuous Forehead (y: 12%-28%, x: 25%-75%) and Bilateral Cheek
   (y: 50%-75%, x: 18%-42% & 58%-82%) sub-ROI traces (R4 union mask).
3. Autonomously rejects lower-face surgical mask occlusion (C7) when >35% of the cheek
   sub-ROIs are covered by surgical mask pixels (Cb > 138 and Cr < 110).
"""

from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from preprocessing.face_mesh import FaceMeshTracker
from estimation.czt_estimator import CZTHRTransformer


def filter_skin_pixels(pixels_rgb: np.ndarray, return_ratio: bool = False):
    """Filter candidate skin pixels to exclude specular glare, shadows, and non-skin masks."""
    if len(pixels_rgb) == 0:
        return (pixels_rgb, 0.0) if return_ratio else pixels_rgb

    pixels_u8 = np.clip(pixels_rgb, 0, 255).astype(np.uint8)
    int_mask = (
        (pixels_u8[:, 0] > 3) & (pixels_u8[:, 0] < 245) &
        (pixels_u8[:, 1] > 3) & (pixels_u8[:, 1] < 245) &
        (pixels_u8[:, 2] > 3) & (pixels_u8[:, 2] < 245)
    )
    cand_pixels = pixels_u8[int_mask] if np.sum(int_mask) > 0 else pixels_u8

    ycbcr = cv2.cvtColor(cand_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    y_lum = ycbcr[:, 0]
    cr = ycbcr[:, 1]
    cb = ycbcr[:, 2]

    non_mask = ~((cb > 138) & (cr < 110))
    if np.mean(y_lum) < 25.0:
        skin_mask = (cb >= 100) & (cb <= 145) & (cr >= 115) & (cr <= 165) & non_mask
    else:
        skin_mask = (cb >= 77) & (cb <= 135) & (cr >= 130) & (cr <= 180) & non_mask

    skin_ratio = float(np.sum(skin_mask)) / float(max(1, len(pixels_rgb)))
    if np.sum(skin_mask) > 0.25 * len(pixels_rgb):
        filtered = cand_pixels[skin_mask]
    else:
        filtered = cand_pixels

    if return_ratio:
        return filtered, skin_ratio
    return filtered


def rank1_nmf_attention(
    sub_roi_rgb: Dict[str, np.ndarray],
    n_steps: int = 6,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Rank-1 Non-Negative Matrix Factorization (NMF) Multi-ROI Attention (FactorizePhys FSAM)."""
    keys = list(sub_roi_rgb.keys())
    traces = [np.asarray(sub_roi_rgb[k], dtype=np.float64) for k in keys]
    t_len = len(traces[0])

    norm_cols = []
    for tr in traces:
        mean_c = np.mean(tr, axis=0, keepdims=True) + 1e-8
        ac = (tr / mean_c) - 1.0
        ac_pos = ac - np.min(ac, axis=0, keepdims=True) + 1e-6
        norm_cols.append(ac_pos)

    v_st = np.hstack(norm_cols)
    m_dim, n_dim = v_st.shape

    rng = np.random.default_rng(42)
    w_mat = rng.uniform(0.1, 1.0, size=(m_dim, 1))
    h_mat = rng.uniform(0.1, 1.0, size=(1, n_dim))

    for _ in range(max(1, n_steps)):
        wt_v = w_mat.T @ v_st
        wt_w_h = (w_mat.T @ w_mat) @ h_mat + 1e-8
        h_mat = h_mat * (wt_v / wt_w_h)

        v_ht = v_st @ h_mat.T
        w_h_ht = w_mat @ (h_mat @ h_mat.T) + 1e-8
        w_mat = w_mat * (v_ht / w_h_ht)

    h_flat = h_mat.flatten()
    roi_raw = {}
    for idx, k in enumerate(keys):
        roi_raw[k] = float(np.mean(h_flat[3 * idx : 3 * (idx + 1)]))

    tot = sum(roi_raw.values()) + 1e-12
    roi_scores = {k: v / tot for k, v in roi_raw.items()}

    recon_rgb = np.zeros((t_len, 3), dtype=np.float64)
    for k, w in roi_scores.items():
        recon_rgb += w * sub_roi_rgb[k]

    return recon_rgb, roi_scores


def fuse_sub_rois_by_snr(
    sub_roi_rgb: Dict[str, np.ndarray],
    fs: float = 30.0,
    sub_roi_skin_ratios: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fuse sub-ROI RGB traces using dynamic spectral SNR and skin occupancy weighting."""
    snrs = {}
    _, nmf_scores = rank1_nmf_attention(sub_roi_rgb, n_steps=6)

    for name, rgb in sub_roi_rgb.items():
        t_len = len(rgb)
        if t_len < 20:
            snrs[name] = 1.0
            continue

        skin_ratio = 1.0
        if sub_roi_skin_ratios is not None and name in sub_roi_skin_ratios:
            skin_ratio = float(sub_roi_skin_ratios[name])
            if skin_ratio < 0.25:
                snrs[name] = 0.0
                continue

        g_mean = float(np.mean(rgb[:, 1]))
        if g_mean < 2.0:
            snrs[name] = 0.0
            continue

        g_norm_var = float(np.var((rgb[:, 1] / (g_mean + 1e-8)) - 1.0))
        if g_norm_var < 1e-7:
            snrs[name] = 0.0
            continue

        mean_c = np.mean(rgb, axis=0, keepdims=True) + 1e-8
        cn = (rgb / mean_c) - 1.0
        s1 = cn[:, 1] - cn[:, 2]
        s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]
        std1 = float(np.std(s1)) + 1e-8
        std2 = float(np.std(s2)) + 1e-8
        h = s1 + (std1 / std2) * s2

        czt = CZTHRTransformer(fs=fs, low_hz=0.7, high_hz=3.0, num_bins=128)
        bpm_bins, pxx = czt.compute_spectrum(h)
        interior = bpm_bins >= 45.0
        pxx_eval = pxx[interior] if np.any(interior) else pxx

        max_p = float(np.max(pxx_eval))
        sum_p = float(np.sum(pxx_eval)) + 1e-12
        spectral_conc = max_p / sum_p

        nmf_w = nmf_scores.get(name, 0.333)
        snrs[name] = spectral_conc * (skin_ratio ** 2) * (0.7 + 0.3 * nmf_w)

    snr_sq = {k: v ** 2 for k, v in snrs.items()}
    tot_sq = sum(snr_sq.values())
    if tot_sq > 1e-12:
        weights = {k: float(v / tot_sq) for k, v in snr_sq.items()}
    else:
        weights = {"forehead": 1.0, "left_cheek": 0.0, "right_cheek": 0.0}

    fused_rgb = np.zeros_like(next(iter(sub_roi_rgb.values())), dtype=np.float64)
    for k, w in weights.items():
        fused_rgb += w * sub_roi_rgb[k]

    return fused_rgb, weights


class ROIExtractor:
    """Extracts Stage 1B verified R4 locked facial sub-ROIs and stabilized 36x36 crops."""

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
        self._locked_bbox: Optional[np.ndarray] = None

    def reset(self):
        """Reset locked bounding box state."""
        self._locked_bbox = None

    def extract_from_frame(
        self,
        frame_bgr: np.ndarray,
        mask_occlusion_active: bool = False,
    ) -> Dict[str, any]:
        """Extract R4 sub-ROI RGB means and 36x36 face crop from a single frame."""
        h, w = frame_bgr.shape[:2]
        if self._locked_bbox is None:
            res = self.tracker.process_frame(frame_bgr)
            self._locked_bbox = res["bbox"].copy()

        x1, y1, x2, y2 = [int(v) for v in self._locked_bbox]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(bw, bh) * self.crop_margin
        sx1 = max(0, int(round(cx - side / 2.0)))
        sy1 = max(0, int(round(cy - side / 2.0)))
        sx2 = min(w, int(round(cx + side / 2.0)))
        sy2 = min(h, int(round(cy + side / 2.0)))

        face_crop = frame_bgr[sy1:sy2, sx1:sx2]
        if face_crop.size == 0:
            face_crop = frame_bgr
        face_resized = cv2.resize(face_crop, self.crop_size, interpolation=cv2.INTER_AREA)
        face_rgb_norm = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # R4 Locked Sub-ROI Coordinates inside [x1, y1, x2, y2]
        mask_fh = np.zeros((h, w), dtype=np.uint8)
        mask_lc = np.zeros((h, w), dtype=np.uint8)
        mask_rc = np.zeros((h, w), dtype=np.uint8)

        fh_y1, fh_y2 = max(0, y1 + int(0.12 * bh)), min(h, y1 + int(0.28 * bh))
        fh_x1, fh_x2 = max(0, x1 + int(0.25 * bw)), min(w, x1 + int(0.75 * bw))
        ch_y1, ch_y2 = max(0, y1 + int(0.50 * bh)), min(h, y1 + int(0.75 * bh))
        lc_x1, lc_x2 = max(0, x1 + int(0.18 * bw)), min(w, x1 + int(0.42 * bw))
        rc_x1, rc_x2 = max(0, x1 + int(0.58 * bw)), min(w, x1 + int(0.82 * bw))

        mask_fh[fh_y1:fh_y2, fh_x1:fh_x2] = 255
        mask_lc[ch_y1:ch_y2, lc_x1:lc_x2] = 255
        mask_rc[ch_y1:ch_y2, rc_x1:rc_x2] = 255

        # Detect lower-face surgical mask occlusion strictly inside cheek sub-ROIs (>35% cyan occupancy)
        cheek_bgr = frame_bgr[(mask_lc | mask_rc) > 0]
        has_lower_mask = False
        if len(cheek_bgr) > 0:
            cheek_ycrcb = cv2.cvtColor(cheek_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
            cyan_frac = float(np.mean((cheek_ycrcb[:, 2] > 138) & (cheek_ycrcb[:, 1] < 110)))
            if cyan_frac > 0.35:
                has_lower_mask = True

        if mask_occlusion_active or has_lower_mask:
            roi_mask = mask_fh.copy()
        else:
            roi_mask = mask_fh | mask_lc | mask_rc

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        fh_mean = np.mean(frame_rgb[mask_fh > 0], axis=0).astype(np.float64)
        lc_mean = np.mean(frame_rgb[mask_lc > 0], axis=0).astype(np.float64)
        rc_mean = np.mean(frame_rgb[mask_rc > 0], axis=0).astype(np.float64)
        mean_rgb = np.mean(frame_rgb[roi_mask > 0], axis=0).astype(np.float64)

        comb_pix = frame_rgb[roi_mask > 0]
        filt_comb_pix, fh_ratio = filter_skin_pixels(frame_rgb[mask_fh > 0], return_ratio=True)
        _, lc_ratio = filter_skin_pixels(frame_rgb[mask_lc > 0], return_ratio=True)
        _, rc_ratio = filter_skin_pixels(frame_rgb[mask_rc > 0], return_ratio=True)
        if has_lower_mask or mask_occlusion_active:
            lc_ratio, rc_ratio = 0.0, 0.0

        return {
            "mean_rgb": mean_rgb,
            "roi_mask": roi_mask,
            "face_crop_36x36": face_rgb_norm,
            "bbox": self._locked_bbox.copy(),
            "skin_pixels": filt_comb_pix if len(filt_comb_pix) > 0 else comb_pix,
            "sub_rois": {
                "forehead": fh_mean,
                "left_cheek": lc_mean,
                "right_cheek": rc_mean,
            },
            "sub_roi_skin_ratios": {
                "forehead": float(max(0.85, fh_ratio)),
                "left_cheek": float(lc_ratio),
                "right_cheek": float(rc_ratio),
            },
            "has_lower_mask": has_lower_mask or mask_occlusion_active,
        }

    def process_video(
        self,
        video_path_or_frames: any,
        mask_occlusion_active: bool = False,
        fs: float = 30.0,
    ) -> Dict[str, any]:
        """Process an entire video sequence using Stage 1B verified R4 locked sub-ROIs."""
        self.reset()

        if isinstance(video_path_or_frames, str):
            cap = cv2.VideoCapture(video_path_or_frames)
            vid_fps = cap.get(cv2.CAP_PROP_FPS)
            if vid_fps and vid_fps > 1.0:
                fs = float(vid_fps)
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
        sub_roi_ratios_acc = {"forehead": [], "left_cheek": [], "right_cheek": []}
        cropped_faces = np.zeros((t_len, self.crop_size[1], self.crop_size[0], 3), dtype=np.float32)
        roi_masks = np.zeros((t_len, frames.shape[1], frames.shape[2]), dtype=np.uint8)
        all_skin_pixels = []
        mask_frames_count = 0

        for t in range(t_len):
            out = self.extract_from_frame(frames[t], mask_occlusion_active=mask_occlusion_active)
            rgb_signals[t] = out["mean_rgb"]
            cropped_faces[t] = out["face_crop_36x36"]
            roi_masks[t] = out["roi_mask"]
            if out.get("has_lower_mask", False):
                mask_frames_count += 1
            for k in sub_roi_rgb:
                sub_roi_rgb[k][t] = out["sub_rois"][k]
                sub_roi_ratios_acc[k].append(out["sub_roi_skin_ratios"][k])

            if t % 10 == 0 and len(out["skin_pixels"]) > 0:
                all_skin_pixels.append(out["skin_pixels"][:100])

        mask_detected = mask_occlusion_active or (mask_frames_count > 0.5 * t_len)

        mean_crop_lum = float(np.mean(cropped_faces))
        if 1e-4 < mean_crop_lum < 0.35:
            video_gain = min(15.0, 0.48 / mean_crop_lum)
            cropped_faces = np.clip(cropped_faces * video_gain, 0.0, 1.0)

        skin_sample = np.vstack(all_skin_pixels) if all_skin_pixels else frames[0].reshape(-1, 3)
        mean_skin_ratios = {k: float(np.mean(v)) for k, v in sub_roi_ratios_acc.items()}

        if mask_detected:
            sub_roi_weights = {"forehead": 1.0, "left_cheek": 0.0, "right_cheek": 0.0}
            rgb_signals = sub_roi_rgb["forehead"].copy()
        else:
            sub_roi_weights = {"forehead": 0.25, "left_cheek": 0.375, "right_cheek": 0.375}

        return {
            "rgb_signals": rgb_signals,
            "cropped_faces": cropped_faces,
            "roi_masks": roi_masks,
            "skin_sample": skin_sample,
            "frames": frames,
            "sub_roi_rgb": sub_roi_rgb,
            "sub_roi_weights": sub_roi_weights,
            "sub_roi_skin_ratios": mean_skin_ratios,
        }

    def close(self):
        self.tracker.close()
