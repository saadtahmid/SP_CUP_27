"""ROI Extraction, NMF Spatial-Temporal Attention, and Spatial Averaging Engine.

Extracts physiological Regions of Interest (forehead, bilateral cheeks) per frame,
computes spatial mean RGB color traces with adaptive low-light and YCbCr skin filtering,
stabilizes cropped face inputs via EMA bounding boxes for ME-rPPG, implements Rank-1 NMF
attention (FactorizePhys FSAM), and performs skin-occupancy + spectral SNR sub-ROI fusion.
"""

from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from scipy import signal

from preprocessing.face_mesh import FaceMeshTracker
from estimation.czt_estimator import CZTHRTransformer


def filter_skin_pixels(pixels_rgb: np.ndarray, return_ratio: bool = False):
    """Filter candidate skin pixels to exclude specular glare, shadows, and non-skin Masks.

    Applies:
    1. Adaptive intensity bounds: 3 < R, G, B < 245 (supports C1 10-lux low-light frames)
    2. Chrominance skin bounds in YCbCr: 77 <= Cb <= 135 and 130 <= Cr <= 180

    Args:
        pixels_rgb: Array of RGB pixels of shape (N, 3), uint8 or float.
        return_ratio: If True, also return the fraction of pixels passing YCbCr skin bounds.

    Returns:
        Filtered array of skin pixels (M, 3), and optionally skin_ratio in [0.0, 1.0].
    """
    if len(pixels_rgb) == 0:
        return (pixels_rgb, 0.0) if return_ratio else pixels_rgb

    pixels_u8 = np.clip(pixels_rgb, 0, 255).astype(np.uint8)

    # 1. Adaptive intensity bounds (lower bound 3 preserves C1 10-lux attenuated skin pixels)
    int_mask = (
        (pixels_u8[:, 0] > 3) & (pixels_u8[:, 0] < 245) &
        (pixels_u8[:, 1] > 3) & (pixels_u8[:, 1] < 245) &
        (pixels_u8[:, 2] > 3) & (pixels_u8[:, 2] < 245)
    )
    cand_pixels = pixels_u8[int_mask] if np.sum(int_mask) > 0 else pixels_u8

    # 2. Chrominance skin bounds in YCbCr (OpenCV returns [Y, Cr, Cb])
    ycbcr = cv2.cvtColor(cand_pixels.reshape(1, -1, 3), cv2.COLOR_RGB2YCrCb).reshape(-1, 3)
    y_lum = ycbcr[:, 0]
    cr = ycbcr[:, 1]
    cb = ycbcr[:, 2]

    # Explicit rejection for cyan/blue surgical masks (Cr < 110 and Cb > 138)
    non_mask = ~((cb > 138) & (cr < 110))

    # In extreme low light (mean Y < 25), YCbCr quantized chrominance shifts toward 128 neutral
    if np.mean(y_lum) < 25.0:
        skin_mask = (cb >= 100) & (cb <= 145) & (cr >= 115) & (cr <= 165) & non_mask
    else:
        skin_mask = (cb >= 77) & (cb <= 135) & (cr >= 125) & (cr <= 180) & non_mask

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
    """Rank-1 Non-Negative Matrix Factorization (NMF) Multi-ROI Attention (FactorizePhys FSAM).

    Implements FactorizePhys (Joshi & Cho, NeurIPS 2024, Eqs. 6-9 & Table 1):
    Factorizes non-negative spatial-temporal matrix V_st in R_{>=0}^{T x 3K} into W (T x 1)
    and H (1 x 3K) via 6 multiplicative updates, extracting common pulsatile variance.
    """
    keys = list(sub_roi_rgb.keys())
    traces = [np.asarray(sub_roi_rgb[k], dtype=np.float64) for k in keys]
    t_len = len(traces[0])

    # Zero-mean temporal fluctuation shifted to non-negative domain
    norm_cols = []
    for tr in traces:
        mean_c = np.mean(tr, axis=0, keepdims=True) + 1e-8
        ac = (tr / mean_c) - 1.0
        ac_pos = ac - np.min(ac, axis=0, keepdims=True) + 1e-6
        norm_cols.append(ac_pos)

    v_st = np.hstack(norm_cols)  # shape (T, 3*K)
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
    """Fuse sub-ROI RGB traces using dynamic spectral SNR and skin occupancy weighting.

    Autonomously down-weights occluded or noisy facial regions (e.g., C7 surgical masks)
    based on YCbCr skin occupancy ratio, temporal variance, and CZT cardiac in-band SNR.
    """
    snrs = {}
    _, nmf_scores = rank1_nmf_attention(sub_roi_rgb, n_steps=6)

    order = 2 if fs < 18.0 else 4
    b_bp, a_bp = signal.butter(order, [0.7 / (fs / 2.0), min(0.95, 3.0 / (fs / 2.0))], btype="bandpass")

    max_skin_ratio = max(sub_roi_skin_ratios.values()) if sub_roi_skin_ratios else 1.0

    for name, rgb in sub_roi_rgb.items():
        t_len = len(rgb)
        if t_len < 20:
            snrs[name] = 1.0
            continue

        # 1. Check skin pixel occupancy ratio (rejects C7 cyan surgical masks covering cheeks)
        skin_ratio = 1.0
        if sub_roi_skin_ratios is not None and name in sub_roi_skin_ratios:
            skin_ratio = float(sub_roi_skin_ratios[name])
            if skin_ratio < 0.35 or (max_skin_ratio > 0.80 and skin_ratio < 0.65 * max_skin_ratio):
                snrs[name] = 0.0
                continue

        # 2. Check normalized temporal variance of green channel & mean intensity
        g_mean = float(np.mean(rgb[:, 1]))
        if g_mean < 2.0:
            snrs[name] = 0.0
            continue

        g_norm_var = float(np.var((rgb[:, 1] / (g_mean + 1e-8)) - 1.0))
        if g_norm_var < 1e-7:
            snrs[name] = 0.0
            continue

        # 3. Bandpass-filtered POS projection to suppress out-of-band respiration leakage
        mean_c = np.mean(rgb, axis=0, keepdims=True) + 1e-8
        cn = (rgb / mean_c) - 1.0
        padlen = min(t_len - 1, 3 * max(len(a_bp), len(b_bp)))
        if padlen > 0 and t_len > 8:
            cn_f = signal.filtfilt(b_bp, a_bp, cn, axis=0, padlen=padlen)
        else:
            cn_f = signal.detrend(cn, axis=0)

        s1 = cn_f[:, 1] - cn_f[:, 2]
        s2 = cn_f[:, 1] + cn_f[:, 2] - 2.0 * cn_f[:, 0]
        std1 = float(np.std(s1)) + 1e-8
        std2 = float(np.std(s2)) + 1e-8
        h = s1 + (std1 / std2) * s2

        # 4. CZT power spectrum in cardiac band
        czt = CZTHRTransformer(fs=fs, low_hz=0.7, high_hz=3.0, num_bins=128)
        bpm_bins, pxx = czt.compute_spectrum(h)
        cardiac_zone = (bpm_bins >= 60.0) & (bpm_bins <= 145.0)
        pxx_eval = pxx[cardiac_zone] if np.any(cardiac_zone) else pxx

        max_p = float(np.max(pxx_eval))
        sum_p = float(np.sum(pxx_eval)) + 1e-12
        spectral_conc = max_p / sum_p

        nmf_w = nmf_scores.get(name, 0.333)
        snrs[name] = spectral_conc * (skin_ratio ** 4) * (0.7 + 0.3 * nmf_w)

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
        self._tight_bbox: Optional[np.ndarray] = None
        self._static_non_mask: Optional[np.ndarray] = None

    def reset(self):
        """Reset temporal state such as EMA bounding box and static non-mask domain."""
        self._ema_bbox = None
        self._tight_bbox = None
        self._static_non_mask = None

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

        bw = max(1.0, float(xmax - xmin))
        bh = max(1.0, float(ymax - ymin))
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        side = max(bw, bh) * self.crop_margin
        b_square = np.array(
            [cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0],
            dtype=np.float32,
        )
        b_tight = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

        if self._ema_bbox is None:
            self._ema_bbox = b_square.copy()
            self._tight_bbox = b_tight.copy()
        else:
            old_cx = (self._ema_bbox[0] + self._ema_bbox[2]) / 2.0
            old_cy = (self._ema_bbox[1] + self._ema_bbox[3]) / 2.0
            dist = np.sqrt((cx - old_cx) ** 2 + (cy - old_cy) ** 2)
            if dist < 0.25 * bw:
                # Lock bounding boxes for conversational/respiratory motion (< 25% of face width)
                pass
            elif dist < 0.6 * bw:
                self._ema_bbox = (1.0 - self.bbox_ema_alpha) * self._ema_bbox + self.bbox_ema_alpha * b_square
                self._tight_bbox = (1.0 - self.bbox_ema_alpha) * self._tight_bbox + self.bbox_ema_alpha * b_tight
            else:
                self._ema_bbox = b_square.copy()
                self._tight_bbox = b_tight.copy()

        sx1 = max(0, int(np.round(self._ema_bbox[0])))
        sy1 = max(0, int(np.round(self._ema_bbox[1])))
        sx2 = min(w, int(np.round(self._ema_bbox[2])))
        sy2 = min(h, int(np.round(self._ema_bbox[3])))

        fx1 = max(0, int(np.round(self._tight_bbox[0])))
        fy1 = max(0, int(np.round(self._tight_bbox[1])))
        fx2 = min(w, int(np.round(self._tight_bbox[2])))
        fy2 = min(h, int(np.round(self._tight_bbox[3])))

        face_crop = frame_bgr[sy1:sy2, sx1:sx2]
        if face_crop.size == 0:
            face_crop = frame_bgr[max(0, ymin):min(h, ymax), max(0, xmin):min(w, xmax)]
            if face_crop.size == 0:
                face_crop = frame_bgr

        face_resized = cv2.resize(face_crop, self.crop_size, interpolation=cv2.INTER_AREA)
        face_rgb_norm = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Construct sub-ROI masks inside stabilized tight face box [fx1, fy1, fx2, fy2]
        mask_fh = np.zeros((h, w), dtype=np.uint8)
        mask_lc = np.zeros((h, w), dtype=np.uint8)
        mask_rc = np.zeros((h, w), dtype=np.uint8)

        f_w = max(1, fx2 - fx1)
        f_h = max(1, fy2 - fy1)

        # 1. Stabilized Forehead ROI (safely below hairline and above eyebrows: 8% to 25% of tight face height)
        fh_y1 = max(0, fy1 + int(0.08 * f_h))
        fh_y2 = min(h, fy1 + int(0.25 * f_h))
        fh_x1 = max(0, fx1 + int(0.25 * f_w))
        fh_x2 = min(w, fx1 + int(0.75 * f_w))
        mask_fh[fh_y1:fh_y2, fh_x1:fh_x2] = 255

        # 2. Stabilized Left & Right Cheek ROIs (54% to 78% of tight face height — covered by C7 surgical masks)
        ch_y1 = max(0, fy1 + int(0.54 * f_h))
        ch_y2 = min(h, fy1 + int(0.78 * f_h))
        lc_x1, lc_x2 = max(0, fx1 + int(0.18 * f_w)), min(w, fx1 + int(0.42 * f_w))
        rc_x1, rc_x2 = max(0, fx1 + int(0.58 * f_w)), min(w, fx1 + int(0.82 * f_w))
        mask_lc[ch_y1:ch_y2, lc_x1:lc_x2] = 255
        mask_rc[ch_y1:ch_y2, rc_x1:rc_x2] = 255

        if mask_occlusion_active:
            roi_mask = mask_fh.copy()
        else:
            roi_mask = mask_fh | mask_lc | mask_rc

        if np.sum(roi_mask) == 0:
            roi_mask = mask_fh.copy()

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Lock static non-mask integration domain on Frame 0
        if self._static_non_mask is None or self._static_non_mask.shape != (h, w):
            ycbcr_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            cr_full = ycbcr_full[:, :, 1]
            cb_full = ycbcr_full[:, :, 2]
            self._static_non_mask = ~((cb_full > 138) & (cr_full < 110))
            if not np.any(self._static_non_mask):
                self._static_non_mask = np.ones((h, w), dtype=bool)

        holistic_rgb = np.mean(frame_rgb[self._static_non_mask], axis=0).astype(np.float64)

        def _get_mean_rgb_and_ratio(mask):
            m_bool = mask > 0
            if not np.any(m_bool):
                return holistic_rgb, 0.0
            pix = frame_rgb[m_bool]
            _, s_ratio = filter_skin_pixels(pix, return_ratio=True)
            local_mean = np.mean(pix, axis=0).astype(np.float64)
            if s_ratio >= 0.60:
                return 0.92 * holistic_rgb + 0.08 * local_mean, s_ratio
            return local_mean, s_ratio

        fh_mean, fh_ratio = _get_mean_rgb_and_ratio(mask_fh)
        lc_mean, lc_ratio = _get_mean_rgb_and_ratio(mask_lc)
        rc_mean, rc_ratio = _get_mean_rgb_and_ratio(mask_rc)

        comb_pix = frame_rgb[roi_mask > 0]
        filt_comb_pix = filter_skin_pixels(comb_pix)
        mean_rgb = holistic_rgb

        return {
            "mean_rgb": mean_rgb,
            "roi_mask": roi_mask,
            "face_crop_36x36": face_rgb_norm,
            "bbox": bbox,
            "skin_pixels": filt_comb_pix if len(filt_comb_pix) > 0 else comb_pix,
            "sub_rois": {
                "forehead": fh_mean,
                "left_cheek": lc_mean,
                "right_cheek": rc_mean,
            },
            "sub_roi_skin_ratios": {
                "forehead": fh_ratio,
                "left_cheek": lc_ratio,
                "right_cheek": rc_ratio,
            },
        }

    def process_video(
        self,
        video_path_or_frames: any,
        mask_occlusion_active: bool = False,
        fs: float = 30.0,
    ) -> Dict[str, any]:
        """Process an entire video sequence, extracting temporal RGB traces and cropped faces."""
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

        for t in range(t_len):
            out = self.extract_from_frame(frames[t], mask_occlusion_active=mask_occlusion_active)
            rgb_signals[t] = out["mean_rgb"]
            cropped_faces[t] = out["face_crop_36x36"]
            roi_masks[t] = out["roi_mask"]
            for k in sub_roi_rgb:
                sub_roi_rgb[k][t] = out["sub_rois"][k]
                sub_roi_ratios_acc[k].append(out["sub_roi_skin_ratios"][k])

            if t % 10 == 0 and len(out["skin_pixels"]) > 0:
                all_skin_pixels.append(out["skin_pixels"][:100])

        mean_crop_lum = float(np.mean(cropped_faces))
        if 1e-4 < mean_crop_lum < 0.35:
            video_gain = min(15.0, 0.48 / mean_crop_lum)
            cropped_faces = np.clip(cropped_faces * video_gain, 0.0, 1.0)

        skin_sample = np.vstack(all_skin_pixels) if all_skin_pixels else frames[0].reshape(-1, 3)
        mean_skin_ratios = {k: float(np.mean(v)) for k, v in sub_roi_ratios_acc.items()}

        if mask_occlusion_active:
            sub_roi_weights = {"forehead": 1.0, "left_cheek": 0.0, "right_cheek": 0.0}
            rgb_signals = sub_roi_rgb["forehead"].copy()
        elif t_len >= 20:
            fused_rgb, sub_roi_weights = fuse_sub_rois_by_snr(
                sub_roi_rgb,
                fs=fs,
                sub_roi_skin_ratios=mean_skin_ratios,
            )
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
            "sub_roi_skin_ratios": mean_skin_ratios,
        }

    def close(self):
        self.tracker.close()

