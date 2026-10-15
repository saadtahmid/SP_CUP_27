"""Evaluation metrics for Remote Photoplethysmography (rPPG).

Implements official 2027 IEEE Signal Processing Cup evaluation metrics (§2.3 Task 1):
- Mean Absolute Error (MAE) in BPM
- Root Mean Square Error (RMSE) in BPM
- Pearson correlation coefficient (r)
- Signal-to-Noise Ratio (SNR) in dB
"""

from typing import Dict, Union
import numpy as np


def compute_mae(predictions: np.ndarray, references: np.ndarray) -> float:
    """Mean Absolute Error (MAE) in BPM."""
    preds = np.asarray(predictions, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)
    return float(np.mean(np.abs(preds - refs)))


def compute_rmse(predictions: np.ndarray, references: np.ndarray) -> float:
    """Root Mean Square Error (RMSE) in BPM."""
    preds = np.asarray(predictions, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)
    return float(np.sqrt(np.mean((preds - refs) ** 2)))


def compute_pearson_r(predictions: np.ndarray, references: np.ndarray) -> float:
    """Pearson correlation coefficient (r)."""
    preds = np.asarray(predictions, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)
    if len(preds) < 2:
        return 1.0

    p_std = np.std(preds)
    r_std = np.std(refs)
    if p_std < 1e-8 or r_std < 1e-8:
        return 0.0

    corr_matrix = np.corrcoef(preds, refs)
    return float(corr_matrix[0, 1])


def compute_snr_db(
    psd: np.ndarray,
    freqs: np.ndarray,
    gt_hr_bpm: float,
    delta_hz: float = 0.15,
) -> float:
    """Calculate Signal-to-Noise Ratio (SNR) in dB for rPPG power spectrum.

    Measures pulse harmonic energy relative to background in-band noise energy.
    """
    gt_freq = gt_hr_bpm / 60.0
    first_harmonic = gt_freq
    second_harmonic = 2.0 * gt_freq

    # Mask for first and second harmonics
    mask_h1 = (freqs >= first_harmonic - delta_hz) & (freqs <= first_harmonic + delta_hz)
    mask_h2 = (freqs >= second_harmonic - delta_hz) & (freqs <= second_harmonic + delta_hz)
    signal_mask = mask_h1 | mask_h2

    # In-band physiological mask [0.7, 3.0] Hz
    band_mask = (freqs >= 0.7) & (freqs <= 3.0)
    noise_mask = band_mask & (~signal_mask)

    signal_energy = np.sum(psd[signal_mask])
    noise_energy = np.sum(psd[noise_mask])

    if noise_energy < 1e-12:
        return 20.0
    if signal_energy < 1e-12:
        return -20.0

    return float(10.0 * np.log10(signal_energy / noise_energy))


def compute_all_metrics(
    predictions: np.ndarray,
    references: np.ndarray,
) -> Dict[str, float]:
    """Compute full suite of official competition metrics."""
    preds = np.asarray(predictions, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)

    mae = compute_mae(preds, refs)
    medae = float(np.median(np.abs(preds - refs)))
    rmse = compute_rmse(preds, refs)
    r = compute_pearson_r(preds, refs)

    diffs = preds - refs
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    loa_upper = float(bias + 1.96 * sd)
    loa_lower = float(bias - 1.96 * sd)

    return {
        "mae": round(mae, 3),
        "medae": round(medae, 3),
        "rmse": round(rmse, 3),
        "pearson_r": round(r, 4),
        "bias": round(bias, 3),
        "sd": round(sd, 3),
        "loa_upper": round(loa_upper, 3),
        "loa_lower": round(loa_lower, 3),
    }
