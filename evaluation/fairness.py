"""Skin tone fairness evaluation and tolerance compliance (§2.2 C2).

Evaluates whether HR extraction accuracy systematically degrades across Fitzpatrick types I-VI.
Applies pre-specified tolerance check (e.g. per-type MAE within +/- 3.0 bpm or +/- 25% of population mean).
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd


def evaluate_skin_tone_fairness(
    predictions: np.ndarray,
    references: np.ndarray,
    fitzpatrick_types: List[str],
    tolerance_bpm: float = 3.0,
    tolerance_pct: float = 25.0,
) -> Dict[str, Any]:
    """Stratify errors across Fitzpatrick skin types and evaluate fairness tolerance.

    Args:
        predictions: Array of estimated HR values in BPM.
        references: Array of ground-truth HR values in BPM.
        fitzpatrick_types: List of Fitzpatrick type strings (e.g. ['Type II', 'Type V', ...]).
        tolerance_bpm: Maximum allowable deviation in BPM from the population mean MAE.
        tolerance_pct: Maximum allowable percentage deviation from population mean MAE.

    Returns:
        Summary dict containing:
        - population_mae: Overall MAE across all subjects
        - strata_metrics: Per-type MAE, RMSE, Pearson r, count
        - tolerance_met: Boolean flag whether all types are within tolerance
        - max_deviation_bpm: Maximum deviation observed across types
    """
    preds = np.asarray(predictions, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)
    types = np.asarray(fitzpatrick_types)

    abs_errors = np.abs(preds - refs)
    population_mae = float(np.mean(abs_errors))

    strata_metrics = {}
    unique_types = sorted(list(set(types)))
    tolerance_met = True
    max_dev = 0.0

    for st in unique_types:
        idx = (types == st)
        n_samples = int(np.sum(idx))
        if n_samples == 0:
            continue

        stratum_preds = preds[idx]
        stratum_refs = refs[idx]
        stratum_mae = float(np.mean(np.abs(stratum_preds - stratum_refs)))
        stratum_rmse = float(np.sqrt(np.mean((stratum_preds - stratum_refs) ** 2)))

        if n_samples > 1 and np.std(stratum_preds) > 1e-6 and np.std(stratum_refs) > 1e-6:
            stratum_r = float(np.corrcoef(stratum_preds, stratum_refs)[0, 1])
        else:
            stratum_r = 1.0

        dev_bpm = abs(stratum_mae - population_mae)
        dev_pct = (dev_bpm / population_mae * 100.0) if population_mae > 0 else 0.0

        if dev_bpm > max_dev:
            max_dev = dev_bpm

        type_compliant = (dev_bpm <= tolerance_bpm) or (dev_pct <= tolerance_pct)
        if not type_compliant:
            tolerance_met = False

        strata_metrics[st] = {
            "count": n_samples,
            "mae": round(stratum_mae, 3),
            "rmse": round(stratum_rmse, 3),
            "pearson_r": round(stratum_r, 4),
            "deviation_from_pop_bpm": round(dev_bpm, 3),
            "deviation_pct": round(dev_pct, 2),
            "compliant": type_compliant,
        }

    return {
        "population_mae": round(population_mae, 3),
        "tolerance_bpm_threshold": tolerance_bpm,
        "tolerance_pct_threshold": tolerance_pct,
        "max_deviation_bpm": round(max_dev, 3),
        "tolerance_met": tolerance_met,
        "strata": strata_metrics,
    }


def generate_fairness_markdown_table(fairness_results: Dict[str, Any]) -> str:
    """Generate markdown table of per-stratum metrics suitable for IEEE technical report."""
    rows = [
        "| Fitzpatrick Skin Type | Count | MAE (bpm) | RMSE (bpm) | Pearson r | Deviation from Mean (bpm) | Compliant |",
        "|---|---|---|---|---|---|---|",
    ]
    strata = fairness_results.get("strata", {})
    for st, m in strata.items():
        comp_str = "PASSED" if m["compliant"] else "FAILED"
        rows.append(
            f"| {st} | {m['count']} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['pearson_r']:.3f} | {m['deviation_from_pop_bpm']:+.2f} | {comp_str} |"
        )
    pop_mae = fairness_results.get("population_mae", 0.0)
    rows.append(f"| **Population Mean** | - | **{pop_mae:.2f}** | - | - | - | - |")
    return "\n".join(rows)
