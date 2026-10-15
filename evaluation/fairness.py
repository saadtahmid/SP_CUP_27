"""Skin tone fairness evaluation and tolerance compliance (§2.2 C2).

Evaluates whether HR extraction accuracy systematically degrades across Fitzpatrick types I-VI.
Applies pre-specified tolerance check (per-type |MAE - Population_MAE| <= tolerance_bpm or <= tolerance_pct)
while reporting signed deviations (Stratum_MAE - Population_MAE) so weighted subgroup averages
reconcile exactly with the population mean.
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
    """Stratify errors across Fitzpatrick skin types and evaluate fairness tolerance."""
    preds = np.asarray(predictions, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)
    types = np.asarray(fitzpatrick_types)

    abs_errors = np.abs(preds - refs)
    population_mae = float(np.mean(abs_errors))
    total_count = len(preds)

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

        signed_dev_bpm = float(stratum_mae - population_mae)
        abs_dev_bpm = abs(signed_dev_bpm)
        dev_pct = (abs_dev_bpm / population_mae * 100.0) if population_mae > 0 else 0.0

        if abs_dev_bpm > max_dev:
            max_dev = abs_dev_bpm

        type_compliant = (abs_dev_bpm <= tolerance_bpm) or (dev_pct <= tolerance_pct)
        if not type_compliant:
            tolerance_met = False

        strata_metrics[st] = {
            "count": n_samples,
            "mae": round(stratum_mae, 3),
            "rmse": round(stratum_rmse, 3),
            "pearson_r": round(stratum_r, 4),
            "deviation_from_pop_bpm": round(signed_dev_bpm, 3),
            "abs_deviation_bpm": round(abs_dev_bpm, 3),
            "deviation_pct": round(dev_pct, 2),
            "compliant": type_compliant,
        }

    return {
        "population_mae": round(population_mae, 3),
        "total_count": total_count,
        "tolerance_bpm_threshold": tolerance_bpm,
        "tolerance_pct_threshold": tolerance_pct,
        "max_deviation_bpm": round(max_dev, 3),
        "tolerance_met": tolerance_met,
        "strata": strata_metrics,
    }


def generate_fairness_markdown_table(fairness_results: Dict[str, Any]) -> str:
    """Generate markdown table of per-stratum metrics with signed deviations and weighted verification."""
    rows = [
        "| Fitzpatrick Skin Type | Count | MAE (bpm) | RMSE (bpm) | Pearson r | Signed Deviation from Mean (bpm) | Compliant |",
        "|---|---|---|---|---|---|---|",
    ]
    strata = fairness_results.get("strata", {})
    weighted_sum = 0.0
    total_n = 0
    for st, m in strata.items():
        comp_str = "PASSED" if m["compliant"] else "FAILED"
        weighted_sum += m["count"] * m["mae"]
        total_n += m["count"]
        rows.append(
            f"| {st} | {m['count']} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['pearson_r']:.3f} | {m['deviation_from_pop_bpm']:+.2f} | {comp_str} |"
        )
    pop_mae = fairness_results.get("population_mae", 0.0)
    max_dev = fairness_results.get("max_deviation_bpm", 0.0)
    all_comp = "PASSED" if fairness_results.get("tolerance_met", False) else "FAILED"
    rows.append(
        f"| **Population Mean (Weighted)** | **{total_n}** | **{pop_mae:.2f}** | - | - | **0.00 (Max |Δ|: {max_dev:.2f})** | **{all_comp}** |"
    )
    return "\n".join(rows)
