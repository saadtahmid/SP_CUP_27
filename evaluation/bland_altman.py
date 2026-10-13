"""Bland-Altman analysis and publication-quality plot generation.

Complies with 2027 IEEE Signal Processing Cup evaluation requirements (§2.3 Task 1)
and Altmann & Bland (1983) statistical standards.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bland_altman_analysis(
    predictions: np.ndarray,
    references: np.ndarray,
) -> Dict[str, float]:
    """Calculate Bland-Altman mean bias, SD, and 95% Limits of Agreement (LOA)."""
    preds = np.asarray(predictions, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)
    diffs = preds - refs

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    upper_loa = float(mean_diff + 1.96 * sd_diff)
    lower_loa = float(mean_diff - 1.96 * sd_diff)

    return {
        "mean_bias": round(mean_diff, 3),
        "std_diff": round(sd_diff, 3),
        "upper_loa": round(upper_loa, 3),
        "lower_loa": round(lower_loa, 3),
    }


def plot_bland_altman(
    predictions: np.ndarray,
    references: np.ndarray,
    output_path: str,
    title: str = "Bland-Altman Agreement: Estimated HR vs Reference HR",
    unit: str = "BPM",
) -> Dict[str, float]:
    """Generate and save publication-grade Bland-Altman plot."""
    preds = np.asarray(predictions, dtype=np.float64)
    refs = np.asarray(references, dtype=np.float64)

    means = 0.5 * (preds + refs)
    diffs = preds - refs

    stats = bland_altman_analysis(preds, refs)
    bias = stats["mean_bias"]
    upper = stats["upper_loa"]
    lower = stats["lower_loa"]

    plt.figure(figsize=(7, 5), dpi=300)
    plt.scatter(means, diffs, alpha=0.7, color="#1f77b4", edgecolor="k", s=40)

    # Bias line (dashed)
    plt.axhline(bias, color="red", linestyle="--", linewidth=1.5, label=f"Mean Bias: {bias:+.2f} {unit}")
    # 95% Limits of Agreement
    plt.axhline(upper, color="green", linestyle=":", linewidth=1.5, label=f"+1.96 SD: {upper:+.2f} {unit}")
    plt.axhline(lower, color="green", linestyle=":", linewidth=1.5, label=f"-1.96 SD: {lower:+.2f} {unit}")

    # Shade the 95% agreement region
    plt.axhspan(lower, upper, alpha=0.08, color="green")

    plt.xlabel(f"Mean of Reference and Estimate ({unit})", fontsize=11)
    plt.ylabel(f"Difference (Estimate - Reference) ({unit})", fontsize=11)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper right", frameon=True, fontsize=9)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300)
    plt.close()

    return stats
