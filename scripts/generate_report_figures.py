"""Generate IEEE report figures from repository artifacts."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Figure 1: Bland-Altman (subject20 submission bundle) ----------
ref = pd.read_csv("submission/Round1_Baseline/reference_signal.csv")
est = pd.read_csv("submission/Round1_Baseline/processed_output.csv")
n = min(len(ref), len(est))
gt, pr = ref["ground_truth_hr_bpm"].values[:n], est["estimated_hr_bpm"].values[:n]
mean_h, diff = (gt + pr) / 2, pr - gt
bias, sd = diff.mean(), diff.std(ddof=1)

fig, ax = plt.subplots(figsize=(3.5, 2.8), dpi=300)
ax.scatter(mean_h, diff, s=8, c="#0f3d5e", alpha=0.55, edgecolors="none")
ax.axhline(bias, color="crimson", lw=1.2, label=f"Bias = {bias:+.2f} BPM")
for k, ls in [(1.96, "--"), (-1.96, "--")]:
    ax.axhline(bias + k * sd, color="gray", lw=0.9, ls=ls,
               label=f"95% LoA = {bias-1.96*sd:+.2f} / {bias+1.96*sd:+.2f}")
ax.set_xlabel("Mean of reference and estimated HR (BPM)")
ax.set_ylabel("Estimated − Reference (BPM)")
ax.legend(fontsize=6, loc="best")
fig.tight_layout()
fig.savefig("fig_bland_altman_subject20.png", bbox_inches="tight")

# ---------- Figure 2: C1 lux-vs-MAE degradation curve ----------
c1 = pd.read_csv("artifacts/constraints/c1_degradation_curve.csv")
fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
ax.errorbar(c1.lux, c1.hybrid_mae_bpm, yerr=c1.hybrid_sd_bpm, marker="o",
            ms=4, lw=1.4, capsize=3, color="#0f3d5e", label="Hybrid v4")
ax.errorbar(c1.lux, c1.chrom_mae_bpm, yerr=c1.chrom_sd_bpm, marker="s",
            ms=4, lw=1.4, capsize=3, color="#c0392b", label="CHROM + CZT")
ax.axvline(50, color="gray", ls=":", lw=1)
ax.text(52, ax.get_ylim()[1]*0.9, "50 lux spec", fontsize=7, color="gray")
ax.set_xscale("log")
ax.set_xticks([10, 25, 50, 300])
ax.set_xticklabels(["10", "25", "50", "300"])
ax.set_xlabel("Illuminance (lux, log scale)")
ax.set_ylabel("MAE (BPM)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("fig_c1_degradation.png", bbox_inches="tight")
print("Figures written: fig_bland_altman_subject20.png, fig_c1_degradation.png")
