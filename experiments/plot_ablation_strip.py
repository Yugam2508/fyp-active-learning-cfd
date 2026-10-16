"""
plot_ablation_strip.py — per-seed RMSE for every paired ablation arm.

Reads results/results_<arm>.json (no reruns) and writes
figures/fig_ablation_strip.{pdf,png}: one row per arm, one dot per seed,
log RMSE axis, median tick, success threshold (RMSE < 200) as a dashed line.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

ARMS = [  # (file key, label) — top to bottom
    ("hard",   "Full framework\n(UEigV, hard LCS)"),
    ("varacq", "Variance acquisition\n(hard LCS)"),
    ("soft",   "Soft CV weights\n(UEigV)"),
    ("nopce",  "No PCE members\n(UEigV, hard LCS)"),
    ("gpvar",  "Single GP\n(posterior std)"),
]
THRESH = 200.0
INK, MUTED, DOT = "#0b0b0b", "#52514e", "#2a78d6"

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.edgecolor": MUTED, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.4, 2.4))
rs = np.random.RandomState(0)
for i, (key, _) in enumerate(ARMS):
    r = np.array([t["rmse"] for t in json.load(open(os.path.join(RES, f"results_{key}.json")))])
    y = i + rs.uniform(-0.18, 0.18, len(r))
    ax.scatter(r, y, s=9, color=DOT, alpha=0.75, linewidths=0)
    med = np.median(r)
    ax.plot([med, med], [i - 0.32, i + 0.32], color=INK, lw=1.2)
    ok = int((r < THRESH).sum())
    ax.text(1.02, i, f"{ok}/{len(r)}", transform=ax.get_yaxis_transform(),
            va="center", ha="left", color=INK, fontsize=7.5)

ax.axvline(THRESH, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(THRESH * 1.15, len(ARMS) - 0.45, "success\nthreshold", color=MUTED,
        fontsize=6.5, va="top")
ax.text(1.02, -0.85, "succ.", transform=ax.get_yaxis_transform(),
        ha="left", va="center", color=MUTED, fontsize=6.5)
ax.set_xscale("log")
ax.set_yticks(range(len(ARMS)))
ax.set_yticklabels([lab for _, lab in ARMS], fontsize=7, color=INK)
ax.set_ylim(len(ARMS) - 0.5, -1.1)
ax.set_xlabel("Test RMSE (log scale; one dot per seed, bar = median)", color=INK)
ax.tick_params(axis="both", colors=MUTED, length=2)
ax.grid(axis="x", which="major", color="#e4e3df", lw=0.5)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_ablation_strip.pdf"))
fig.savefig(os.path.join(FIG, "fig_ablation_strip.png"), dpi=200)
print("wrote", os.path.join(FIG, "fig_ablation_strip.pdf"))
