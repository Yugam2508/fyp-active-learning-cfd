"""
fig_cfd_case.py — figure of the CFD dispersion case (figures/journal/fig_cfd_case.pdf):
(a) streamwise velocity and concentration field for the centre of the input
box, (b) concentration at 1.5 m against distance from the building for
eight cached reference runs. Usage: python experiments/fig_cfd_case.py
"""

import os, sys, re, json, glob, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import cfd_dispersion as cfd

FLOW = (3.5, 0.155, 10.5, 0.9)
with tempfile.TemporaryDirectory(dir=os.environ.get("CFD_WORKDIR")) as tmp:
    d = os.path.join(tmp, "case")
    res = cfd.run_case(*FLOW, workdir=d, keep=True)
    t = str(res["iterations"])
    C = cfd.cell_centres(d)
    s = cfd.read_internal_scalar(os.path.join(d, t, "T"))
    txt = open(os.path.join(d, t, "U")).read()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*(\d+)\s*\(", txt)
    U = np.array([list(map(float, v.split())) for v in
                  re.findall(r"\(([^()]+)\)", txt[m.end():])[:int(m.group(1))]])

fig = plt.figure(figsize=(7.2, 4.4))
ax1 = fig.add_axes([0.08, 0.56, 0.84, 0.38])
ax2 = fig.add_axes([0.10, 0.09, 0.36, 0.34])
ax3 = fig.add_axes([0.57, 0.09, 0.38, 0.34])
sel = (C[:, 2] < 45) & (C[:, 0] > -40) & (C[:, 0] < 160)
tri = mtri.Triangulation(C[sel, 0], C[sel, 2])
H = FLOW[2]
xc, zc = tri.x[tri.triangles].mean(1), tri.y[tri.triangles].mean(1)
tri.set_mask((xc > 0) & (xc < 10) & (zc < H))
conc = 1e3 * np.maximum(s[sel], 1e-12)              # mg m^-3 for Q = 1 g/s/m
cs = ax1.tricontourf(tri, np.log10(conc), levels=np.linspace(-1, 2.5, 15), cmap="viridis", extend="both")
ax1.tricontour(tri, U[sel, 0], levels=[0.0], colors="w", linewidths=0.8)
ax1.add_patch(plt.Rectangle((0, 0), 10, H, color="0.3"))
ax1.plot([-20], [0.5], "r*", ms=7)
ax1.set_xlim(-40, 160); ax1.set_ylim(0, 45); ax1.set_aspect("equal")
ax1.set_xlabel("x (m)", fontsize=8); ax1.set_ylabel("z (m)", fontsize=8)
ax1.tick_params(labelsize=7)
cb = fig.colorbar(cs, ax=ax1, fraction=0.025, pad=0.01)
cb.set_label("log$_{10}$ C (mg m$^{-3}$), Q = 1 g s$^{-1}$ m$^{-1}$", fontsize=7); cb.ax.tick_params(labelsize=6)
ax1.set_title(f"(a) $U_{{ref}}$ = {FLOW[0]} m/s, $z_0$ = {FLOW[1]} m, $H_b$ = {H} m, "
              f"$Sc_t$ = {FLOW[3]}; white: $U_x$ = 0 (recirculation)", fontsize=7.5)

recs = sorted((json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results", "cfd_runs", "*.json"))),
              key=lambda r: r["key"])[:8]
for r in recs:
    ax2.semilogy(r["x"], 1e3 * np.array(r["s"]), lw=0.9)
ax2.set_xlim(0, 125); ax2.axvspan(5, 120, color="0.9", zorder=0)
ax2.set_xlabel("distance from leeward face $x_r$ (m)", fontsize=8)
ax2.set_ylabel("C at 1.5 m (mg m$^{-3}$), Q = 1", fontsize=8)
ax2.set_title("(b) eight runs; grey: range of $x_r$", fontsize=7.5); ax2.tick_params(labelsize=7)
from matplotlib.ticker import FuncFormatter, NullFormatter
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax2.yaxis.set_minor_formatter(NullFormatter())

allr = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results", "cfd_runs", "*.json"))]
it = np.array([r["iterations"] for r in allr]); conv = np.array([r["converged"] for r in allr])
ax3.hist(it, bins=30, color="C0")
ax3.set_xlabel("SIMPLE iterations", fontsize=8); ax3.set_ylabel("runs", fontsize=8)
ax3.set_title(f"(c) {len(allr)} runs, {conv.mean():.0%} met the residual criteria", fontsize=7.5)
ax3.tick_params(labelsize=7)
os.makedirs(os.path.join(ROOT, "figures", "journal"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "figures", "journal", "fig_cfd_case.pdf"))
print("runs", len(allr), "converged", conv.mean(), "iterations median", np.median(it))
