"""
worked_example_cfd.py — the procedure of the paper (src/procedure.py)
applied to the CFD dispersion case as a practitioner would: 20 CFD runs
(the Latin hypercube of seed 1000), M = 1e5 Monte Carlo trials through the
chosen surrogate, the surrogate contribution from LOO residuals, an
uncertainty budget, and a comparison with the reference built from 800 CFD
runs (which a practitioner would not have).
Writes results/worked_example_cfd.json and results/journal/tab_worked.tex.
"""

import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from framework_core import lhs
from cfd_benchmark import CFDDispersion
from procedure import evaluate, budget_table

b = CFDDispersion()
np.random.seed(1000)
X = b.ppf(lhs(b.dim, 20))
y = b.evaluate(X)                                     # 20 cached CFD runs
lo, hi = np.array(b.bounds).T
sample = lambda n, rng: lo + rng.uniform(0, 1, (n, b.dim)) * (hi - lo)
res = evaluate(X, y, sample, bounds=b.bounds, positive=True, pos_in=b.pos_in,
               threshold=b.threshold, M=100_000, seed=0)
names = ["U_ref", "z0", "H_b", "Sc_t", "x_r", "Q"]
print(budget_table(res, names, "mg/m3"))

yr = b.evaluate(b.reference_sets()["ref"])
ref = {"mean": float(yr.mean()), "sd": float(yr.std()), "q025": float(np.quantile(yr, 0.025)),
       "q975": float(np.quantile(yr, 0.975)), "q95": float(np.quantile(yr, 0.95)),
       "p_exceed": float(np.mean(yr > b.threshold))}
# standard uncertainty of the reference (cluster bootstrap, cfd_reference_stats.py)
u_ref = json.load(open(os.path.join(ROOT, "results", "cfd_reference.json")))["u_reference"]
u_ref = {k: u_ref[k] for k in ("mean", "sd", "q025", "q975", "q95")} | {"p_exceed": u_ref["p_idlh"]}
res["reference"] = ref
# share of the reference distribution inside the reported coverage interval
w = res["with_surrogate_contribution"]
res["reference_share_in_interval"] = float(np.mean((yr >= w["q025"]) & (yr <= w["q975"])))
res["u_reference"] = u_ref
res["input_names"] = names
json.dump(res, open(os.path.join(ROOT, "results", "worked_example_cfd.json"), "w"), indent=1)

w, p, j = res["with_surrogate_contribution"], res["surrogate_as_exact"], res["jackknife_95"]
f = lambda v: f"{v:.0f}" if abs(v) >= 10 else f"{v:.3f}"
rows = [("Estimate $y$ (mean)", "mean"), ("Standard uncertainty $u(y)$", "sd"),
        ("95\\% coverage interval, lower", "q025"), ("95\\% coverage interval, upper", "q975"),
        ("95th percentile $y_{0.95}$", "q95"), ("$P(C>\\mathrm{IDLH})$", "p_exceed")]
tex = []
for label, k in rows:
    jk = f"[{f(j[k][0])}, {f(j[k][1])}]" if k in j else "--"
    fu = f"{u_ref[k]:.3f}" if k == "p_exceed" else f"{u_ref[k]:.{0 if u_ref[k] >= 10 else 1}f}"
    rf = f"{f(ref[k])} ({fu})"
    tex.append(f"{label} & {f(p[k])} & {f(w[k])} & {jk} & {rf} \\\\")
os.makedirs(os.path.join(ROOT, "results", "journal"), exist_ok=True)
open(os.path.join(ROOT, "results", "journal", "tab_worked.tex"), "w").write("\n".join(tex) + "\n")
print("reference:", {k: round(v, 3) for k, v in ref.items()})
