"""
cfd_reference_stats.py — the reference distribution of the CFD case and its
own Monte Carlo uncertainty. The 40 000 reference points come from 800 CFD
runs (50 (x_r, Q) draws each), so the standard uncertainty of each reference
quantity is estimated by a cluster bootstrap over the runs (2000 resamples,
RandomState(0)). Also reports the convergence statistics of the cached runs.
Writes results/cfd_reference.json.
"""

import os, sys, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from cfd_benchmark import CFDDispersion, N_FLOW, K_PAIRS

b = CFDDispersion()
y = b.evaluate(b.reference_sets()["ref"]).reshape(N_FLOW, K_PAIRS)


def q(v):
    v = v.ravel()
    return {"mean": float(v.mean()), "sd": float(v.std()),
            "q025": float(np.quantile(v, 0.025)), "q975": float(np.quantile(v, 0.975)),
            "q95": float(np.quantile(v, 0.95)), "p_idlh": float(np.mean(v > b.threshold)),
            "min": float(v.min()), "max": float(v.max())}


est = q(y)
rs = np.random.RandomState(0)
boot = [q(y[rs.randint(0, N_FLOW, N_FLOW)]) for _ in range(2000)]
u = {k: float(np.std([bb[k] for bb in boot])) for k in ("mean", "sd", "q025", "q975", "q95", "p_idlh")}
recs = [json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results", "cfd_runs", "*.json"))]
it = np.array([r["iterations"] for r in recs]); conv = np.array([r["converged"] for r in recs])
out = {"reference": est, "u_reference": u, "n_runs_cached": len(recs),
       "share_converged": float(conv.mean()), "iterations_median": float(np.median(it)),
       "iterations_q90": float(np.quantile(it[conv], 0.9))}
json.dump(out, open(os.path.join(ROOT, "results", "cfd_reference.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
