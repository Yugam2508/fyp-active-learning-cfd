"""
cfd_reference2.py — research plan T2.6: is the 800-run CFD reference good
enough?

A second, independent reference set (800 flow states, RandomState(18), 50
(x_r, Q) pairs each; runs precomputed by cfd_precompute.py --ref2) gives
  - the reference quantities from set 2 and from both sets (1600 runs),
    with standard uncertainties by a cluster bootstrap over runs (2000
    resamples, RandomState(0)), and
  - the validation errors (y95, P(C > IDLH)) of the passive single
    surrogates of the main study (gp_m52, gp_log, gp_quadlog, spce_log; the
    same fits as campaign_generalisation.py) against the original and the
    combined reference, to check that the conclusions do not depend on
    which reference is used.
Writes results/cfd_reference2.json.
"""

import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
from framework_core import lhs
from cfd_benchmark import CFDDispersion, N_FLOW, K_PAIRS
from surrogates import GP, SparsePCE

b = CFDDispersion()
Y1 = b.evaluate(b.reference_sets(8)["ref"]).reshape(N_FLOW, K_PAIRS)
Y2 = b.evaluate(b.reference_sets(18)["ref"]).reshape(N_FLOW, K_PAIRS)


def q(v):
    v = v.ravel()
    return {"mean": float(v.mean()), "sd": float(v.std()), "q95": float(np.quantile(v, 0.95)),
            "p_idlh": float(np.mean(v > b.threshold))}


def with_u(Y):
    rs = np.random.RandomState(0)
    boot = [q(Y[rs.randint(0, len(Y), len(Y))]) for _ in range(2000)]
    est = q(Y)
    return {"est": est, "u": {k: float(np.std([x[k] for x in boot])) for k in est}}


refs = {"set1": with_u(Y1), "set2": with_u(Y2), "both": with_u(np.vstack([Y1, Y2]))}
X_mc = b.ppf(np.random.RandomState(9).uniform(0, 1, (30_000, b.dim)))
errs = {}
for arm in ("gp_m52", "gp_log", "gp_quadlog", "spce_log"):
    e = {"set1_q95": [], "both_q95": [], "set1_p": [], "both_p": []}
    for seed in range(1000, 1030):
        np.random.seed(seed)
        X = b.ppf(lhs(b.dim, 20))
        y = b.evaluate(X)
        s = (GP(arm[3:], b.dim, b.pos_in) if arm.startswith("gp_")
             else SparsePCE(b.dim, b.bounds, log=True, pos_in=b.pos_in)).fit(X, y)
        mu = s.predict(X_mc)
        q95, p = np.quantile(mu, 0.95), np.mean(mu > b.threshold)
        for tag in ("set1", "both"):
            r = refs[tag]["est"]
            e[f"{tag}_q95"].append(abs(q95 - r["q95"]) / r["q95"])
            e[f"{tag}_p"].append(abs(p - r["p_idlh"]))
    errs[arm] = {k: float(np.median(v)) for k, v in e.items()}
    print(arm, {k: round(v, 4) for k, v in errs[arm].items()}, flush=True)
json.dump({"references": refs, "median_errors": errs},
          open(os.path.join(ROOT, "results", "cfd_reference2.json"), "w"), indent=1)
print(json.dumps(refs, indent=1))
