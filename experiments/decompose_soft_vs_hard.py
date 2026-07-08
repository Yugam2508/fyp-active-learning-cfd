"""
decompose_soft_vs_hard.py
=========================
The paired campaign shows soft CV-weighting beats hard LCS rejection, but the
two arms differ in THREE ways simultaneously:
   (1) model manager      permanent rejection      vs  reversible soft weights
   (2) final predictor    unweighted mean of       vs  CV-weighted average
                          surviving models             over all K models
   (3) acquisition        UEigV over survivors     vs  UEigV over unmuted set

This script isolates (2) from (1)+(3) by holding the TRAJECTORY fixed:

  A) run hard-LCS active learning exactly as in campaign.py            -> rmse_A
  B) take A's FINAL DATASET, refit all K models on it, apply the CV
     inverse-error weights, and predict                                -> rmse_B

  rmse_A vs rmse_B  isolates the PREDICTOR (same data, same samples).
  rmse_B vs rmse_C  (soft, from campaign.py) isolates the TRAJECTORY
                    (same predictor, different sample placement).

We also record when hard's active committee collapses to a single model,
because at that point framework_core.ActiveLearner.ueigv() returns all-ones
and argmax selects candidate 0 -- i.e. hard rejection silently degenerates to
UNIFORM RANDOM SAMPLING for the rest of the budget.

Usage: python decompose_soft_vs_hard.py <start_trial> <n_trials>
Appends to results_decompose.json (resumable).
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
RESULTS_DIR = _os.path.join(_ROOT, "results"); _os.makedirs(RESULTS_DIR, exist_ok=True)
FIGURES_DIR = _os.path.join(_ROOT, "figures"); _os.makedirs(FIGURES_DIR, exist_ok=True)

import sys, json, os, time
import numpy as np
try:
    import torch
except Exception:
    torch = None

from framework_core import (ToyProblem, ActiveLearner, GPSurrogate,
                            PCESurrogate, RFSurrogate, SVRSurrogate, DNNSurrogate)
from uq_extension import SoftLCS, make_bound_penalty

start, n = int(sys.argv[1]), int(sys.argv[2])
OUT = _os.path.join(RESULTS_DIR, f"results_decompose.json")
MAX_STEPS = 15

problem = ToyProblem(3, "physics")
domain = problem.bounds


def fixed_sampler(n_pts, rs):
    b = np.array(domain, float)
    return b[:, 0] + rs.uniform(0, 1, (n_pts, 3)) * (b[:, 1] - b[:, 0])


X_test = fixed_sampler(5000, np.random.RandomState(7))   # identical to campaign.py
y_test = problem.evaluate(X_test)


def build_committee(dim):
    """Same nine members as ActiveLearner._build_models(), fresh/untrained."""
    return [GPSurrogate("RBF"), GPSurrogate("Matern"),
            PCESurrogate(2), PCESurrogate(3),
            RFSurrogate(50), RFSurrogate(100),
            SVRSurrogate("rbf"), DNNSurrogate(dim, 64), DNNSurrogate(dim, 128)]


def rmse(pred):
    return float(np.sqrt(np.mean((y_test - pred) ** 2)))


def run_trial(seed):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)

    # ---- (A) hard-LCS active learning, instrumented -----------------------
    a = ActiveLearner(ToyProblem(3, "physics"), domain)
    a.train()
    n_active, collapse_iter = [], None
    for t in range(MAX_STEPS):
        k = len([m for m in a.models if m not in a.rejected])
        n_active.append(k)
        if k <= 1 and collapse_iter is None:
            collapse_iter = t                       # queries from here are random
        a.step(); a.train()
    n_active.append(len([m for m in a.models if m not in a.rejected]))

    survivors = [m for m in a.models if m not in a.rejected]
    rmse_A = rmse(np.mean([m.predict(X_test) for m in survivors], axis=0))

    # ---- (B) SAME final dataset, swap in the soft predictor ---------------
    X, y = a.X.copy(), a.y.copy()
    fresh = build_committee(3)
    for m in fresh:
        m.fit(X, y)
    lcs = SoftLCS(n_folds=3, physics_penalty=make_bound_penalty(lower=-1.0))
    X_probe = np.random.uniform(np.array(domain)[:, 0], np.array(domain)[:, 1],
                                size=(200, 3))
    w, _ = lcs.weights(fresh, X, y, X_probe=X_probe)
    rmse_B = rmse(np.average([m.predict(X_test) for m in fresh], axis=0,
                             weights=w))

    return {"seed": seed, "rmse_A_hard_reported": rmse_A,
            "rmse_B_hard_traj_soft_predictor": rmse_B,
            "n_survivors": len(survivors),
            "survivor_names": [m.name for m in survivors],
            "collapse_iter": collapse_iter,
            "random_queries_after_collapse":
                (MAX_STEPS - collapse_iter) if collapse_iter is not None else 0,
            "n_active_trace": n_active}


results = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {r["seed"] for r in results}
for t in range(start, start + n):
    seed = 1000 + t
    if seed in done:
        print(f"trial {t} (seed {seed}) already done, skipping"); continue
    t0 = time.time()
    r = run_trial(seed)
    r["runtime_s"] = round(time.time() - t0, 1)
    results.append(r)
    json.dump(results, open(OUT, "w"))
    print(f"trial {t:2d} seed {seed} | A(hard)={r['rmse_A_hard_reported']:9.3f} "
          f"| B(hard traj + soft pred)={r['rmse_B_hard_traj_soft_predictor']:9.3f} "
          f"| survivors={r['n_survivors']} | random queries after collapse="
          f"{r['random_queries_after_collapse']} | {r['runtime_s']}s", flush=True)
print(f"total recorded: {len(results)}")
