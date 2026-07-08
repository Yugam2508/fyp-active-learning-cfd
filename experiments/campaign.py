"""
campaign.py — 30-trial hard-vs-soft comparison + p-box coverage study.

Usage:  python3 campaign.py <method: hard|soft> <start_trial> <n_trials>
Results append to results_<method>.json (resumable; trials keyed by seed).

Per trial we record:
  rmse            weighted-ensemble (soft) / surviving-mean (hard) RMSE on 5k test
  exc_*           exceedance interval [lo,up], central, truth, covered (bool)
  mean_covered    epistemic band on the output mean brackets the true mean
  pbox_coverage   fraction of CDF grid where F_low <= F_true <= F_up
  q_relerr        relative errors of weighted-mixture q05/q50/q95 vs truth
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

from framework_core import ToyProblem, ActiveLearner
from uq_extension import (SoftActiveLearner, UQAnalyzer,
                          uniform_sampler, make_bound_penalty)

method, start, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
OUT = _os.path.join(RESULTS_DIR, f"results_{method}.json")
MAX_STEPS = 15

problem = ToyProblem(3, "physics")
domain = problem.bounds
sampler = uniform_sampler(domain)

# ---- fixed ground truth (same for every trial) --------------------------
def fixed_sampler(n_pts, rs):
    b = np.array(domain, float)
    return b[:, 0] + rs.uniform(0, 1, (n_pts, 3)) * (b[:, 1] - b[:, 0])

X_test = fixed_sampler(5000, np.random.RandomState(7))
y_test = problem.evaluate(X_test)
X_ref  = fixed_sampler(200_000, np.random.RandomState(8))
y_ref  = problem.evaluate(X_ref)
y_crit = float(np.quantile(y_ref, 0.95))
p_true = float(np.mean(y_ref > y_crit))
mean_true = float(y_ref.mean())
q_true = {q: float(np.quantile(y_ref, q)) for q in (0.05, 0.5, 0.95)}

# common CDF grid for coverage
GRID = np.linspace(np.quantile(y_ref, 0.001), np.quantile(y_ref, 0.999), 120)
F_true = np.array([np.mean(y_ref <= g) for g in GRID])

# fixed MC input sample for propagation (shared across trials for fairness)
X_mc = fixed_sampler(30_000, np.random.RandomState(9))


def run_trial(seed):
    np.random.seed(seed)
    if torch is not None: torch.manual_seed(seed)
    t0 = time.time()
    if method == "hard":
        a = ActiveLearner(ToyProblem(3, "physics"), domain)
        a.train()
        for _ in range(MAX_STEPS):
            a.step(); a.train()
        models = [m for m in a.models if m not in a.rejected]
        weights = None
    else:
        a = SoftActiveLearner(ActiveLearner, ToyProblem(3, "physics"), domain,
                              n_folds=3, update_every=3,
                              physics_penalty=make_bound_penalty(lower=-1.0))
        a.train()
        for i in range(MAX_STEPS):
            a.step(); a.train(force_weight_update=(i == MAX_STEPS - 1))
        models, weights = a.models, a.w

    uq = UQAnalyzer(models, weights=weights, noise_std=0.1)
    P = uq._preds(X_mc)
    mu = np.average(P, axis=0, weights=uq.w)
    rmse_pred = np.average(uq._preds(X_test), axis=0, weights=uq.w)
    rmse = float(np.sqrt(np.mean((y_test - rmse_pred) ** 2)))

    exc = uq.exceedance(P, y_crit)
    mean_band = (float(P.mean(axis=1).min()), float(P.mean(axis=1).max()))
    pb = uq.pbox(P, y_grid=GRID)
    pbox_cov = float(np.mean((pb["F_lower"] - 1e-9 <= F_true)
                             & (F_true <= pb["F_upper"] + 1e-9)))
    q_err = {str(q): float(abs(np.quantile(mu, q) - q_true[q])
                           / (abs(q_true[q]) + 1e-12))
             for q in (0.05, 0.5, 0.95)}

    return {
        "seed": seed, "method": method, "rmse": rmse,
        "n_models_final": len(models),
        "exc_lo": exc["lower"], "exc_up": exc["upper"],
        "exc_central": exc["central"], "exc_true": p_true,
        "exc_covered": bool(exc["lower"] - 1e-9 <= p_true <= exc["upper"] + 1e-9),
        "mean_lo": mean_band[0], "mean_up": mean_band[1],
        "mean_true": mean_true,
        "mean_covered": bool(mean_band[0] <= mean_true <= mean_band[1]),
        "pbox_coverage": pbox_cov,
        "q_relerr": q_err,
        "runtime_s": round(time.time() - t0, 1),
    }


results = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {r["seed"] for r in results}
for t in range(start, start + n):
    seed = 1000 + t
    if seed in done:
        print(f"trial {t} (seed {seed}) already done, skipping"); continue
    r = run_trial(seed)
    results.append(r)
    json.dump(results, open(OUT, "w"))
    print(f"[{method}] trial {t:2d} seed {seed} | RMSE={r['rmse']:9.3f} | "
          f"exc_cov={r['exc_covered']} | pbox={r['pbox_coverage']:.2f} | "
          f"{r['runtime_s']}s", flush=True)
print(f"total recorded: {len(results)}")
