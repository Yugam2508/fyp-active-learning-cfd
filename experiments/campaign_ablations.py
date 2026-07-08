"""
campaign_ablations.py — the three reviewer-proofing experiments.

Usage:  python campaign_ablations.py <method> <start_trial> <n_trials>
Methods:
  varacq  Same committee + hard LCS, but acquisition = ensemble VARIANCE
          instead of UEigV.  Tests whether the spectral metric adds value
          over plain query-by-committee variance.
  nopce   Full UEigV framework with the PCE models REMOVED from the
          committee.  Tests how much of the win is the PCE-d2 anchor.
  gpvar   Single GP, acquisition = GP posterior std (the field's default
          adaptive baseline).  Same budget, same seeds.

Protocol identical to campaign.py: seeds 1000+trial, physics [-200,200]^3,
N0=5, 15 AL steps, RMSE on the same fixed 5,000-point test set.
Results append to results_<method>.json (resumable).
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

from framework_core import (ToyProblem, ActiveLearner, GPSurrogate)

method, start, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
assert method in ("varacq", "nopce", "gpvar")
OUT = _os.path.join(RESULTS_DIR, f"results_{method}.json")
MAX_STEPS = 15

problem = ToyProblem(3, "physics")
domain = problem.bounds


def fixed_sampler(n_pts, rs):
    b = np.array(domain, float)
    return b[:, 0] + rs.uniform(0, 1, (n_pts, 3)) * (b[:, 1] - b[:, 0])

X_test = fixed_sampler(5000, np.random.RandomState(7))   # same as campaign.py
y_test = problem.evaluate(X_test)


# ── Ablation 1: variance acquisition (same committee, same hard LCS) ─────
class VarAcqLearner(ActiveLearner):
    """Identical to ActiveLearner except the acquisition score is the
    ensemble prediction variance (classic QBC) instead of UEigV."""

    def ueigv(self, grid):                     # name kept so step() is reused
        preds = np.array([m.predict(grid)
                          for m in self.models if m not in self.rejected])
        if preds.shape[0] < 2:
            return np.ones(len(grid))
        return preds.var(axis=0)


# ── Ablation 2: committee without PCE (anchor removed) ───────────────────
class NoPCELearner(ActiveLearner):
    def _build_models(self):
        return [m for m in super()._build_models() if m.family != "PCE"]


def run_committee(learner_cls, seed):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    a = learner_cls(ToyProblem(3, "physics"), domain)
    a.train()
    for _ in range(MAX_STEPS):
        a.step(); a.train()
    models = [m for m in a.models if m not in a.rejected]
    pred = np.mean([m.predict(X_test) for m in models], axis=0)
    rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
    return rmse, len(models), [m.name for m in models]


# ── Ablation 3: single-GP posterior-variance active learning ─────────────
def run_gpvar(seed):
    np.random.seed(seed)
    from framework_core import lhs
    dom = np.array(domain)
    X = dom[:, 0] + lhs(3, 5) * (dom[:, 1] - dom[:, 0])
    y = problem.noisy(X)
    gp = GPSurrogate("RBF")
    for _ in range(MAX_STEPS):
        gp.fit(X, y)
        grid = np.random.uniform(dom[:, 0], dom[:, 1], size=(300, 3))
        _, std = gp.gp.predict(gp.scaler.transform(grid), return_std=True)
        x_new = grid[np.argmax(std)][None, :]
        X = np.vstack([X, x_new])
        y = np.append(y, problem.noisy(x_new))
    gp.fit(X, y)
    rmse = float(np.sqrt(np.mean((y_test - gp.predict(X_test)) ** 2)))
    return rmse, 1, ["GP-RBF"]


DISPATCH = {
    "varacq": lambda s: run_committee(VarAcqLearner, s),
    "nopce":  lambda s: run_committee(NoPCELearner, s),
    "gpvar":  run_gpvar,
}

results = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {r["seed"] for r in results}
for t in range(start, start + n):
    seed = 1000 + t
    if seed in done:
        print(f"trial {t} (seed {seed}) already done, skipping"); continue
    t0 = time.time()
    rmse, k, names = DISPATCH[method](seed)
    results.append({"seed": seed, "method": method, "rmse": rmse,
                    "n_models_final": k, "final_models": names,
                    "runtime_s": round(time.time() - t0, 1)})
    json.dump(results, open(OUT, "w"))
    print(f"[{method}] trial {t:2d} seed {seed} | RMSE={rmse:10.3f} | "
          f"final models={k} | {results[-1]['runtime_s']}s", flush=True)
print(f"total recorded: {len(results)}")
