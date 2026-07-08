"""
campaign_hardmin.py
===================
Fair-baseline check for the "soft beats hard" claim.

decompose_soft_vs_hard.py showed that hard LCS collapses to K=1 surviving
model in 30/30 trials (median at AL iteration 10 of 15).  Once K<2,
ActiveLearner.ueigv() returns all-ones and argmax selects candidate 0 -- a
uniformly random point.  So ~31% of hard's active-learning budget is random
sampling, and its final "ensemble" is a single model (hence a zero-width
p-box).  Both of the paper's soft-vs-hard claims are therefore confounded by
an implementation artefact rather than by the hard/soft distinction itself.

This script runs hard LCS with a configurable MINIMUM committee size, which
removes the confound: the acquisition signal stays alive and the final
ensemble retains diversity.

Usage: python campaign_hardmin.py <min_active> <start_trial> <n_trials>
   e.g. python campaign_hardmin.py 3 0 30
Appends to results_hardmin<min_active>.json (resumable).
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

min_active, start, n = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
OUT = _os.path.join(RESULTS_DIR, f"results_hardmin{min_active}.json")
MAX_STEPS = 15

problem = ToyProblem(3, "physics")
domain = problem.bounds


def fixed_sampler(n_pts, rs):
    b = np.array(domain, float)
    return b[:, 0] + rs.uniform(0, 1, (n_pts, 3)) * (b[:, 1] - b[:, 0])


X_test = fixed_sampler(5000, np.random.RandomState(7))   # identical to campaign.py
y_test = problem.evaluate(X_test)


class HardMinLearner(ActiveLearner):
    """Hard LCS rejection, but never below `min_active` surviving models."""
    MIN_ACTIVE = 3

    def step(self, verbose=False):
        grid = np.random.uniform(self.domain[:, 0], self.domain[:, 1],
                                 size=(300, self.dim))
        ue = self.ueigv(grid)
        idx = np.argmax(ue)
        x_new = grid[idx:idx+1]
        y_new = self.problem.noisy(x_new)
        self.X = np.vstack([self.X, x_new])
        self.y = np.append(self.y, y_new)
        active = [m for m in self.models if m not in self.rejected]
        if len(active) > self.MIN_ACTIVE:            # <-- the only change
            rej, info = self.lcs.reject_one(active, self.X, self.y)
            if rej is not None:
                self.rejected.add(rej)
        return x_new, y_new


def run_trial(seed):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    HardMinLearner.MIN_ACTIVE = min_active
    a = HardMinLearner(ToyProblem(3, "physics"), domain)
    a.train()
    for _ in range(MAX_STEPS):
        a.step(); a.train()
    survivors = [m for m in a.models if m not in a.rejected]
    pred = np.mean([m.predict(X_test) for m in survivors], axis=0)
    rmse = float(np.sqrt(np.mean((y_test - pred) ** 2)))
    return {"seed": seed, "min_active": min_active, "rmse": rmse,
            "n_survivors": len(survivors),
            "survivor_names": [m.name for m in survivors]}


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
    print(f"[hardmin{min_active}] trial {t:2d} seed {seed} | RMSE={r['rmse']:9.3f} "
          f"| survivors={r['n_survivors']} {r['survivor_names']} | {r['runtime_s']}s",
          flush=True)
print(f"total recorded: {len(results)}")
