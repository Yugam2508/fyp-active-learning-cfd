"""
campaign_domain.py — domain-size sweep and Schwefel control (failure criterion).

Usage:  python campaign_domain.py <config> <start_trial> <n_trials>
Configs:
  phys2 | phys10 | phys50 | phys100 | phys200
          physics benchmark on [-h, h]^3, h = 2 ... 200
  schwefel
          Schwefel function on [-500, 500]^3 (large domain, small dynamic range)

Each trial runs two arms with the same seed and the same N = 20 budget:
  AL      original framework: 5-point LHS + 15 UEigV queries, hard LCS,
          prediction = uniform mean of surviving models (= ActiveLearner)
  LHSGP   passive baseline: 20-point LHS, single GP-RBF
RMSE is computed on 5,000 uniform test points drawn with RandomState(7) over
the configuration's own domain.  Seeds are 1000 + trial index.
Results append to results/results_domain_<config>.json (resumable).

The headline statistic is rho = mean RMSE_AL / mean RMSE_LHSGP (the thesis
definition); medians and per-seed records are stored as well.
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
RESULTS_DIR = _os.path.join(_ROOT, "results"); _os.makedirs(RESULTS_DIR, exist_ok=True)

import sys, json, os, time
import numpy as np
try:
    import torch
    torch.set_num_threads(1)          # one process per config runs in parallel
except Exception:
    torch = None

from framework_core import ToyProblem, ActiveLearner, GPSurrogate, lhs

CONFIGS = {
    "phys2":    ("physics", 2),
    "phys10":   ("physics", 10),
    "phys50":   ("physics", 50),
    "phys100":  ("physics", 100),
    "phys200":  ("physics", 200),
    "schwefel": ("schwefel", 500),
}
MAX_STEPS, N_INIT, N_TEST = 15, 5, 5000

config, start, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
assert config in CONFIGS, f"unknown config {config}"
fname, h = CONFIGS[config]
problem = ToyProblem(3, fname)
domain = np.array([(-h, h)] * 3, float)
OUT = os.path.join(RESULTS_DIR, f"results_domain_{config}.json")

X_test = domain[:, 0] + np.random.RandomState(7).uniform(0, 1, (N_TEST, 3)) \
    * (domain[:, 1] - domain[:, 0])
y_test = problem.evaluate(X_test)


def rmse(pred):
    return float(np.sqrt(np.mean((y_test - pred) ** 2)))


def run_al(seed):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    a = ActiveLearner(problem, domain, n_init=N_INIT)
    a.train()
    for _ in range(MAX_STEPS):
        a.step(); a.train()
    models = [m for m in a.models if m not in a.rejected]
    pred = np.mean([m.predict(X_test) for m in models], axis=0)
    return rmse(pred), [m.name for m in models]


def run_lhsgp(seed):
    np.random.seed(seed)
    X = domain[:, 0] + lhs(3, N_INIT + MAX_STEPS) * (domain[:, 1] - domain[:, 0])
    y = problem.noisy(X)
    gp = GPSurrogate("RBF")
    gp.fit(X, y)
    return rmse(gp.predict(X_test))


results = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {r["seed"] for r in results}
for t in range(start, start + n):
    seed = 1000 + t
    if seed in done:
        print(f"trial {t} (seed {seed}) already done, skipping"); continue
    t0 = time.time()
    r_al, survivors = run_al(seed)
    r_gp = run_lhsgp(seed)
    results.append({"seed": seed, "config": config, "function": fname,
                    "half_width": h, "rmse_al": r_al, "rmse_lhsgp": r_gp,
                    "n_models_final": len(survivors), "final_models": survivors,
                    "output_std": float(y_test.std()),
                    "runtime_s": round(time.time() - t0, 1)})
    json.dump(results, open(OUT, "w"))
    print(f"[{config}] trial {t:2d} seed {seed} | AL={r_al:10.4g} "
          f"LHS+GP={r_gp:10.4g} | {results[-1]['runtime_s']}s", flush=True)
print(f"total recorded: {len(results)}")
