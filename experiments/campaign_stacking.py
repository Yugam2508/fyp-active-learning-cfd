"""
campaign_stacking.py — research plan T2.3: would combining the candidate
surrogates beat choosing one of them?

On the same designs as campaign_selection.py (seeds 1000-1029), the LOO
predictions (original scale) of every candidate are combined by stacking
(Breiman 1996): non-negative weights w minimising
sum_i (y_i - sum_c w_c yhat_{c,-i}(x_i))^2 (non-negative least squares),
applied to the candidates fitted to all N runs. Recorded: the weights, the
stacked nRMSE on the 5000 test points of campaign_generalisation.py, the
nRMSE of the candidate with the smallest LOO error (the selection of the
procedure) and of the best candidate in hindsight.

Usage: python campaign_stacking.py <benchmark> [N]
Results: results/results_stack_<benchmark>[_N<N>].json (resumable)
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
RESULTS_DIR = _os.environ.get("RESULTS_DIR", _os.path.join(_ROOT, "results"))

import sys, json, os, time
import numpy as np
from scipy.optimize import nnls
from framework_core import lhs
from benchmarks import BENCHMARKS
from surrogates import GP, SparsePCE

bench = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
problem = BENCHMARKS[bench]()
d = problem.dim
CANDS = ["gp_m52", "gp_quad", "spce"] + (["gp_log", "gp_quadlog", "spce_log"]
                                         if problem.positive else [])
OUT = os.path.join(RESULTS_DIR, f"results_stack_{bench}" + ("" if N == 20 else f"_N{N}") + ".json")
if hasattr(problem, "reference_sets"):
    _sets = problem.reference_sets()
    X_test, X_ref = _sets["test"], _sets["ref"]
else:
    X_test = problem.ppf(np.random.RandomState(7).uniform(0, 1, (5000, d)))
    X_ref = problem.ppf(np.random.RandomState(8).uniform(0, 1, (200_000, d)))
y_test = problem.evaluate(X_test)
REF_SD = float(problem.evaluate(X_ref).std())


def make(c):
    if c.startswith("gp_"):
        return GP(c[3:], d, problem.pos_in)
    unit = None if problem.marginals is None else problem.cdf
    return SparsePCE(d, problem.bounds, log=c == "spce_log", pos_in=problem.pos_in,
                     unit_map=unit)


def run(seed):
    np.random.seed(seed)
    X = problem.ppf(lhs(d, N))
    y = problem.noisy(X)
    P_loo, P_test = [], []
    for c in CANDS:
        pred = np.empty(N)
        for i in range(N):
            keep = np.arange(N) != i
            pred[i] = make(c).fit(X[keep], y[keep]).predict(X[i:i + 1])[0]
        P_loo.append(pred)
        P_test.append(make(c).fit(X, y).predict(X_test))
    P_loo, P_test = np.array(P_loo).T, np.array(P_test).T
    w, _ = nnls(P_loo, y)
    nr = lambda p: float(np.sqrt(np.mean((y_test - p) ** 2)) / REF_SD)
    single = {c: nr(P_test[:, j]) for j, c in enumerate(CANDS)}
    loo = {c: float(np.sqrt(np.mean((y - P_loo[:, j]) ** 2))) for j, c in enumerate(CANDS)}
    chosen = min(CANDS, key=loo.get)
    return {"stack_weights": dict(zip(CANDS, w.tolist())), "stack_nrmse": nr(P_test @ w),
            "selected": chosen, "selected_nrmse": single[chosen],
            "oracle_nrmse": min(single.values()), "single_nrmse": single,
            "stack_loo_rmse_insample": float(np.sqrt(np.mean((y - P_loo @ w) ** 2)))}


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["seed"] for r in results}
    for seed in range(1000, 1030):
        if seed in done:
            continue
        t0 = time.time()
        rec = {"seed": seed, "benchmark": bench, "N": N, **run(seed)}
        rec["runtime_s"] = round(time.time() - t0, 1)
        results.append(rec)
        json.dump(results, open(OUT, "w"))
        print(f"[{bench}/N{N}] seed {seed}: stack {rec['stack_nrmse']:.4g} selected "
              f"{rec['selected_nrmse']:.4g} oracle {rec['oracle_nrmse']:.4g} | "
              f"{rec['runtime_s']}s", flush=True)
    print(f"total recorded: {len(results)}")
