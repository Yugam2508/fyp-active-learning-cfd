"""
campaign_selection.py — leave-one-out (LOO) errors of the candidate
surrogates, for evaluating the selection step of the proposed procedure
(choose the candidate with the smallest LOO error) without the true model.

For each seed, on the same N-point design as campaign_generalisation.py,
every candidate is re-fitted N times without one run and the LOO RMSE is
computed on the ORIGINAL output scale (so that raw and log-scale candidates
are comparable):  sqrt(mean_i (y_i - yhat_{-i}(x_i))^2).
Candidates: gp_m52, gp_quad, spce, and for positive outputs also gp_log,
gp_quadlog, spce_log. The chosen candidate's validation errors are then read
from the campaign_generalisation.py records of the same seed (identical fit).

Usage: python campaign_selection.py <benchmark> [N]
Results: results/results_sel_<benchmark>[_N<N>].json (resumable)
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
RESULTS_DIR = _os.environ.get("RESULTS_DIR", _os.path.join(_ROOT, "results"))

import sys, json, os, time
import numpy as np
from framework_core import lhs
from benchmarks import BENCHMARKS
from surrogates import GP, SparsePCE

bench = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
problem = BENCHMARKS[bench]()
d = problem.dim
CANDS = ["gp_m52", "gp_quad", "spce"] + (["gp_log", "gp_quadlog", "spce_log"]
                                         if problem.positive else [])
OUT = os.path.join(RESULTS_DIR, f"results_sel_{bench}" + ("" if N == 20 else f"_N{N}") + ".json")


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
    out = {}
    for c in CANDS:
        pred = np.empty(N)
        for i in range(N):
            keep = np.arange(N) != i
            s = make(c).fit(X[keep], y[keep])
            pred[i] = s.predict(X[i:i + 1])[0]
        out[f"loo_rmse_{c}"] = float(np.sqrt(np.mean((y - pred) ** 2)))
    return out


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
        best = min(CANDS, key=lambda c: rec[f"loo_rmse_{c}"])
        print(f"[{bench}/N{N}] seed {seed}: chosen {best} | {rec['runtime_s']}s", flush=True)
    print(f"total recorded: {len(results)}")
