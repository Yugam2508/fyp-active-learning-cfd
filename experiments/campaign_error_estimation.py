"""
campaign_error_estimation.py — can the surrogate error be estimated from the
20 runs alone?

In a real uncertainty evaluation the true model cannot be run 2e5 times, so
the surrogate's contribution has to be estimated from the training runs. For
the recommended single-GP surrogates this script compares, per seed:

  true_rmse       RMSE on 5000 test points (needs the true model; benchmark only)
  cv_rmse         5-fold cross-validation RMSE on the 20 runs, hyperparameters
                  re-fitted in every fold (available in practice)
  sd_rms          root mean square of the GP posterior SD on the test points
                  (available in practice; on the log scale for gp_log, mapped
                  back by the delta method sd_y = exp(m) sd_log)
  cov95           share of test points inside the GP 95 % predictive interval
                  (for gp_log: exp of the log-scale interval)
  p_idlh          plume only: P(C > 210 mg/m3) from the surrogate, versus the
                  reference value p_idlh_ref

Usage:  python campaign_error_estimation.py <benchmark> <gp_m52|gp_log> <start> <n>
Same seeds (1000 + trial) and 20-point Latin hypercube as
campaign_generalisation.py, so true_rmse equals that study's value.
Results: results/results_err_<benchmark>_<arm>.json (resumable).
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
RESULTS_DIR = _os.path.join(_ROOT, "results")

import sys, json, os, time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from framework_core import lhs
from benchmarks import BENCHMARKS

bench, arm, start, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
assert arm in ("gp_m52", "gp_log")
problem = BENCHMARKS[bench]()
d = problem.dim
dom = np.array(problem.bounds, float)
LOG = arm == "gp_log"
if LOG:
    assert problem.lower_bound == 0.0, "gp_log needs a positive output"
POS_IN = dom[:, 0] > 0
OUT = os.path.join(RESULTS_DIR, f"results_err_{bench}_{arm}.json")
IDLH = 210.0


def fixed(n_pts, rs):
    return dom[:, 0] + rs.uniform(0, 1, (n_pts, d)) * (dom[:, 1] - dom[:, 0])


X_test = fixed(5000, np.random.RandomState(7)); y_test = problem.evaluate(X_test)
X_mc = fixed(30_000, np.random.RandomState(9))
P_IDLH_REF = (float(np.mean(problem.evaluate(fixed(200_000, np.random.RandomState(8)))
                            > IDLH)) if bench == "plume" else None)


def tin(X):
    return np.where(POS_IN, np.log(np.abs(X) + 1e-300), X) if LOG else X


def fit(X, y):
    k = (ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(d), (1e-2, 1e3), nu=2.5)
         + WhiteKernel(1e-4, (1e-8, 1e-1)))
    sc = StandardScaler().fit(tin(X))
    g = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=5,
                                 random_state=0)
    g.fit(sc.transform(tin(X)), np.log(y) if LOG else y)
    return g, sc


def predict(model, X):
    g, sc = model
    m, s = g.predict(sc.transform(tin(X)), return_std=True)
    if LOG:
        return np.exp(m), np.exp(m) * s, np.exp(m - 1.96 * s), np.exp(m + 1.96 * s)
    return m, s, m - 1.96 * s, m + 1.96 * s


def run(seed):
    np.random.seed(seed)
    X = dom[:, 0] + lhs(d, 20) * (dom[:, 1] - dom[:, 0])
    y = problem.noisy(X)
    model = fit(X, y)
    mu, sd, lo, hi = predict(model, X_test)
    true_rmse = float(np.sqrt(np.mean((y_test - mu) ** 2)))
    resid = []
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        mu_te = predict(fit(X[tr], y[tr]), X[te])[0]
        resid.extend(y[te] - mu_te)
    out = {"true_rmse": true_rmse,
           "cv_rmse": float(np.sqrt(np.mean(np.square(resid)))),
           "sd_rms": float(np.sqrt(np.mean(sd ** 2))),
           "cov95": float(np.mean((y_test >= lo) & (y_test <= hi))),
           "ref_sd": float(y_test.std())}
    if bench == "plume":
        out["p_idlh"] = float(np.mean(predict(model, X_mc)[0] > IDLH))
        out["p_idlh_ref"] = P_IDLH_REF
    return out


results = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {r["seed"] for r in results}
for t in range(start, start + n):
    seed = 1000 + t
    if seed in done:
        continue
    t0 = time.time()
    r = {"seed": seed, "benchmark": bench, "arm": arm, **run(seed)}
    r["runtime_s"] = round(time.time() - t0, 1)
    results.append(r)
    json.dump(results, open(OUT, "w"))
    print(f"[{bench}/{arm}] seed {seed} | true {r['true_rmse']:.4g} cv {r['cv_rmse']:.4g} "
          f"sd {r['sd_rms']:.4g} cov95 {r['cov95']:.2f} | {r['runtime_s']}s", flush=True)
print(f"total recorded: {len(results)}")
