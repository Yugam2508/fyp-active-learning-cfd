"""
campaign_generalisation.py — controlled comparison of surrogate strategies for
GUM S1 propagation at a budget of N = 20 model runs, on four test models.

Usage:  python campaign_generalisation.py <benchmark> <arm> <start_trial> <n_trials>
Benchmarks: physics | plume | borehole | ishigami      (src/benchmarks.py)

Arms
  Single GP on a 20-point Latin hypercube (passive), kernel specification:
    gp_orig      RBF, isotropic, NO amplitude  (the committee's original member)
    gp_amp       RBF, isotropic, with amplitude (ConstantKernel)
    gp_ardnoamp  RBF, ARD, no amplitude
    gp_ard       RBF, ARD, with amplitude
    gp_m52       Matern-5/2, ARD, with amplitude (common default in practice)
  Committees of 9 members (2 GP, 2 PCE, 2 RF, SVR, 2 DNN):
    pas_orig     passive 20-point LHS, original GP members
    pas_ard      passive 20-point LHS, GP members with amplitude + ARD
                 (both record uniform AND CV-weighted aggregation of the same fit)
    al_gp        adaptive single GP (Matern-5/2 ARD + amplitude): 5 LHS + 15
                 queries at maximum posterior SD -- standard adaptive design
    al_var       adaptive committee (ARD members): 15 committee-variance queries,
                 CV inverse-error weights
    al_ueigv     same with UEigV (spectral) acquisition

Seeds 1000 + trial. The seeded LHS (framework_core.lhs) gives every passive arm
the identical 20-point design for a given seed, and every adaptive arm the
identical 5-point start.

Metrics, against a reference Monte Carlo of the true model (2e5 points,
RandomState(8)); propagation uses 3e4 points (RandomState(9)); inputs uniform:
  nrmse            RMSE on 5000 test points (RandomState(7)) / reference SD
  relerr_mean/sd/q95  relative error of the propagated mean, SD, 95th pct.
  exc_*            P(Y > y95_ref): estimate, member envelope, coverage
Committee arms store each metric twice, prefixed "uni_" and "cv_".
Results append to results/results_gen_<benchmark>_<arm>.json (resumable).
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
RESULTS_DIR = _os.path.join(_ROOT, "results"); _os.makedirs(RESULTS_DIR, exist_ok=True)

import sys, json, os, time
import numpy as np
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    torch = None
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, ConstantKernel,
                                              WhiteKernel)
from sklearn.preprocessing import StandardScaler

from framework_core import (ActiveLearner, GPSurrogate, PCESurrogate,
                            RFSurrogate, SVRSurrogate, DNNSurrogate, lhs)
from uq_extension import SoftActiveLearner, SoftLCS, make_bound_penalty
from benchmarks import BENCHMARKS

ARMS = ("gp_orig", "gp_amp", "gp_ardnoamp", "gp_ard", "gp_m52",
        "pas_orig", "pas_ard", "al_gp", "al_var", "al_ueigv")
bench, arm, start, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
assert bench in BENCHMARKS and arm in ARMS, (bench, arm)
problem = BENCHMARKS[bench]()
d = problem.dim
dom = np.array(problem.bounds, float)
OUT = os.path.join(RESULTS_DIR, f"results_gen_{bench}_{arm}.json")
N_TOTAL, N_INIT = 20, 5
penalty = (make_bound_penalty(lower=problem.lower_bound)
           if problem.lower_bound is not None else None)


def fixed(n_pts, rs):
    return dom[:, 0] + rs.uniform(0, 1, (n_pts, d)) * (dom[:, 1] - dom[:, 0])


X_test = fixed(5000, np.random.RandomState(7)); y_test = problem.evaluate(X_test)
X_ref = fixed(200_000, np.random.RandomState(8)); y_ref = problem.evaluate(X_ref)
X_mc = fixed(30_000, np.random.RandomState(9))
REF = {"mean": float(y_ref.mean()), "sd": float(y_ref.std()),
       "q95": float(np.quantile(y_ref, 0.95))}
P_TRUE = float(np.mean(y_ref > REF["q95"]))


# ── single GP with a chosen kernel specification ─────────────────────────
class GP:
    def __init__(self, spec):
        self.spec, self.name = spec, f"GP[{spec}]"

    def _kernel(self):
        ls = np.ones(d)
        return {
            "orig":     lambda: RBF() + WhiteKernel(1e-2),
            "amp":      lambda: ConstantKernel(1.0, (1e-3, 1e3)) * RBF()
                                + WhiteKernel(1e-2),
            "ardnoamp": lambda: RBF(ls) + WhiteKernel(1e-2),
            "ard":      lambda: ConstantKernel(1.0, (1e-3, 1e3))
                                * RBF(ls, (1e-2, 1e3))
                                + WhiteKernel(1e-4, (1e-8, 1e-1)),
            "m52":      lambda: ConstantKernel(1.0, (1e-3, 1e3))
                                * Matern(ls, (1e-2, 1e3), nu=2.5)
                                + WhiteKernel(1e-4, (1e-8, 1e-1)),
        }[self.spec]()

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.gp = GaussianProcessRegressor(self._kernel(), normalize_y=True,
                                           n_restarts_optimizer=5,
                                           random_state=0)
        self.gp.fit(self.sc.transform(X), y)

    def predict(self, X, return_std=False):
        return self.gp.predict(self.sc.transform(X), return_std=return_std)


# ── committee with correctly specified GP members ────────────────────────
class ARDLearner(ActiveLearner):
    def _build_models(self):
        return [GPSurrogate("RBF", spec="ard"), GPSurrogate("Matern", spec="ard"),
                PCESurrogate(2), PCESurrogate(3), RFSurrogate(50), RFSurrogate(100),
                SVRSurrogate("rbf"), DNNSurrogate(self.dim, 64),
                DNNSurrogate(self.dim, 128)]


class SoftVarLearner(SoftActiveLearner):
    """CV-weighted committee; acquisition = variance over active members."""
    def ueigv(self, grid):
        models = [m for m, a in zip(self.base.models, self.active) if a]
        P = np.array([m.predict(grid) for m in models])
        return P.var(axis=0) if len(models) >= 2 else np.ones(len(grid))


def metrics(Pt, P, w, names):
    """Pt: member predictions on test set, P: on MC sample, w: weights."""
    w = np.asarray(w, float) / np.sum(w)
    rmse = float(np.sqrt(np.mean((y_test - w @ Pt) ** 2)))
    mu = w @ P
    p_k = (P > REF["q95"]).mean(axis=1)
    p_hat = float(w @ p_k)
    return {
        "nrmse": rmse / REF["sd"], "rmse": rmse,
        "relerr_mean": float(abs(mu.mean() - REF["mean"]) / abs(REF["mean"])),
        "relerr_sd": float(abs(mu.std() - REF["sd"]) / REF["sd"]),
        "relerr_q95": float(abs(np.quantile(mu, 0.95) - REF["q95"]) / abs(REF["q95"])),
        "exc_hat": p_hat, "exc_abserr": abs(p_hat - P_TRUE),
        "exc_lo": float(p_k.min()), "exc_up": float(p_k.max()),
        "exc_covered": bool(p_k.min() - 1e-9 <= P_TRUE <= p_k.max() + 1e-9),
        "max_weight": float(w.max()), "top_member": names[int(np.argmax(w))],
    }


def single(model):
    return metrics(model.predict(X_test)[None, :], model.predict(X_mc)[None, :],
                   [1.0], [model.name])


def committee(models, w_cv):
    Pt = np.array([m.predict(X_test) for m in models])
    P = np.array([m.predict(X_mc) for m in models])
    names = [m.name for m in models]
    out = {f"uni_{k}": v for k, v in metrics(Pt, P, np.ones(len(models)), names).items()}
    out.update({f"cv_{k}": v for k, v in metrics(Pt, P, w_cv, names).items()})
    out["cv_weights"] = {nm: float(x) for nm, x in zip(names, np.asarray(w_cv) / np.sum(w_cv))}
    return out


def cv_weights(models, X, y):
    X_probe = np.random.uniform(dom[:, 0], dom[:, 1], size=(200, d))
    w, _ = SoftLCS(n_folds=3, physics_penalty=penalty).weights(models, X, y,
                                                               X_probe=X_probe)
    return w


def run(seed):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    if arm.startswith("gp_"):
        X = dom[:, 0] + lhs(d, N_TOTAL) * (dom[:, 1] - dom[:, 0])
        g = GP(arm[3:]); g.fit(X, problem.noisy(X))
        return single(g)
    if arm.startswith("pas_"):
        cls = ActiveLearner if arm == "pas_orig" else ARDLearner
        a = cls(problem, dom, n_init=N_TOTAL)        # same LHS draw as gp_* arms
        a.train()
        return committee(a.models, cv_weights(a.models, a.X, a.y))
    if arm == "al_gp":
        X = dom[:, 0] + lhs(d, N_INIT) * (dom[:, 1] - dom[:, 0])
        y = problem.noisy(X)
        g = GP("m52")
        for _ in range(N_TOTAL - N_INIT):
            g.fit(X, y)
            grid = np.random.uniform(dom[:, 0], dom[:, 1], size=(300, d))
            _, sd = g.predict(grid, return_std=True)
            x_new = grid[np.argmax(sd)][None, :]
            X = np.vstack([X, x_new]); y = np.append(y, problem.noisy(x_new))
        g.fit(X, y)
        return single(g)
    cls = SoftVarLearner if arm == "al_var" else SoftActiveLearner
    a = cls(ARDLearner, problem, dom, n_init=N_INIT, n_folds=3, update_every=3,
            physics_penalty=penalty)
    a.train()
    for i in range(N_TOTAL - N_INIT):
        a.step(); a.train(force_weight_update=(i == N_TOTAL - N_INIT - 1))
    return committee(a.models, a.w)


results = json.load(open(OUT)) if os.path.exists(OUT) else []
done = {r["seed"] for r in results}
for t in range(start, start + n):
    seed = 1000 + t
    if seed in done:
        print(f"trial {t} (seed {seed}) already done, skipping"); continue
    t0 = time.time()
    r = {"seed": seed, "benchmark": bench, "arm": arm, **run(seed)}
    r["runtime_s"] = round(time.time() - t0, 1)
    results.append(r)
    json.dump(results, open(OUT, "w"))
    key = "nrmse" if "nrmse" in r else "cv_nrmse"
    print(f"[{bench}/{arm}] seed {seed} | {key}={r[key]:.4g} | {r['runtime_s']}s",
          flush=True)
print(f"total recorded: {len(results)}")
