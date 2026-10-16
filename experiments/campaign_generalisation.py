"""
campaign_generalisation.py — controlled comparison of surrogate strategies for
GUM S1 propagation with a small budget of model runs.

Usage:  python campaign_generalisation.py <benchmark> <arm> <start_trial> <n_trials> [N]
Benchmarks: see src/benchmarks.py (physics, plume, borehole, ishigami, otl,
            piston; non-uniform inputs: borehole_gum, plume_gum, flood)
N:          budget of model runs (default 20). Adaptive arms start from
            max(5, N // 4) Latin hypercube runs.

Arms
  Single GP on an N-point Latin hypercube (passive), kernel specification
  (src/surrogates.py):
    gp_orig      RBF, isotropic, amplitude fixed at 1 (the original kernel)
    gp_amp       RBF, isotropic, amplitude estimated
    gp_ardnoamp  RBF, ARD, amplitude fixed
    gp_ard       RBF, ARD, amplitude estimated
    gp_m52       Matern-5/2, ARD, amplitude estimated
    gp_quad      gp_m52 + quadratic polynomial kernel (GP with quadratic trend)
    gp_log       gp_m52 on log(output), log-transformed positive inputs;
                 prediction exp(latent mean) = predictive median
    gp_quadlog   gp_quad on the log scale (positive outputs only)
  Sparse polynomial chaos (LAR, corrected leave-one-out selection), passive:
    spce, spce_log
  Committees of 9 members (2 GP, 2 PCE, 2 RF, SVR, 2 DNN):
    pas_orig     passive, original GP members
    pas_ard      passive, GP members with amplitude + ARD
                 (both record uniform AND CV-weighted aggregation of one fit)
    al_var       adaptive committee (ARD members), committee-variance queries,
                 CV inverse-error weights   (uniform-input benchmarks only)
    al_ueigv     same with UEigV (spectral) acquisition (uniform inputs only)
  Adaptive single GP:
    al_gp        gp_m52, queries at maximum posterior SD (space-filling
                 variance reduction)
    ak_m52, ak_log
                 gp_m52 / gp_log with the U-function of AK-MCS (Echard et al.
                 2011): queries at minimum |mean - T| / sd on the latent scale,
                 targeting the exceedance threshold T (the model's physical
                 threshold if it has one, else the reference 95 % quantile, or
                 5 % quantile for a lower tail)

Seeds 1000 + trial. The seeded LHS (framework_core.lhs) gives every passive arm
the identical design and every adaptive arm the identical start. Candidate
pools for adaptive arms are drawn from the input distribution.

Metrics, against a reference Monte Carlo of the true model (2e5 points,
RandomState(8)); propagation uses 3e4 points (RandomState(9)):
  nrmse            RMSE on 5000 test points (RandomState(7)) / reference SD
  relerr_mean/sd/q95  relative error of the propagated mean, SD, 95th pct.
  exc_*            P(Y > y95_ref): estimate, member envelope, coverage
  thr_*            P(Y > T) (tail "upper") or P(Y < T) (tail "lower") at the
                   model's physical threshold T (if any; else the reference
                   95 % or 5 % quantile)
Committee arms store each metric twice, prefixed "uni_" and "cv_".
Results: results/results_gen_<benchmark>_<arm>.json for N = 20, else
results_gen_<benchmark>_<arm>_N<N>.json (resumable). RESULTS_DIR may be
overridden by the environment variable of the same name (used by tests).

--augment   recompute every existing record, check that all stored fields
            are reproduced (relative tolerance 1e-9), and add fields that the
            record lacks (e.g. thr_* for records made before they existed).
            A record that is not reproduced is reported and left unchanged;
            the exit status is then 1.
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
RESULTS_DIR = _os.environ.get("RESULTS_DIR", _os.path.join(_ROOT, "results"))
_os.makedirs(RESULTS_DIR, exist_ok=True)

import sys, json, os, time
import numpy as np
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    torch = None

from framework_core import (ActiveLearner, GPSurrogate, PCESurrogate,
                            RFSurrogate, SVRSurrogate, DNNSurrogate, lhs)
from uq_extension import SoftActiveLearner, SoftLCS, make_bound_penalty
from benchmarks import BENCHMARKS
from surrogates import GP as _GP, SparsePCE

ARMS = ("gp_orig", "gp_amp", "gp_ardnoamp", "gp_ard", "gp_m52", "gp_quad",
        "gp_log", "gp_quadlog", "spce", "spce_log",
        "pas_orig", "pas_ard", "al_gp", "al_var", "al_ueigv", "ak_m52", "ak_log")
LOG_ARMS = ("gp_log", "gp_quadlog", "spce_log", "ak_log")
AUGMENT = "--augment" in sys.argv
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
bench, arm, start, n = _args[0], _args[1], int(_args[2]), int(_args[3])
N_TOTAL = int(_args[4]) if len(_args) > 4 else 20
assert bench in BENCHMARKS and arm in ARMS, (bench, arm)
problem = BENCHMARKS[bench]()
d = problem.dim
dom = np.array(problem.bounds, float)
UNIFORM = problem.marginals is None
if arm in LOG_ARMS:
    assert problem.positive, f"{arm} needs a positive output"
if arm in ("al_var", "al_ueigv"):
    assert UNIFORM, f"{arm} is implemented for uniform inputs only"
N_INIT = 5 if N_TOTAL <= 20 else N_TOTAL // 4
suffix = "" if N_TOTAL == 20 else f"_N{N_TOTAL}"
OUT = os.path.join(RESULTS_DIR, f"results_gen_{bench}_{arm}{suffix}.json")
penalty = (make_bound_penalty(lower=problem.lower_bound)
           if problem.lower_bound is not None else None)
POS_IN = problem.pos_in


def fixed(n_pts, rs):
    return problem.ppf(rs.uniform(0, 1, (n_pts, d)))


def draw(n_pts):
    """Candidate or probe points from the input distribution (global RNG)."""
    return problem.ppf(np.random.uniform(0, 1, (n_pts, d)))


X_test = fixed(5000, np.random.RandomState(7)); y_test = problem.evaluate(X_test)
X_ref = fixed(200_000, np.random.RandomState(8)); y_ref = problem.evaluate(X_ref)
X_mc = fixed(30_000, np.random.RandomState(9))
REF = {"mean": float(y_ref.mean()), "sd": float(y_ref.std()),
       "q95": float(np.quantile(y_ref, 0.95))}
P_TRUE = float(np.mean(y_ref > REF["q95"]))
LOWER = problem.tail == "lower"
T_THR = (problem.threshold if problem.threshold is not None
         else float(np.quantile(y_ref, 0.05 if LOWER else 0.95)))
exceed = (lambda v: v < T_THR) if LOWER else (lambda v: v > T_THR)
P_THR = float(np.mean(exceed(y_ref)))


def GP(spec):
    return _GP(spec, d, POS_IN)


# ── committees ───────────────────────────────────────────────────────────
def ard_members(dim):
    return [GPSurrogate("RBF", spec="ard"), GPSurrogate("Matern", spec="ard"),
            PCESurrogate(2), PCESurrogate(3), RFSurrogate(50), RFSurrogate(100),
            SVRSurrogate("rbf"), DNNSurrogate(dim, 64), DNNSurrogate(dim, 128)]


class ARDLearner(ActiveLearner):
    def _build_models(self):
        return ard_members(self.dim)


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
    t_k = exceed(P).mean(axis=1)
    t_hat = float(w @ t_k)
    return {
        "nrmse": rmse / REF["sd"], "rmse": rmse,
        "relerr_mean": float(abs(mu.mean() - REF["mean"]) / abs(REF["mean"])),
        "relerr_sd": float(abs(mu.std() - REF["sd"]) / REF["sd"]),
        "relerr_q95": float(abs(np.quantile(mu, 0.95) - REF["q95"]) / abs(REF["q95"])),
        "exc_hat": p_hat, "exc_abserr": abs(p_hat - P_TRUE),
        "exc_lo": float(p_k.min()), "exc_up": float(p_k.max()),
        "exc_covered": bool(p_k.min() - 1e-9 <= P_TRUE <= p_k.max() + 1e-9),
        "thr_hat": t_hat, "thr_true": P_THR, "thr_abserr": abs(t_hat - P_THR),
        "thr_covered": bool(t_k.min() - 1e-9 <= P_THR <= t_k.max() + 1e-9),
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
    X_probe = (np.random.uniform(dom[:, 0], dom[:, 1], size=(200, d)) if UNIFORM
               else draw(200))
    w, _ = SoftLCS(n_folds=3, physics_penalty=penalty).weights(models, X, y,
                                                               X_probe=X_probe)
    return w


def design(n_pts):
    return problem.ppf(lhs(d, n_pts))


def run(seed):
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    if arm.startswith("gp_"):
        X = design(N_TOTAL)
        g = GP(arm[3:]); g.fit(X, problem.noisy(X))
        return single(g)
    if arm.startswith("spce"):
        X = design(N_TOTAL)
        unit = None if UNIFORM else problem.cdf
        s = SparsePCE(d, problem.bounds, log=(arm == "spce_log"), pos_in=POS_IN,
                      unit_map=unit).fit(X, problem.noisy(X))
        return single(s)
    if arm.startswith("pas_"):
        if UNIFORM:                     # the code path of the published records
            cls = ActiveLearner if arm == "pas_orig" else ARDLearner
            a = cls(problem, dom, n_init=N_TOTAL)      # same LHS draw as gp_* arms
            a.train()
            return committee(a.models, cv_weights(a.models, a.X, a.y))
        assert arm == "pas_ard"
        X = design(N_TOTAL); y = problem.noisy(X)
        models = ard_members(d)
        for m in models:
            m.fit(X, y)
        return committee(models, cv_weights(models, X, y))
    if arm in ("al_gp", "ak_m52", "ak_log"):
        X = design(N_INIT)
        y = problem.noisy(X)
        g = GP("log" if arm == "ak_log" else "m52")
        t_lat = g.to_latent(np.array([T_THR]))[0] if arm.startswith("ak") else None
        for _ in range(N_TOTAL - N_INIT):
            g.fit(X, y)
            if arm == "al_gp":
                grid = (np.random.uniform(dom[:, 0], dom[:, 1], size=(300, d))
                        if UNIFORM else draw(300))
                _, sd = g.predict(grid, return_std=True)
                x_new = grid[np.argmax(sd)][None, :]
            else:
                grid = draw(2000)
                mu, sd = g.latent(grid)
                x_new = grid[np.argmin(np.abs(mu - t_lat) / np.maximum(sd, 1e-12))][None, :]
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


def _same(a, b):
    if isinstance(a, dict):
        return all(k in b and _same(v, b[k]) for k, v in a.items())
    if isinstance(a, float):
        return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12))
    return a == b


if __name__ == "__main__" and AUGMENT:
    results = json.load(open(OUT))
    n_bad = 0
    for r in results:
        new = run(r["seed"])
        bad = [k for k, v in new.items() if k in r and not _same(r[k], v)]
        if bad:                     # leave the record untouched, report, go on
            n_bad += 1
            print(f"[{bench}/{arm}] seed {r['seed']}: MISMATCH in {bad}", flush=True)
            continue
        added = [k for k in new if k not in r]
        r.update({k: new[k] for k in added})
        json.dump(results, open(OUT, "w"))
        print(f"[{bench}/{arm}] seed {r['seed']}: reproduced, added {len(added)} fields",
              flush=True)
    print(f"augmented: {len(results) - n_bad}, mismatched: {n_bad}")
    sys.exit(1 if n_bad else 0)

if __name__ == "__main__":
    results = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["seed"] for r in results}
    for t in range(start, start + n):
        seed = 1000 + t
        if seed in done:
            print(f"trial {t} (seed {seed}) already done, skipping"); continue
        t0 = time.time()
        r = {"seed": seed, "benchmark": bench, "arm": arm, "N": N_TOTAL, **run(seed)}
        r["runtime_s"] = round(time.time() - t0, 1)
        results.append(r)
        json.dump(results, open(OUT, "w"))
        key = "nrmse" if "nrmse" in r else "cv_nrmse"
        print(f"[{bench}/{arm}/N{N_TOTAL}] seed {seed} | {key}={r[key]:.4g} | "
              f"{r['runtime_s']}s", flush=True)
    print(f"total recorded: {len(results)}")
