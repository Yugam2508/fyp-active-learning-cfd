"""
campaign_coverage.py — is the surrogate uncertainty contribution valid?

The conference paper adds the surrogate (emulator) error to the GUM S1
propagation as an extra uncertainty contribution estimated from the N runs.
This script tests, against the true model, whether the intervals obtained in
that way reach their nominal 95 % coverage. Per benchmark, surrogate and
seed (same seeds, Latin hypercube and fits as campaign_generalisation.py, so
nrmse equals that study's value):

1. Pointwise prediction intervals at 5000 test points (RandomState(7)):
     gp       GP predictive interval  mean +- 1.96 sd          (GP only)
     cvnorm   mean +- 1.96 s_LOO, s_LOO the RMS leave-one-out residual
     jkplus   jackknife+ (Barber et al., Ann. Stat. 2021): bounds are the
              floor(a(n+1))-th smallest of mu_-i(x) - |R_i| and the
              ceil((1-a)(n+1))-th smallest of mu_-i(x) + |R_i|, a = 0.05,
              with mu_-i the surrogate refitted (hyperparameters included)
              without run i and R_i its residual at run i; distribution-free
              guarantee of coverage >= 1 - 2a = 90 %
   recorded: coverage, mean width / reference SD, interval score / ref. SD.

2. GUM S1 95 % coverage interval of the output. Propagation of 3e4 input
   draws (RandomState(9)) through the surrogate, with the surrogate error E
   added to every draw as
     plain    E = 0 (surrogate treated as exact)
     normal   E ~ N(0, s_LOO^2)
     resample E drawn from the n leave-one-out residuals
     gp       E ~ N(0, sd(x)^2), the GP predictive SD at the draw (GP only)
   recorded: achieved coverage P(lo <= Y <= hi) under the reference
   distribution (2e5 true-model draws, RandomState(8)), interval length
   relative to the reference interval, u(y) / reference SD, and the
   exceedance probability at the benchmark's threshold.

3. Uncertainty of derived quantities (mean, SD, 95th percentile, exceedance
   probability) of the output over 2000 fixed input draws (the first 2000 of
   the propagation sample; for the CFD case the 1600 "q" points of
   src/cfd_benchmark.py, which reuse the reference runs), compared with the same quantities of the true
   model on the same draws, so that only the emulator error is assessed:
     gpjoint  95 % interval of the quantity over 1000 joint GP posterior
              sample paths (GP only)
     jack     estimate +- 1.96 jackknife SE from the n leave-one-out fits
   recorded: estimate, bounds, truth, covered.

Errors are added on the fitting (latent) scale and mapped back, so for the
log variants E is multiplicative.

Usage: python campaign_coverage.py <benchmark> <surrogate> <start> <n> [N]
Surrogates: gp_orig, gp_m52, gp_quad, gp_log, spce, spce_log
Results: results/results_cov_<benchmark>_<surrogate>[_N<N>].json (resumable)
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

SURR = ("gp_orig", "gp_m52", "gp_quad", "gp_log", "spce", "spce_log")
bench, surr, start, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
N = int(sys.argv[5]) if len(sys.argv) > 5 else 20
assert bench in BENCHMARKS and surr in SURR, (bench, surr)
problem = BENCHMARKS[bench]()
d = problem.dim
IS_GP = surr.startswith("gp_")
if surr.endswith("log"):
    assert problem.positive, f"{surr} needs a positive output"
OUT = os.path.join(RESULTS_DIR, f"results_cov_{bench}_{surr}"
                   + ("" if N == 20 else f"_N{N}") + ".json")
ALPHA = 0.05
Z = 1.959963984540054
N_Q = 2000            # draws for the derived-quantity assessment
N_PATHS = 1000        # GP posterior sample paths


def fixed(n_pts, rs):
    return problem.ppf(rs.uniform(0, 1, (n_pts, d)))


X_mc = fixed(30_000, np.random.RandomState(9))
HAS_SETS = hasattr(problem, "reference_sets")
if HAS_SETS:                             # CFD: reference built from cached runs
    _sets = problem.reference_sets()
    X_test, y_ref, X_q = _sets["test"], problem.evaluate(_sets["ref"]), _sets["q"]
    N_Q = len(X_q)
else:
    X_test = fixed(5000, np.random.RandomState(7))
    y_ref = problem.evaluate(fixed(200_000, np.random.RandomState(8)))
    X_q = X_mc[:N_Q]
y_test = problem.evaluate(X_test); y_q = problem.evaluate(X_q)
REF_SD = float(y_ref.std())
REF_LO, REF_HI = np.quantile(y_ref, [ALPHA / 2, 1 - ALPHA / 2])
LOWER = problem.tail == "lower"
T_THR = (problem.threshold if problem.threshold is not None
         else float(np.quantile(y_ref, 0.05 if LOWER else 0.95)))


def exceed(v):
    return v < T_THR if LOWER else v > T_THR


P_THR = float(np.mean(exceed(y_ref)))


def make():
    if IS_GP:
        return GP(surr[3:], d, problem.pos_in)
    unit = None if problem.marginals is None else problem.cdf
    return SparsePCE(d, problem.bounds, log=surr == "spce_log",
                     pos_in=problem.pos_in, unit_map=unit)


def quantities(Y):
    """Derived quantities of output samples; Y has draws along the last axis."""
    return {"mean": Y.mean(-1), "sd": Y.std(-1),
            "q95": np.quantile(Y, 0.95, axis=-1), "p_thr": exceed(Y).mean(-1)}


def interval_stats(lo, hi, y):
    w = hi - lo
    score = w + 2 / ALPHA * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    return {"cov": float(np.mean((y >= lo) & (y <= hi))),
            "width": float(np.mean(w) / REF_SD),
            "score": float(np.mean(score) / REF_SD)}


def jkplus(mu_loo, R):
    """Jackknife+ bounds; mu_loo (n, m) LOO predictions, R (n,) |residuals|."""
    n_ = len(R)
    k_lo, k_hi = int(np.floor(ALPHA * (n_ + 1))), int(np.ceil((1 - ALPHA) * (n_ + 1)))
    lo = (np.sort(mu_loo - R[:, None], axis=0)[k_lo - 1] if k_lo >= 1
          else np.full(mu_loo.shape[1], -np.inf))
    hi = (np.sort(mu_loo + R[:, None], axis=0)[k_hi - 1] if k_hi <= n_
          else np.full(mu_loo.shape[1], np.inf))
    return lo, hi


def run(seed):
    np.random.seed(seed)
    X = problem.ppf(lhs(d, N))
    y = problem.noisy(X)
    rs = np.random.RandomState(seed + 10 ** 6)          # extra randomness only
    s = make().fit(X, y)
    m = s.from_latent
    z = s.to_latent(y)
    mu_t, sd_t = s.latent(X_test)
    mu_mc, sd_mc = s.latent(X_mc)
    out = {"nrmse": float(np.sqrt(np.mean((y_test - m(mu_t)) ** 2)) / REF_SD)}

    # leave-one-out refits
    mu_loo_t, mu_loo_q, r = [], [], []
    for i in range(N):
        keep = np.arange(N) != i
        si = make().fit(X[keep], y[keep])
        r.append(z[i] - si.latent(X[i:i + 1])[0][0])
        mu_loo_t.append(si.latent(X_test)[0])
        mu_loo_q.append(si.latent(X_q)[0])
    r = np.array(r); mu_loo_t = np.array(mu_loo_t); mu_loo_q = np.array(mu_loo_q)
    s_loo = float(np.sqrt(np.mean(r ** 2)))
    out["s_loo_latent"] = s_loo

    # 1. pointwise intervals
    pw = {"cvnorm": (mu_t - Z * s_loo, mu_t + Z * s_loo),
          "jkplus": jkplus(mu_loo_t, np.abs(r))}
    if IS_GP:
        pw["gp"] = (mu_t - Z * sd_t, mu_t + Z * sd_t)
    for k, (lo, hi) in pw.items():
        out.update({f"pw_{k}_{q}": v
                    for q, v in interval_stats(m(lo), m(hi), y_test).items()})

    # 2. GUM S1 coverage interval with the surrogate error added
    eps = rs.standard_normal(len(X_mc))
    E = {"plain": 0.0, "normal": s_loo * eps,
         "resample": r[rs.randint(0, N, len(X_mc))]}
    if IS_GP:
        E["gp"] = sd_mc * eps
    for k, e in E.items():
        ys = m(mu_mc + e)
        lo, hi = np.quantile(ys, [ALPHA / 2, 1 - ALPHA / 2])
        thr = float(np.mean(exceed(ys)))
        out.update({f"ci_{k}_cov": float(np.mean((y_ref >= lo) & (y_ref <= hi))),
                    f"ci_{k}_len": float((hi - lo) / (REF_HI - REF_LO)),
                    f"ci_{k}_u": float(ys.std() / REF_SD),
                    f"ci_{k}_thr": thr, f"ci_{k}_thr_abserr": abs(thr - P_THR)})

    # 3. derived quantities over the same N_Q draws
    truth = quantities(y_q)
    est = quantities(m(s.latent(X_q)[0] if HAS_SETS else mu_mc[:N_Q]))
    jk = quantities(m(mu_loo_q))
    for k in truth:
        se = np.sqrt((N - 1) / N * np.sum((jk[k] - jk[k].mean()) ** 2))
        lo, hi = est[k] - Z * se, est[k] + Z * se
        out.update({f"dq_{k}_true": float(truth[k]), f"dq_{k}_hat": float(est[k]),
                    f"dq_{k}_jack_lo": float(lo), f"dq_{k}_jack_hi": float(hi),
                    f"dq_{k}_jack_cov": bool(lo <= truth[k] <= hi)})
    if IS_GP:
        Xs = s.sc.transform(s._tin(X_q))
        mu_q, C = s.gp.predict(Xs, return_cov=True)
        jit = 1e-10 * np.mean(np.diag(C))
        for _ in range(8):
            try:
                L = np.linalg.cholesky(C + jit * np.eye(N_Q)); break
            except np.linalg.LinAlgError:
                jit *= 10
        paths = m(mu_q[None, :] + (L @ rs.standard_normal((N_Q, N_PATHS))).T)
        gq = quantities(paths)
        for k in truth:
            lo, hi = np.quantile(gq[k], [ALPHA / 2, 1 - ALPHA / 2])
            out.update({f"dq_{k}_gp_lo": float(lo), f"dq_{k}_gp_hi": float(hi),
                        f"dq_{k}_gp_cov": bool(lo <= truth[k] <= hi)})
        out["gp_jitter"] = float(jit)
    return out


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["seed"] for r in results}
    for t in range(start, start + n):
        seed = 1000 + t
        if seed in done:
            continue
        t0 = time.time()
        rec = {"seed": seed, "benchmark": bench, "surrogate": surr, "N": N, **run(seed)}
        rec["runtime_s"] = round(time.time() - t0, 1)
        results.append(rec)
        json.dump(results, open(OUT, "w"))
        print(f"[{bench}/{surr}/N{N}] seed {seed} | nrmse={rec['nrmse']:.3g} "
              f"pw cov jk+={rec['pw_jkplus_cov']:.2f} cv={rec['pw_cvnorm_cov']:.2f}"
              + (f" gp={rec['pw_gp_cov']:.2f}" if IS_GP else "")
              + f" | CI cov plain={rec['ci_plain_cov']:.3f} "
              f"normal={rec['ci_normal_cov']:.3f} | {rec['runtime_s']}s", flush=True)
    print(f"total recorded: {len(results)}")
