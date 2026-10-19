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

4. GP predictive distributions that do not plug in the maximum-likelihood
   hyperparameters (research plan T1.2; src/gp_bayes.py), for GP surrogates,
   assessed in all three ways above (pointwise, GUM S1 interval with E,
   derived quantities); extra randomness from RandomState(seed + 2e6), so
   the fields of 1-3 are unchanged:
     gpt    Student-t process: variance scale and constant mean integrated
            out (reference prior), correlation hyperparameters at ML
     gpcal  GP SD recalibrated by the LOO standardised residuals
            z_i = r_i / s_-i(x_i): pointwise mu +- q s, q the
            ceil(0.95(N+1))-th smallest |z_i|; E = s(x) z_J; paths with
            covariance scaled by mean(z_i^2)
     gpmc   all hyperparameters integrated out by MCMC (800 steps, 300 burn-in,
            200 posterior draws,
            uniform prior on the log-hyperparameters within the ML bounds);
            mixture predictive (derived quantities: 40 paths for each of 25
            draws); N <= 40 only (cost)

Usage: python campaign_coverage.py <benchmark> <surrogate> <start> <n> [N] [--augment]
Surrogates: gp_orig, gp_m52, gp_quad, gp_log, gp_quadlog, spce, spce_log
Results: results/results_cov_<benchmark>_<surrogate>[_N<N>].json (resumable)
--design=K design type (src/designs.py), suffix _d<K> for K != lhs (T2.1)
--nobayes  skip section 4 (used with --design)
--augment  recompute every existing record, check that all stored fields are
           reproduced (relative tolerance 1e-9) and add the fields it lacks;
           a record that is not reproduced is reported and left unchanged
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
from designs import make_design
from scipy.stats import t as student
import gp_bayes

SURR = ("gp_orig", "gp_m52", "gp_quad", "gp_log", "gp_quadlog", "spce", "spce_log")
AUGMENT = "--augment" in sys.argv
_args = [a for a in sys.argv[1:] if not a.startswith("--")]
DESIGN = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--design=")), "lhs")
BAYES = "--nobayes" not in sys.argv
bench, surr, start, n = _args[0], _args[1], int(_args[2]), int(_args[3])
N = int(_args[4]) if len(_args) > 4 else 20
assert bench in BENCHMARKS and surr in SURR, (bench, surr)
problem = BENCHMARKS[bench]()
d = problem.dim
IS_GP = surr.startswith("gp_")
if surr.endswith("log"):
    assert problem.positive, f"{surr} needs a positive output"
OUT = os.path.join(RESULTS_DIR, f"results_cov_{bench}_{surr}"
                   + ("" if N == 20 else f"_N{N}")
                   + ("" if DESIGN == "lhs" else f"_d{DESIGN}") + ".json")
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
    X = problem.ppf(make_design(DESIGN, d, N))
    y = problem.noisy(X)
    rs = np.random.RandomState(seed + 10 ** 6)          # extra randomness only
    s = make().fit(X, y)
    m = s.from_latent
    z = s.to_latent(y)
    mu_t, sd_t = s.latent(X_test)
    mu_mc, sd_mc = s.latent(X_mc)
    out = {"nrmse": float(np.sqrt(np.mean((y_test - m(mu_t)) ** 2)) / REF_SD)}

    # leave-one-out refits
    mu_loo_t, mu_loo_q, r, s_loo_pt = [], [], [], []
    for i in range(N):
        keep = np.arange(N) != i
        si = make().fit(X[keep], y[keep])
        lat_i = si.latent(X[i:i + 1])
        r.append(z[i] - lat_i[0][0])
        if IS_GP:
            s_loo_pt.append(lat_i[1][0])
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
    if IS_GP and BAYES:
        out.update(bayes_variants(s, m, r, np.array(s_loo_pt), mu_t, sd_t, mu_mc, sd_mc,
                                  truth, np.random.RandomState(seed + 2 * 10 ** 6)))
    return out


def ci_stats(tag, ys):
    """GUM S1 95 % interval of the propagated samples ys, scored as in 2."""
    lo, hi = np.quantile(ys, [ALPHA / 2, 1 - ALPHA / 2])
    thr = float(np.mean(exceed(ys)))
    return {f"ci_{tag}_cov": float(np.mean((y_ref >= lo) & (y_ref <= hi))),
            f"ci_{tag}_len": float((hi - lo) / (REF_HI - REF_LO)),
            f"ci_{tag}_u": float(ys.std() / REF_SD),
            f"ci_{tag}_thr": thr, f"ci_{tag}_thr_abserr": abs(thr - P_THR)}


def dq_stats(tag, paths, truth):
    """95 % intervals of the derived quantities over sample paths (n, N_Q)."""
    out = {}
    gq = quantities(paths)
    for k in truth:
        lo, hi = np.quantile(gq[k], [ALPHA / 2, 1 - ALPHA / 2])
        out.update({f"dq_{k}_{tag}_lo": float(lo), f"dq_{k}_{tag}_hi": float(hi),
                    f"dq_{k}_{tag}_cov": bool(lo <= truth[k] <= hi)})
    return out


def bayes_variants(s, m, r, s_loo_pt, mu_t, sd_t, mu_mc, sd_mc, truth, rs):
    """Section 4 of the docstring: gpt, gpcal and (N <= 40) gpmc."""
    out = {}
    n_mc = len(X_mc)
    # gpt: Student-t process
    mt, st, nu = gp_bayes.student_t(s, X_test)
    tq = student.ppf(1 - ALPHA / 2, nu)
    out.update({f"pw_gpt_{q}": v for q, v in
                interval_stats(m(mt - tq * st), m(mt + tq * st), y_test).items()})
    mm, sm, _ = gp_bayes.student_t(s, X_mc)
    out.update(ci_stats("gpt", m(mm + sm * rs.standard_t(nu, n_mc))))
    mq, Cq, _ = gp_bayes.student_t(s, X_q, full_cov=True)
    g = gp_bayes.mvn_paths(mq, Cq, N_PATHS, rs)
    g = mq[None, :] + (g - mq[None, :]) * np.sqrt(nu / rs.chisquare(nu, N_PATHS))[:, None]
    out.update(dq_stats("gpt", m(g), truth))
    out["gpt_nu"] = int(nu)
    # gpcal: GP SD recalibrated by the LOO standardised residuals
    zs = r / np.maximum(s_loo_pt, 1e-300)
    k_hi = int(np.ceil((1 - ALPHA) * (N + 1)))
    qz = np.sort(np.abs(zs))[k_hi - 1] if k_hi <= N else np.inf
    out.update({f"pw_gpcal_{q}": v for q, v in
                interval_stats(m(mu_t - qz * sd_t), m(mu_t + qz * sd_t), y_test).items()})
    out.update(ci_stats("gpcal", m(mu_mc + sd_mc * zs[rs.randint(0, N, n_mc)])))
    c2 = float(np.mean(zs ** 2))
    mq0, Cq0 = gp_bayes.gp_cov(s, X_q)
    out.update(dq_stats("gpcal", m(gp_bayes.mvn_paths(mq0, c2 * Cq0, N_PATHS, rs)), truth))
    out.update({"gpcal_rms_z": float(np.sqrt(c2)), "gpcal_q": float(qz)})
    # gpmc: all hyperparameters integrated out by MCMC
    if N <= 40:
        th, acc, tau = gp_bayes.mcmc_thetas(s, rs)
        md, sdd = gp_bayes.predict_draws(s, th, X_test)
        smp = md[:, None, :] + sdd[:, None, :] * rs.standard_normal((len(th), 5, len(X_test)))
        lo, hi = np.quantile(smp.reshape(-1, len(X_test)), [ALPHA / 2, 1 - ALPHA / 2], axis=0)
        out.update({f"pw_gpmc_{q}": v for q, v in interval_stats(m(lo), m(hi), y_test).items()})
        md, sdd = gp_bayes.predict_draws(s, th, X_mc)
        j = rs.randint(0, len(th), n_mc)
        out.update(ci_stats("gpmc", m(md[j, np.arange(n_mc)] + sdd[j, np.arange(n_mc)]
                                      * rs.standard_normal(n_mc))))
        g = np.vstack([gp_bayes.mvn_paths(*gp_bayes.gp_cov(s, X_q, t_), N_PATHS // 25, rs)
                       for t_ in th[::len(th) // 25][:25]])
        out.update(dq_stats("gpmc", m(g), truth))
        out.update({"gpmc_acceptance": acc, "gpmc_tau": tau})
    return out


def _same(a, b):
    if isinstance(a, float):
        return bool(np.isclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True))
    return a == b


if __name__ == "__main__" and AUGMENT:
    results = json.load(open(OUT))
    n_bad = 0
    for rec in results:
        t0 = time.time()
        new = run(rec["seed"])
        bad = [k for k, v in new.items() if k in rec and not _same(rec[k], v)]
        if bad:
            n_bad += 1
            print(f"[{bench}/{surr}/N{N}] seed {rec['seed']}: MISMATCH in {bad}", flush=True)
            continue
        added = [k for k in new if k not in rec]
        rec.update({k: new[k] for k in added})
        rec["augment_runtime_s"] = round(time.time() - t0, 1)
        json.dump(results, open(OUT, "w"))
        print(f"[{bench}/{surr}/N{N}] seed {rec['seed']}: reproduced, added {len(added)} fields"
              + (f" | pw gpt={rec['pw_gpt_cov']:.2f} gpcal={rec['pw_gpcal_cov']:.2f}"
                 + (f" gpmc={rec['pw_gpmc_cov']:.2f}" if 'pw_gpmc_cov' in rec else "")
                 if IS_GP else ""), flush=True)
    print(f"augmented: {len(results) - n_bad}, mismatched: {n_bad}")
    sys.exit(1 if n_bad else 0)

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["seed"] for r in results}
    for t in range(start, start + n):
        seed = 1000 + t
        if seed in done:
            continue
        t0 = time.time()
        rec = {"seed": seed, "benchmark": bench, "surrogate": surr, "N": N,
               **({"design": DESIGN} if DESIGN != "lhs" else {}), **run(seed)}
        rec["runtime_s"] = round(time.time() - t0, 1)
        results.append(rec)
        json.dump(results, open(OUT, "w"))
        print(f"[{bench}/{surr}/N{N}] seed {seed} | nrmse={rec['nrmse']:.3g} "
              f"pw cov jk+={rec['pw_jkplus_cov']:.2f} cv={rec['pw_cvnorm_cov']:.2f}"
              + (f" gp={rec['pw_gp_cov']:.2f}" if IS_GP else "")
              + f" | CI cov plain={rec['ci_plain_cov']:.3f} "
              f"normal={rec['ci_normal_cov']:.3f} | {rec['runtime_s']}s", flush=True)
    print(f"total recorded: {len(results)}")
