"""
campaign_vector.py — research plan T2.4: the surrogate contribution for
several output quantities (GUM Supplement 2).

Model: the plume model (uniform inputs Q, u, H, x, y as in benchmarks.py)
evaluated at three receptors on the plume axis, at x, x + 200 m and
x + 400 m: a vector output with strongly correlated components. Each
component gets its own log-scale GP (gp_log); LOO refits give an N x 3
matrix of latent residuals. GUM S2 propagation of 3e4 draws
(RandomState(9)) with the surrogate error E added on the latent scale as
  plain    E = 0
  joint    E = the residual vector of one run, drawn with equal
           probabilities (keeps the correlation between components)
  indep    each component's residual drawn independently
  normal   independent normal errors with the LOO RMS of each component
Coverage region: the GUM S2 hyperellipsoidal region from the propagated
sample (mean, covariance, k from the 95 % quantile of the Mahalanobis
distances of the sample), on the original scale and on the log scale.
Recorded: its achieved coverage under the reference (2e5 true-model draws,
RandomState(8)), its volume relative to the region of the reference, and
the correlation of the LOO residuals.

Usage: python campaign_vector.py [start n] [N]
Results: results/results_vector_plume[_N<N>].json (resumable)
"""

import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
import numpy as np
from framework_core import lhs
from benchmarks import BENCHMARKS
from surrogates import GP

start, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 30)
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20
plume = BENCHMARKS["plume"]()
d, DX = plume.dim, np.array([0.0, 200.0, 400.0])
OUT = os.path.join(RESULTS_DIR, "results_vector_plume" + ("" if N == 20 else f"_N{N}") + ".json")


def f(X):
    return np.column_stack([plume.evaluate(X + np.array([0, 0, 0, dx, 0])) for dx in DX])


fixed = lambda m, rs: plume.ppf(rs.uniform(0, 1, (m, d)))
X_mc = fixed(30_000, np.random.RandomState(9))
Y_ref = f(fixed(200_000, np.random.RandomState(8)))


def region(S):
    c = S.mean(0)
    Ui = np.linalg.inv(np.cov(S.T))
    d2 = np.einsum("ij,jk,ik->i", S - c, Ui, S - c)
    return c, Ui, np.quantile(d2, 0.95)


def assess(S, R):
    """Coverage of the region of sample S for reference sample R, and its
    volume relative to the region of R itself."""
    c, Ui, k2 = region(S)
    d2 = np.einsum("ij,jk,ik->i", R - c, Ui, R - c)
    _, Ui_r, k2_r = region(R)
    vol = np.sqrt(np.linalg.det(Ui_r) / np.linalg.det(Ui)) * (k2 / k2_r) ** 1.5
    return float(np.mean(d2 <= k2)), float(vol)


def run(seed):
    np.random.seed(seed)
    X = plume.ppf(lhs(d, N))
    Y = f(X)
    rs = np.random.RandomState(seed + 10 ** 6)
    mu, R = np.empty((len(X_mc), 3)), np.empty((N, 3))
    gps = []
    for k in range(3):
        g = GP("log", d, plume.pos_in).fit(X, Y[:, k])
        gps.append(g)
        mu[:, k] = g.latent(X_mc)[0]
        z = g.to_latent(Y[:, k])
        for i in range(N):
            keep = np.arange(N) != i
            gi = GP("log", d, plume.pos_in).fit(X[keep], Y[keep, k])
            R[i, k] = z[i] - gi.latent(X[i:i + 1])[0][0]
    M = len(X_mc)
    E = {"plain": np.zeros((M, 3)),
         "joint": R[rs.randint(0, N, M)],
         "indep": np.column_stack([R[rs.randint(0, N, M), k] for k in range(3)]),
         "normal": rs.standard_normal((M, 3)) * np.sqrt(np.mean(R ** 2, axis=0))}
    out = {"loo_corr": np.corrcoef(R.T)[np.triu_indices(3, 1)].tolist(),
           "true_corr_log": np.corrcoef(np.log(Y_ref.T))[np.triu_indices(3, 1)].tolist()}
    for tag, e in E.items():
        S = np.exp(mu + e)
        for scale, tr in (("orig", lambda v: v), ("log", np.log)):
            cov, vol = assess(tr(S), tr(Y_ref))
            out[f"{scale}_{tag}_cov"], out[f"{scale}_{tag}_vol"] = cov, vol
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
        rec = {"seed": seed, "N": N, **run(seed)}
        rec["runtime_s"] = round(time.time() - t0, 1)
        results.append(rec)
        json.dump(results, open(OUT, "w"))
        print(f"[vector/N{N}] seed {seed} | log-scale coverage plain {rec['log_plain_cov']:.3f} "
              f"joint {rec['log_joint_cov']:.3f} indep {rec['log_indep_cov']:.3f} "
              f"normal {rec['log_normal_cov']:.3f} | {rec['runtime_s']}s", flush=True)
    print(f"total recorded: {len(results)}")
