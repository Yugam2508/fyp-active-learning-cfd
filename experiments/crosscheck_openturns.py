"""
crosscheck_openturns.py — research plan T1.3: are our sparse PCE and GP as
accurate as an independent, established implementation?

On the same 30 Latin hypercube designs as campaign_generalisation.py (seeds
1000-1029, N = 20 runs) and the same 5000 test points (RandomState(7)),
fits with OpenTURNS:
  ot_pce      sparse PCE: Legendre basis on the uniform inputs, total degree
              1-5 with hyperbolic truncation q = 0.75 above degree 2 (as in
              src/surrogates.py), least-angle regression with corrected LOO
              selection within each degree (LARS + CorrectedLeaveOneOut), the
              degree chosen by the corrected LOO error of the selected basis
  ot_gp       GP (GaussianProcessFitter): Matern-5/2, one scale per input,
              estimated amplitude and nugget factor, constant trend
  ot_gp_quad  the same with a full quadratic trend basis (universal kriging;
              1 + d + d(d+1)/2 coefficients, estimated by generalised least
              squares)
  ot_pck      PC-Kriging (Schoebi et al. 2015), research plan T2.3: the same
              GP with the basis selected by ot_pce as trend
GPs are fitted on the inputs mapped affinely to [0, 1]^d, scales in [0.01,
100], 10 Latin hypercube starts (TNC). Records the nRMSE of each, to be
paired with gp_m52, gp_quad and spce of the main study.

Usage: python crosscheck_openturns.py <benchmark> [start n]   (uniform-input
models only). Results: results/results_ot_<benchmark>.json (resumable).
"""

import os, sys, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
import numpy as np
import openturns as ot
from framework_core import lhs
from benchmarks import BENCHMARKS
from surrogates import SparsePCE

ot.Log.Show(ot.Log.NONE)
bench = sys.argv[1]
start, n = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 30)
N = 20
problem = BENCHMARKS[bench]()
assert problem.marginals is None, "uniform-input models only"
d = problem.dim
b = np.array(problem.bounds, float)
X_test = problem.ppf(np.random.RandomState(7).uniform(0, 1, (5000, d)))
y_test = problem.evaluate(X_test)
REF_SD = float(problem.evaluate(problem.ppf(
    np.random.RandomState(8).uniform(0, 1, (200_000, d)))).std())
OUT = os.path.join(RESULTS_DIR, f"results_ot_{bench}.json")


def nrmse(pred):
    return float(np.sqrt(np.mean((y_test - pred) ** 2)) / REF_SD)


def ot_pce(X, y):
    dist = ot.JointDistribution([ot.Uniform(float(lo), float(hi)) for lo, hi in b])
    best = (np.inf, None, None, None)
    for p in range(1, 6):
        enum = (ot.LinearEnumerateFunction(d) if p <= 2
                else ot.HyperbolicAnisotropicEnumerateFunction(d, 0.75))
        basis = ot.OrthogonalProductPolynomialFactory([ot.LegendreFactory()] * d, enum)
        P = enum.getBasisSizeFromTotalDegree(p)
        if P > 3000:
            continue
        algo = ot.FunctionalChaosAlgorithm(
            ot.Sample(X), ot.Sample(y[:, None]), dist, ot.FixedStrategy(basis, P),
            ot.LeastSquaresStrategy(ot.LeastSquaresMetaModelSelectionFactory(
                ot.LARS(), ot.CorrectedLeaveOneOut())))
        algo.run()
        res = algo.getResult()
        # corrected LOO error of the selected basis (same formula as ours)
        T = res.getTransformation()
        funcs = [ot.ComposedFunction(f, T) for f in res.getReducedBasis()]
        Psi = np.column_stack([np.array(f(ot.Sample(X))).ravel() for f in funcs])
        keep = np.std(Psi, axis=0) > 1e-12                    # drop the constant
        e, _ = SparsePCE._loo(Psi[:, keep], y)
        if e < best[0]:
            best = (e, res.getMetaModel(), p, funcs)
    return best[1], best[2], best[3]


def ot_gp(X, y, basis):
    """GP with Matern-5/2 ARD covariance and the given trend basis, fitted on
    the inputs mapped to [0, 1]^d (basis functions defined on [0, 1]^d)."""
    span = b[:, 1] - b[:, 0]
    Xs, Xt = (X - b[:, 0]) / span, (X_test - b[:, 0]) / span
    cov = ot.MaternModel([0.5] * d, [float(np.std(y))], 2.5)
    cov.activateNuggetFactor(True)
    cov.setNuggetFactor(1e-6)
    fitter = ot.GaussianProcessFitter(ot.Sample(Xs), ot.Sample(y[:, None]), cov, basis)
    # optimised: the d scales and the nugget factor (amplitude profiled out)
    sc = np.array(ot.LHSExperiment(ot.JointDistribution([ot.Uniform(0.05, 2.0)] * d),
                                   10).generate())
    starts = ot.Sample(np.column_stack([sc, np.full(len(sc), 1e-6)]))
    fitter.setOptimizationAlgorithm(ot.MultiStart(ot.TNC(), starts))
    fitter.setOptimizationBounds(ot.Interval([0.01] * d + [1e-12], [100.0] * d + [0.1]))
    fitter.run()
    reg = ot.GaussianProcessRegression(fitter.getResult())
    reg.run()
    return np.array(reg.getResult().getMetaModel()(ot.Sample(Xt))).ravel()


def on_unit(funcs):
    """Basis functions of x composed with x = lo + u (hi - lo), u in [0, 1]^d."""
    aff = ot.LinearFunction([0.0] * d, b[:, 0].tolist(),
                            ot.Matrix(np.diag(b[:, 1] - b[:, 0]).tolist()))
    return ot.Basis([ot.ComposedFunction(f, aff) for f in funcs])


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {r["seed"] for r in results}
    for t in range(start, start + n):
        seed = 1000 + t
        if seed in done:
            continue
        t0 = time.time()
        np.random.seed(seed)
        X = problem.ppf(lhs(d, N))
        y = problem.noisy(X)
        ot.RandomGenerator.SetSeed(seed)
        rec = {"seed": seed, "benchmark": bench, "N": N}
        mm, p, funcs = ot_pce(X, y)
        rec.update({"ot_pce_nrmse": nrmse(np.array(mm(ot.Sample(X_test))).ravel()),
                    "ot_pce_degree": p, "ot_pck_n_basis": len(funcs)})
        bases = {"ot_gp": ot.ConstantBasisFactory(d).build(),
                 "ot_gp_quad": ot.QuadraticBasisFactory(d).build(),
                 "ot_pck": on_unit(funcs)}
        for tag, basis in bases.items():
            try:
                rec[f"{tag}_nrmse"] = nrmse(ot_gp(X, y, basis))
            except Exception as exc:                       # record, do not hide
                rec[f"{tag}_nrmse"] = None
                rec[f"{tag}_error"] = str(exc)[:200]
        rec["runtime_s"] = round(time.time() - t0, 1)
        results.append(rec)
        json.dump(results, open(OUT, "w"))
        print(f"[{bench}] seed {seed} | pce {rec['ot_pce_nrmse']:.4g} (p={p}) "
              f"gp {rec['ot_gp_nrmse']} quad {rec['ot_gp_quad_nrmse']} pck {rec['ot_pck_nrmse']}"
              f" | {rec['runtime_s']}s", flush=True)
    print(f"total recorded: {len(results)}")
