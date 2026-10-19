"""End-to-end test of src/procedure.py on a cheap model with a known answer."""
import numpy as np
from benchmarks import BENCHMARKS
from framework_core import lhs
from procedure import evaluate


def test_procedure_on_borehole_is_close_to_reference():
    b = BENCHMARKS["borehole"]()
    np.random.seed(1000)
    X = b.ppf(lhs(b.dim, 20)); y = b.evaluate(X)
    sample = lambda n, rng: b.ppf(rng.uniform(0, 1, (n, b.dim)))
    res = evaluate(X, y, sample, bounds=b.bounds, positive=True, pos_in=b.pos_in,
                   M=20_000, n_sobol=5000, seed=0)
    assert res["chosen"] in ("gp_log", "gp_quadlog")            # log scale wins here
    yr = b.evaluate(b.ppf(np.random.default_rng(8).uniform(0, 1, (200_000, b.dim))))
    w = res["with_surrogate_contribution"]
    assert abs(w["mean"] / yr.mean() - 1) < 0.02
    assert abs(w["sd"] / yr.std() - 1) < 0.05
    for k, (lo, hi) in res["jackknife_95"].items():               # centred on the
        assert lo <= res["surrogate_as_exact"][k] <= hi           # estimate without E
    shares = res["budget"]
    assert 0 <= shares["surrogate_share"] < 0.05
    assert abs(sum(shares["input_share"]) + shares["surrogate_share"]
               + shares["interaction_share"] - 1) < 0.05


def test_adaptive_mc_and_gum_framework_agree_on_a_linear_model():
    """For a linear model the law of propagation of uncertainty is exact, so
    the adaptive Monte Carlo result must agree with it to the tolerance."""
    rng = np.random.default_rng(3)
    X = 1 + rng.uniform(0, 1, (20, 2))
    y = 2 * X[:, 0] + 3 * X[:, 1]
    sample = lambda n, r: 1 + r.uniform(0, 1, (n, 2))
    res = evaluate(X, y, sample, bounds=[(1, 2), (1, 2)], M=10_000, n_sobol=2000,
                   seed=0, ndig=2)
    a, g, w = res["adaptive_mc"], res["gum_framework"], res["with_surrogate_contribution"]
    assert a["met"] and a["batches"] >= 2 and res["M"] == a["batches"] * a["batch_size"]
    u_true = np.sqrt((4 + 9) / 12)
    assert abs(g["u"] / u_true - 1) < 0.01 and abs(g["y"] - 7.5) < 0.01
    assert abs(w["sd"] - g["u"]) < 2 * a["delta"] + 0.01
