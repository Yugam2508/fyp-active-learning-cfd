"""Unit tests for src/benchmarks.py (formulas are cross-checked against
UQTestFuns in experiments/check_benchmarks.py)."""
import numpy as np
import pytest
from benchmarks import BENCHMARKS

ANALYTICAL = [k for k in BENCHMARKS if k != "cfd"]


@pytest.mark.parametrize("name", ANALYTICAL)
def test_ppf_cdf_are_inverse(name):
    b = BENCHMARKS[name]()
    U = np.random.default_rng(0).uniform(0.02, 0.98, (200, b.dim))
    assert np.allclose(b.cdf(b.ppf(U)), U, atol=1e-8)


@pytest.mark.parametrize("name", ANALYTICAL)
def test_positive_models_give_positive_outputs(name):
    b = BENCHMARKS[name]()
    y = b.evaluate(b.ppf(np.random.default_rng(1).uniform(0, 1, (2000, b.dim))))
    assert np.all(np.isfinite(y))
    if b.positive:
        assert np.all(y > 0)


def test_otl20_ignores_inert_inputs():
    b = BENCHMARKS["otl20"]()
    X = b.ppf(np.random.default_rng(2).uniform(0, 1, (100, 20)))
    X2 = X.copy(); X2[:, 6:] = b.ppf(np.full((100, 20), 0.5))[:, 6:]
    assert np.array_equal(b.evaluate(X), b.evaluate(X2))


def test_bump_feature_size_and_variance_share():
    b = BENCHMARKS["bump"]()
    U = np.random.default_rng(3).uniform(0, 1, (400_000, 4))
    y = b.evaluate(U)
    bump = b.A * np.exp(-np.sum((U - b.C) ** 2, axis=1) / (2 * b.W ** 2))
    assert 0.08 < bump.var() / y.var() < 0.12           # about 10 % of Var(Y)
    assert 0.025 < b.in_bump(U).mean() < 0.035          # about 3 % of the volume


def test_borehole_corr_has_the_copula_correlation_and_gum_marginals():
    b, g = BENCHMARKS["borehole_corr"](), BENCHMARKS["borehole_gum"]()
    U = np.random.default_rng(4).uniform(0, 1, (200_000, 8))
    X = b.ppf(U)
    target = 6 / np.pi * np.arcsin(0.5 / 2)             # Pearson, uniform marginals
    assert abs(np.corrcoef(X[:, 2], X[:, 4])[0, 1] - target) < 0.01
    assert abs(np.corrcoef(X[:, 3], X[:, 5])[0, 1] - target) < 0.01
    assert abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1]) < 0.01
    for j in range(8):                                  # marginals unchanged
        assert np.allclose(np.quantile(X[:, j], [0.1, 0.5, 0.9]),
                           np.quantile(g.ppf(U)[:, j], [0.1, 0.5, 0.9]), rtol=0.02)
