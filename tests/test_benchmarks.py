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
