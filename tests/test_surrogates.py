"""Unit tests for src/surrogates.py and src/procedure.py helpers."""
import itertools
import numpy as np
import pytest
from surrogates import GP, SparsePCE
from procedure import jackknife_plus


def _old_indices(d, p, q):
    out = []
    for a in itertools.product(range(p + 1), repeat=d):
        if sum(a) == 0 or sum(a) > p:
            continue
        if q < 1 and sum(ai ** q for ai in a) ** (1 / q) > p + 1e-9:
            continue
        out.append(a)
    return out


@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("p", [1, 2, 3, 4])
def test_pce_indices_same_order_as_grid_filter(d, p):
    q = 1.0 if p <= 2 else 0.75
    assert SparsePCE._indices(d, p, q) == _old_indices(d, p, q)


def test_pce_indices_scale_to_20_inputs():
    assert len(SparsePCE._indices(20, 2, 1.0)) == 230          # C(22, 2) - 1


def test_sparse_pce_recovers_a_quadratic_exactly():
    rng = np.random.default_rng(0)
    b = [(-1.0, 2.0), (0.0, 1.0), (-3.0, 3.0)]
    X = np.column_stack([rng.uniform(lo, hi, 25) for lo, hi in b])
    f = lambda X: 1 + 2 * X[:, 0] - X[:, 1] ** 2 + 0.5 * X[:, 0] * X[:, 2]
    s = SparsePCE(3, b).fit(X, f(X))
    Xt = np.column_stack([rng.uniform(lo, hi, 500) for lo, hi in b])
    assert np.max(np.abs(s.predict(Xt) - f(Xt))) < 1e-8


def test_log_gp_predicts_positive_median():
    rng = np.random.default_rng(1)
    X = rng.uniform(1, 2, (15, 2))
    y = np.exp(X[:, 0]) * X[:, 1] ** 2
    g = GP("log", 2, np.array([True, True])).fit(X, y)
    Xt = rng.uniform(1, 2, (50, 2))
    m, _ = g.latent(Xt)
    assert np.all(g.predict(Xt) > 0)
    assert np.allclose(g.predict(Xt), np.exp(m))


def test_quad_gp_reproduces_a_quadratic_bowl():
    rng = np.random.default_rng(2)
    X = rng.uniform(-200, 200, (20, 3))
    f = lambda X: 0.2 * np.sum(X ** 2, axis=1)
    g = GP("quad", 3).fit(X, f(X))
    Xt = rng.uniform(-200, 200, (200, 3))
    assert np.sqrt(np.mean((g.predict(Xt) - f(Xt)) ** 2)) / np.std(f(Xt)) < 1e-2


def test_jackknife_plus_uses_extreme_order_statistics_for_n20():
    rng = np.random.default_rng(3)
    mu = rng.normal(size=(20, 4)); r = rng.normal(size=20)
    lo, hi = jackknife_plus(mu, r)
    assert np.allclose(lo, np.min(mu - np.abs(r)[:, None], axis=0))
    assert np.allclose(hi, np.max(mu + np.abs(r)[:, None], axis=0))
