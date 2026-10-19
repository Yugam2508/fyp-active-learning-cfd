"""Unit tests for src/designs.py and src/gp_bayes.py."""
import numpy as np
from scipy.spatial.distance import pdist
from framework_core import lhs
from designs import make_design
from surrogates import GP
import gp_bayes


def test_lhs_design_is_the_main_study_design():
    np.random.seed(5); a = make_design("lhs", 3, 20)
    np.random.seed(5); b = lhs(3, 20)
    assert np.array_equal(a, b)


def test_maximin_spreads_points_more_than_random_lhs():
    gains = []
    for s in range(5):
        np.random.seed(s); m = make_design("maximin", 4, 20)
        np.random.seed(s); r = lhs(4, 20)
        gains.append(pdist(m).min() / pdist(r).min())
    assert np.median(gains) > 1.2
    np.random.seed(0); S = make_design("sobol", 4, 20)
    assert S.shape == (20, 4) and S.min() >= 0 and S.max() <= 1


def _fit():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (15, 2)); y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2
    return GP("m52", 2).fit(X, y), rng.uniform(0, 1, (50, 2))


def test_student_t_is_close_to_plug_in_but_wider():
    g, Xs = _fit()
    mt, st, nu = gp_bayes.student_t(g, Xs)
    mu, sd = g.latent(Xs)
    assert nu == 14
    assert np.max(np.abs(mt - mu)) < 0.05 * np.std(mu)
    assert np.median(st / sd) > 0.9
    m2, C, _ = gp_bayes.student_t(g, Xs[:5], full_cov=True)
    assert np.allclose(np.sqrt(np.diag(C)), st[:5], rtol=1e-6)


def test_plug_in_covariance_matches_scikit_learn():
    g, Xs = _fit()
    m, C = gp_bayes.gp_cov(g, Xs)
    mu, sd = g.latent(Xs)
    assert np.allclose(m, mu, atol=1e-8) and np.allclose(np.sqrt(np.diag(C)), sd, rtol=1e-6)


def test_mcmc_draws_lie_within_the_bounds_and_imse_is_nonnegative():
    g, Xs = _fit()
    th, acc, tau = gp_bayes.mcmc_thetas(g, np.random.RandomState(0), n_draws=20,
                                        n_steps=60, burn=20)
    lo, hi = g.gp.kernel_.bounds.T
    assert th.shape[0] == 20 and np.all(th >= lo) and np.all(th <= hi) and 0 < acc < 1
    mus, sds = gp_bayes.predict_draws(g, th, Xs)
    assert mus.shape == (20, 50) and np.all(sds >= 0)
    assert np.all(gp_bayes.imse_reduction(g, Xs, Xs) >= -1e-12)
