"""
gp_bayes.py — predictive distributions of a fitted surrogates.GP that do not
plug in the maximum-likelihood hyperparameters, for the coverage study
(research plan T1.2: is the GP's overconfidence a plug-in artefact?).

student_t(s, Xs)
    Hyperparameters of the correlation at their ML values, but the overall
    variance scale and a constant mean integrated out under the reference
    prior p(beta, sigma^2) ~ 1/sigma^2 (the standard Bayesian emulator,
    O'Hagan 2006): a Student-t process with nu = N - 1 degrees of freedom,
    GLS mean m*(x) and scale^2 S^2 c*(x, x'), with
      c*(x, x') = k(x, x') - k(x)' K^-1 k(x')
                  + (1 - 1'K^-1 k(x)) (1 - 1'K^-1 k(x')) / (1'K^-1 1),
      S^2 = (y - b)' K^-1 (y - b) / (N - 1),  b the GLS mean,
    where k is the fitted kernel (all its amplitudes treated as one scale).
mcmc_thetas(s, rs)
    Posterior draws of all kernel hyperparameters (log scale), uniform prior
    on each log-hyperparameter within the bounds of the ML fit, sampled with
    the affine-invariant ensemble sampler (emcee, Foreman-Mackey et al. 2013).
predict_draws(s, thetas, Xs)
    Latent predictive mean and SD for every draw (mixture predictive).
paths(...)
    Joint posterior sample paths for derived quantities.

All latent quantities are on the GP's fitting scale (log scale for the log
variants), in the original units of that scale (not normalised).
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _parts(s):
    g = s.gp
    return g, g.X_train_, g.y_train_, float(g._y_train_mean), float(g._y_train_std)


def _xs(s, X):
    return s.sc.transform(s._tin(X))


def student_t(s, X, full_cov=False):
    """Latent t-process at X: (mean, scale [sd or cov], nu)."""
    g, Xt, y, ym, ys = _parts(s)
    K = g.kernel_(Xt) + g.alpha * np.eye(len(Xt))
    cf = cho_factor(K, lower=True)
    one = np.ones(len(Xt))
    Ki1, Kiy = cho_solve(cf, one), cho_solve(cf, y)
    b = (one @ Kiy) / (one @ Ki1)
    r = y - b
    nu = len(Xt) - 1
    S2 = (r @ cho_solve(cf, r)) / nu
    Xs = _xs(s, X)
    k = g.kernel_(Xs, Xt)
    mean = b + k @ cho_solve(cf, r)
    h = 1.0 - k @ Ki1
    V = cho_solve(cf, k.T)
    if full_cov:
        c = g.kernel_(Xs) - k @ V + np.outer(h, h) / (one @ Ki1)
        return mean * ys + ym, S2 * c * ys ** 2, nu
    c = g.kernel_.diag(Xs) - np.einsum("ij,ji->i", k, V) + h ** 2 / (one @ Ki1)
    return mean * ys + ym, np.sqrt(np.maximum(S2 * c, 0.0)) * ys, nu


def mcmc_thetas(s, rs, n_draws=200, n_steps=800, burn=300):
    """Posterior draws of the log-hyperparameters; returns (thetas, mean
    acceptance fraction, largest integrated autocorrelation time in steps)."""
    import emcee
    g = s.gp
    th0 = g.kernel_.theta
    lo, hi = g.kernel_.bounds.T
    dim = len(th0)

    def logp(th):
        if np.any(th < lo) or np.any(th > hi):
            return -np.inf
        v = g.log_marginal_likelihood(th)
        return v if np.isfinite(v) else -np.inf

    nw = max(24, 2 * dim + 2)
    p0 = np.clip(th0 + 1e-3 * rs.standard_normal((nw, dim)), lo + 1e-9, hi - 1e-9)
    sampler = emcee.EnsembleSampler(nw, dim, logp)
    sampler.random_state = np.random.RandomState(rs.randint(2 ** 31)).get_state()
    sampler.run_mcmc(p0, n_steps, progress=False)
    chain = sampler.get_chain(discard=burn, flat=True)
    idx = np.linspace(0, len(chain) - 1, n_draws).astype(int)
    tau = float(np.max(sampler.get_autocorr_time(discard=burn, tol=0)))
    return chain[idx], float(np.mean(sampler.acceptance_fraction)), tau


def predict_draws(s, thetas, X, full_cov_points=None):
    """Latent mean and SD (n_draws, m) at X for each hyperparameter draw."""
    g, Xt, y, ym, ys = _parts(s)
    Xs = _xs(s, X)
    mus, sds = [], []
    for th in thetas:
        k = g.kernel_.clone_with_theta(th)
        cf = cho_factor(k(Xt) + g.alpha * np.eye(len(Xt)), lower=True)
        ks = k(Xs, Xt)
        mus.append(ks @ cho_solve(cf, y))
        var = k.diag(Xs) - np.einsum("ij,ji->i", ks, cho_solve(cf, ks.T))
        sds.append(np.sqrt(np.maximum(var, 0.0)))
    return np.array(mus) * ys + ym, np.array(sds) * ys


def gp_cov(s, X, theta=None):
    """Latent posterior mean and covariance at X (plug-in, or for one draw)."""
    g, Xt, y, ym, ys = _parts(s)
    k = g.kernel_ if theta is None else g.kernel_.clone_with_theta(theta)
    cf = cho_factor(k(Xt) + g.alpha * np.eye(len(Xt)), lower=True)
    Xs = _xs(s, X)
    ks = k(Xs, Xt)
    return (ks @ cho_solve(cf, y)) * ys + ym, (k(Xs) - ks @ cho_solve(cf, ks.T)) * ys ** 2


def mvn_paths(mean, C, n, rs):
    """n joint Gaussian sample paths (n, m), with increasing jitter if needed."""
    jit = 1e-10 * max(np.mean(np.diag(C)), 1e-300)
    for _ in range(12):
        try:
            L = np.linalg.cholesky(C + jit * np.eye(len(C)))
            break
        except np.linalg.LinAlgError:
            jit *= 10
    return mean[None, :] + (L @ rs.standard_normal((len(C), n))).T


def imse_reduction(s, cand, X_int):
    """Reduction of the integrated posterior variance over X_int if each
    candidate were added (plug-in hyperparameters, latent scale):
    sum_j cov(x_j, c)^2 / var(c)."""
    g, Xt = s.gp, s.gp.X_train_
    cf = cho_factor(g.kernel_(Xt) + g.alpha * np.eye(len(Xt)), lower=True)
    A, C = _xs(s, X_int), _xs(s, cand)
    kc = g.kernel_(Xt, C)
    cross = g.kernel_(A, C) - g.kernel_(A, Xt) @ cho_solve(cf, kc)
    var_c = g.kernel_.diag(C) - np.einsum("ij,ij->j", kc, cho_solve(cf, kc))
    return np.sum(cross ** 2, axis=0) / np.maximum(var_c, 1e-300)
