"""
surrogates.py — single-surrogate baselines shared by the comparison scripts.

GP(spec, dim, pos_in)
    Gaussian process on standardised inputs and outputs (scikit-learn,
    L-BFGS-B, five random restarts, random_state=0). Specifications:
      orig      RBF, isotropic, amplitude FIXED at 1 (the original kernel)
      amp       RBF, isotropic, amplitude estimated
      ardnoamp  RBF, ARD, amplitude fixed
      ard       RBF, ARD, amplitude estimated
      m52       Matern-5/2, ARD, amplitude estimated
      quad      m52 + amplitude-scaled quadratic polynomial kernel
                (DotProduct**2): a GP with a quadratic trend whose
                coefficients are integrated out (universal kriging in kernel
                form)
      log       m52 fitted to log(y), with log-transformed inputs where
                pos_in is True; predictions exp(latent mean) = predictive
                median
      quadlog   quad on the log scale, as for log
SparsePCE(dim, bounds, log=False, pos_in=None)
    Sparse polynomial chaos expansion, Legendre basis on inputs mapped
    affinely to [-1, 1] (log-affinely for positive inputs when log=True),
    least-angle regression path with OLS refit on each active set (hybrid
    LAR) and selection of the active set and total degree by the corrected
    leave-one-out error (Blatman & Sudret, J. Comput. Phys. 2011).

Both expose fit(X, y), predict(X) (original scale) and latent(X) ->
(mean, sd) on the fitting scale (sd = None for SparsePCE), plus to_latent(y)
and from_latent(z) to move between scales.
"""

import os
import numpy as np
from numpy.polynomial import legendre
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, ConstantKernel,
                                              WhiteKernel, DotProduct)
from sklearn.linear_model import lars_path
from sklearn.preprocessing import StandardScaler


# Upper bound of the estimated kernel amplitude (standardised output units).
# 1e3 throughout the study; the environment variable GP_AMP_MAX is only for
# the sensitivity check of experiments/sensitivity_amp_bound.py.
AMP_MAX = float(os.environ.get("GP_AMP_MAX", "1e3"))


class GP:
    LOG_SPECS = {"log": "m52", "quadlog": "quad"}

    def __init__(self, spec, dim, pos_in=None):
        self.log = spec in self.LOG_SPECS
        self.spec = self.LOG_SPECS.get(spec, spec)
        self.d = dim
        self.pos_in = (np.zeros(dim, bool) if pos_in is None
                       else np.asarray(pos_in, bool))
        self.name = f"GP[{spec}]"

    # transformations ------------------------------------------------------
    def _tin(self, X):
        return (np.where(self.pos_in, np.log(np.abs(X) + 1e-300), X)
                if self.log else X)

    def to_latent(self, y):
        return np.log(y) if self.log else y

    def from_latent(self, z):
        return np.exp(z) if self.log else z

    # kernel ---------------------------------------------------------------
    def _kernel(self):
        ls = np.ones(self.d)
        hi = AMP_MAX
        m52 = lambda: (ConstantKernel(1.0, (1e-3, hi))
                       * Matern(ls, (1e-2, 1e3), nu=2.5))
        nug = lambda: WhiteKernel(1e-4, (1e-8, 1e-1))
        return {
            "orig":     lambda: RBF() + WhiteKernel(1e-2),
            "amp":      lambda: ConstantKernel(1.0, (1e-3, hi)) * RBF()
                                + WhiteKernel(1e-2),
            "ardnoamp": lambda: RBF(ls) + WhiteKernel(1e-2),
            "ard":      lambda: ConstantKernel(1.0, (1e-3, hi))
                                * RBF(ls, (1e-2, 1e3)) + nug(),
            "m52":      lambda: m52() + nug(),
            "quad":     lambda: m52()
                                + ConstantKernel(1e-2, (1e-6, 1e3))
                                * DotProduct(1.0, (1e-3, 1e3)) ** 2
                                + nug(),
        }[self.spec]()

    # fit / predict ----------------------------------------------------------
    def fit(self, X, y):
        if self.log:
            assert np.all(y > 0), "log variants need a positive output"
        self.sc = StandardScaler().fit(self._tin(X))
        self.gp = GaussianProcessRegressor(self._kernel(), normalize_y=True,
                                           n_restarts_optimizer=5,
                                           random_state=0)
        self.gp.fit(self.sc.transform(self._tin(X)), self.to_latent(y))
        return self

    def latent(self, X):
        return self.gp.predict(self.sc.transform(self._tin(X)), return_std=True)

    def predict(self, X, return_std=False):
        out = self.gp.predict(self.sc.transform(self._tin(X)), return_std=return_std)
        if self.log and not return_std:
            return np.exp(out)
        return out


class SparsePCE:
    def __init__(self, dim, bounds, log=False, pos_in=None, max_degree=5,
                 unit_map=None):
        self.d = dim
        self.unit_map = unit_map          # X -> (0,1)^d, e.g. the input CDFs
        self.b = np.asarray(bounds, float)
        self.log = log
        self.pos_in = (np.zeros(dim, bool) if pos_in is None
                       else np.asarray(pos_in, bool))
        self.max_degree = max_degree
        self.name = "SparsePCE" + ("[log]" if log else "")
        lo, hi = self.b[:, 0].copy(), self.b[:, 1].copy()
        if log:
            lo = np.where(self.pos_in, np.log(np.abs(lo) + 1e-300), lo)
            hi = np.where(self.pos_in, np.log(np.abs(hi) + 1e-300), hi)
        self._lo, self._hi = lo, hi

    def _unit(self, X):
        if self.unit_map is not None:     # probability-integral transform
            return 2 * np.clip(self.unit_map(X), 1e-12, 1 - 1e-12) - 1
        Z = (np.where(self.pos_in, np.log(np.abs(X) + 1e-300), X)
             if self.log else X)
        return 2 * (Z - self._lo) / (self._hi - self._lo) - 1

    def to_latent(self, y):
        return np.log(y) if self.log else y

    def from_latent(self, z):
        return np.exp(z) if self.log else z

    @staticmethod
    def _total_degree(d, p):
        """All multi-indices of length d with total degree <= p, in the
        lexicographic order of itertools.product (first index slowest)."""
        if d == 1:
            return [(k,) for k in range(p + 1)]
        return [(k,) + rest for k in range(p + 1)
                for rest in SparsePCE._total_degree(d - 1, p - k)]

    @staticmethod
    def _indices(d, p, q):
        """Multi-indices with hyperbolic (q-)norm <= p, constant excluded.
        Enumerates only the total-degree set (C(d+p, p) indices), not the
        full (p+1)^d grid, so it scales to many inputs; the order, and hence
        every fit, is the same as filtering itertools.product."""
        out = []
        for a in SparsePCE._total_degree(d, p):
            if sum(a) == 0:
                continue
            if q < 1 and sum(ai ** q for ai in a) ** (1 / q) > p + 1e-9:
                continue
            out.append(a)
        return out

    @staticmethod
    def _legendre(u, k):
        c = np.zeros(k + 1); c[k] = 1
        return legendre.legval(u, c) * np.sqrt(2 * k + 1)   # orthonormal on U(-1,1)

    def _design(self, U, idx):
        cache = {}
        cols = []
        for a in idx:
            col = np.ones(len(U))
            for j, k in enumerate(a):
                if k:
                    key = (j, k)
                    if key not in cache:
                        cache[key] = self._legendre(U[:, j], k)
                    col = col * cache[key]
            cols.append(col)
        return np.column_stack(cols) if cols else np.zeros((len(U), 0))

    @staticmethod
    def _loo(Psi, z):
        """OLS with intercept; corrected LOO error (Blatman & Sudret 2011)."""
        n, P = Psi.shape[0], Psi.shape[1] + 1
        A = np.column_stack([np.ones(n), Psi])
        try:
            C = A.T @ A
            Cinv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            return np.inf, None
        coef = Cinv @ A.T @ z
        h = np.einsum("ij,jk,ik->i", A, Cinv, A)
        r = (z - A @ coef) / np.clip(1 - h, 1e-12, None)
        err = np.mean(r ** 2) / max(np.var(z), 1e-300)
        T = n / (n - P) * (1 + np.trace(np.linalg.inv(C / n)) / n)
        return err * T, coef

    def fit(self, X, y):
        U = self._unit(X)
        z = self.to_latent(y)
        n = len(z)
        best = (np.inf, None, None)
        for p in range(1, self.max_degree + 1):
            q = 1.0 if p <= 2 else 0.75
            idx = self._indices(self.d, p, q)
            if not idx or len(idx) > 3000:
                continue
            Psi = self._design(U, idx)
            _, active_path, _ = lars_path(Psi - Psi.mean(0), z - z.mean(),
                                          method="lar", max_iter=n - 3)
            seen = []
            for j in active_path:
                if j in seen:
                    continue
                seen.append(j)
                if len(seen) > n - 3:
                    break
                cols = list(seen)
                e, coef = self._loo(Psi[:, cols], z)
                if e < best[0]:
                    best = (e, [idx[c] for c in cols], coef)
        if best[1] is None:                       # constant model
            self.idx, self.coef = [], np.array([z.mean()])
        else:
            _, self.idx, self.coef = best
        self.loo_error = best[0]
        return self

    def latent(self, X):
        Psi = self._design(self._unit(X), self.idx)
        return self.coef[0] + Psi @ self.coef[1:], None

    def predict(self, X):
        return self.from_latent(self.latent(X)[0])
