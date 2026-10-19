"""
benchmarks.py — test models for the comparison studies.

Each class exposes the interface framework_core.ActiveLearner expects:
  dim, bounds (list of (lo, hi)), evaluate(X), noisy(X)
plus
  name         key in BENCHMARKS
  lower_bound  known physical lower bound of the output (soft physics
               penalty), or None
  positive     True if the output is strictly positive (log variants apply)
  pos_in       per input, True if its support is strictly positive
  marginals    None (independent uniform inputs over `bounds`) or a list of
               frozen scipy.stats-like distributions with ppf/cdf
  ppf(U)       maps U in (0,1)^d to inputs (inverse CDF per input)
  cdf(X)       maps inputs to (0,1)^d
  threshold, tail
               fixed physical threshold for an exceedance probability,
               P(Y > threshold) for tail "upper", P(Y < threshold) for
               "lower"; None means the reference 95 % (upper) or 5 % (lower)
               quantile is used
For uniform inputs ppf(U) = lo + U (hi - lo), arithmetically identical to the
mapping used by the earlier campaigns, so their records stay reproducible.
The models other than physics are deterministic, so noisy() adds no noise.

Models (uniform inputs unless stated)
------
physics       sin 2x1 cos 2x2 + exp(-x3^2) + 0.2||x||^2 on [-200,200]^3, noise
              SD 0.1 (benchmark of the earlier analysis; its dominant
              quadratic term lies in the degree-2 PCE class).
plume         Gaussian plume ground-level concentration of a continuous ammonia
              release (mg m^-3), Briggs rural coefficients, neutral stability,
              ground reflection:
                C = 1e3 Q / (2 pi u sy sz) exp(-y^2/(2 sy^2)) 2 exp(-H^2/(2 sz^2))
                sy = 0.08 x (1+1e-4 x)^-1/2,  sz = 0.06 x (1+1.5e-3 x)^-1/2
              Q 100-2000 g/s, u 1-6 m/s, H 1-20 m, x 200-1000 m, y -30-30 m.
              Threshold: IDLH for ammonia, 210 mg m^-3 (about 300 ppm).
borehole      Water flow through a borehole (m^3/yr), 8 inputs, ranges of the
              Surjanovic & Bingham library (Morris et al. 1993).
ishigami      sin x1 + 7 sin^2 x2 + 0.1 x3^4 sin x1 on [-pi, pi]^3.
otl           Mid-point voltage (V) of an output-transformerless push-pull
              circuit, 6 inputs (Ben-Ari & Steinberg 2007).
piston        Cycle time (s) of a piston, 7 inputs (Ben-Ari & Steinberg 2007).
wingweight    Weight (lb) of a light-aircraft wing, 10 inputs (Forrester et
              al. 2008); the sweep angle (degrees) can be negative.
otl20         OTL circuit with 14 additional inert inputs U(100, 200), 20
              inputs (Moon 2010): tests surrogates on irrelevant inputs.
bump          x1 + x2^2 + 0.5 sin(2 pi x3) + 0.5 x4
              + 3.6 exp(-||x - 0.7||^2 / (2 0.13^2)) on [0, 1]^4: a local
              feature that the runs can miss (failure-mode test).
Non-uniform inputs
------------------
borehole_gum  borehole with the probabilistic input model of Harper & Gupta
              (1983) as implemented in UQTestFuns v0.7: rw ~ N(0.10,
              0.0161812), ln r ~ N(7.71, 1.0056), Hu ~ U(990, 1100),
              Kw ~ U(9985, 12045), other inputs uniform as above.
borehole_corr borehole_gum with a Gaussian copula: correlation 0.5 between Tu
              and Tl and between Hu and Hl (dependent inputs).
plume_gum     plume with illustrative GUM-style inputs: Q lognormal (median
              500 g/s, geometric SD 2), u Weibull (shape 2, scale 4 m/s)
              truncated to [0.5, 15] m/s, H ~ U(1, 20) m, x ~ U(200, 1000) m,
              y ~ N(0, 15) m.
flood         Maximal annual underflow (m) of a river (Iooss & Lemaitre 2015),
              8 inputs with truncated Gumbel, truncated normal, triangular and
              uniform distributions. Flooding (underflow < 0) has a probability
              below 1e-5 for this input model, so the exceedance assessment
              uses the alert level "underflow below its reference 5 %
              quantile": threshold None, tail "lower".
cfd           OpenFOAM RANS dispersion behind a building, 6 inputs; see
              src/cfd_benchmark.py and src/cfd_dispersion.py.
Formulas and input models were cross-checked numerically against
UQTestFuns v0.7.0 (tests in experiments/check_benchmarks.py).
"""

import numpy as np
from scipy import stats
from framework_core import ToyProblem


class Truncated:
    """Distribution truncated to [a, b] by inverse-CDF transformation."""

    def __init__(self, dist, a=-np.inf, b=np.inf):
        self.dist = dist
        self.Fa = dist.cdf(a) if np.isfinite(a) else 0.0
        self.Fb = dist.cdf(b) if np.isfinite(b) else 1.0
        self.a, self.b = a, b

    def ppf(self, u):
        return self.dist.ppf(self.Fa + u * (self.Fb - self.Fa))

    def cdf(self, x):
        return (self.dist.cdf(x) - self.Fa) / (self.Fb - self.Fa)

    def support(self):
        lo, hi = self.dist.support()
        return max(lo, self.a), min(hi, self.b)


def _uniform(lo, hi):
    return stats.uniform(loc=lo, scale=hi - lo)


def _triangular(lo, hi, mode):
    return stats.triang(c=(mode - lo) / (hi - lo), loc=lo, scale=hi - lo)


class _Bench:
    noise_sd = 0.0
    lower_bound = None
    positive = False
    marginals = None
    threshold = None
    tail = "upper"

    def __init__(self):
        self.dim = len(self.bounds)

    @property
    def pos_in(self):
        if self.marginals is None:
            return np.array([lo > 0 for lo, _ in self.bounds])
        return np.array([m.support()[0] >= 0 for m in self.marginals])

    def ppf(self, U):
        if self.marginals is None:
            b = np.array(self.bounds, float)
            return b[:, 0] + U * (b[:, 1] - b[:, 0])
        return np.column_stack([m.ppf(U[:, j]) for j, m in enumerate(self.marginals)])

    def cdf(self, X):
        if self.marginals is None:
            b = np.array(self.bounds, float)
            return (X - b[:, 0]) / (b[:, 1] - b[:, 0])
        return np.column_stack([m.cdf(X[:, j]) for j, m in enumerate(self.marginals)])

    def noisy(self, X):
        y = self.evaluate(X)
        return y + np.random.randn(len(X)) * self.noise_sd if self.noise_sd else y


# ── uniform-input models ─────────────────────────────────────────────────
class Physics(_Bench):
    name = "physics"
    bounds = [(-200.0, 200.0)] * 3
    noise_sd = 0.1
    lower_bound = -1.0

    def __init__(self):
        super().__init__()
        self._tp = ToyProblem(3, "physics")

    def evaluate(self, X):
        return self._tp.evaluate(X)


class Plume(_Bench):
    name = "plume"
    #           Q (g/s)      u (m/s)    H (m)      x (m)         y (m)
    bounds = [(100., 2000.), (1., 6.), (1., 20.), (200., 1000.), (-30., 30.)]
    lower_bound = 0.0
    positive = True
    threshold = 210.0                       # IDLH, mg m^-3

    def evaluate(self, X):
        Q, u, H, x, y = X.T
        sy = 0.08 * x / np.sqrt(1 + 1e-4 * x)
        sz = 0.06 * x / np.sqrt(1 + 1.5e-3 * x)
        return (1e3 * Q / (2 * np.pi * u * sy * sz)
                * np.exp(-y ** 2 / (2 * sy ** 2))
                * 2 * np.exp(-H ** 2 / (2 * sz ** 2)))


class Borehole(_Bench):
    name = "borehole"
    #          rw            r              Tu                 Hu
    bounds = [(0.05, 0.15), (100., 50000.), (63070., 115600.), (990., 1110.),
              #   Tl             Hl            L               Kw
              (63.1, 116.), (700., 820.), (1120., 1680.), (9855., 12045.)]
    lower_bound = 0.0
    positive = True

    def evaluate(self, X):
        rw, r, Tu, Hu, Tl, Hl, L, Kw = X.T
        lr = np.log(r / rw)
        return (2 * np.pi * Tu * (Hu - Hl)
                / (lr * (1 + 2 * L * Tu / (lr * rw ** 2 * Kw) + Tu / Tl)))


class Ishigami(_Bench):
    name = "ishigami"
    bounds = [(-np.pi, np.pi)] * 3

    def evaluate(self, X):
        x1, x2, x3 = X.T
        return np.sin(x1) + 7 * np.sin(x2) ** 2 + 0.1 * x3 ** 4 * np.sin(x1)


class OTL(_Bench):
    name = "otl"
    #          Rb1 (kOhm)   Rb2         Rf          Rc1         Rc2           beta
    bounds = [(50., 150.), (25., 70.), (0.5, 3.), (1.2, 2.5), (0.25, 1.2), (50., 300.)]
    lower_bound = 0.0
    positive = True

    def evaluate(self, X):
        Rb1, Rb2, Rf, Rc1, Rc2, beta = X.T
        vb1 = 12 * Rb2 / (Rb1 + Rb2)
        den = beta * (Rc2 + 9) + Rf
        return ((vb1 + 0.74) * beta * (Rc2 + 9) / den + 11.35 * Rf / den
                + 0.74 * Rf * beta * (Rc2 + 9) / (Rc1 * den))


class Piston(_Bench):
    name = "piston"
    #          M (kg)      S (m2)          V0 (m3)         k (N/m)
    bounds = [(30., 60.), (0.005, 0.020), (0.002, 0.010), (1000., 5000.),
              #  P0 (N/m2)          Ta (K)       T0 (K)
              (90000., 110000.), (290., 296.), (340., 360.)]
    lower_bound = 0.0
    positive = True

    def evaluate(self, X):
        M, S, V0, k, P0, Ta, T0 = X.T
        A = P0 * S + 19.62 * M - k * V0 / S
        V = S / (2 * k) * (np.sqrt(A ** 2 + 4 * k * P0 * V0 * Ta / T0) - A)
        return 2 * np.pi * np.sqrt(M / (k + S ** 2 * P0 * V0 * Ta / (T0 * V ** 2)))


class WingWeight(_Bench):
    name = "wingweight"
    #          Sw (ft2)      Wfw (lb)      A          Lambda (deg)  q (lb/ft2)
    bounds = [(150., 200.), (220., 300.), (6., 10.), (-10., 10.), (16., 45.),
              #  lambda     tc            Nz          Wdg (lb)        Wp (lb/ft2)
              (0.5, 1.), (0.08, 0.18), (2.5, 6.), (1700., 2500.), (0.025, 0.08)]
    lower_bound = 0.0
    positive = True

    def evaluate(self, X):
        Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp = X.T
        c = np.cos(np.deg2rad(Lam))
        return (0.036 * Sw ** 0.758 * Wfw ** 0.0035 * (A / c ** 2) ** 0.6
                * q ** 0.006 * lam ** 0.04 * (100 * tc / c) ** -0.3
                * (Nz * Wdg) ** 0.49 + Sw * Wp)


class OTL20(OTL):
    name = "otl20"
    bounds = OTL.bounds + [(100., 200.)] * 14

    def evaluate(self, X):
        return super().evaluate(X[:, :6])


class Bump(_Bench):
    """Failure-mode model (research plan T1.6): smooth part plus a narrow
    Gaussian bump carrying about 10 % of the variance, in a region (bump
    above 10 % of its height) of 3 % of the input volume that a 20-run Latin
    hypercube misses with probability about one half."""
    name = "bump"
    bounds = [(0., 1.)] * 4
    A, W, C = 3.6, 0.13, 0.7

    def evaluate(self, X):
        x1, x2, x3, x4 = X.T
        return (x1 + x2 ** 2 + 0.5 * np.sin(2 * np.pi * x3) + 0.5 * x4
                + self.A * np.exp(-np.sum((X - self.C) ** 2, axis=1) / (2 * self.W ** 2)))

    def in_bump(self, X):
        """True where the bump exceeds 10 % of its height."""
        return np.exp(-np.sum((X - self.C) ** 2, axis=1) / (2 * self.W ** 2)) > 0.1


# ── non-uniform-input models ─────────────────────────────────────────────
def _quantile_bounds(marginals, lo=5e-4, hi=1 - 5e-4):
    return [(float(m.ppf(lo)), float(m.ppf(hi))) for m in marginals]


class BoreholeGUM(Borehole):
    name = "borehole_gum"
    marginals = [stats.norm(0.10, 0.0161812),
                 stats.lognorm(s=1.0056, scale=np.exp(7.71)),
                 _uniform(63070., 115600.), _uniform(990., 1100.),
                 _uniform(63.1, 116.), _uniform(700., 820.),
                 _uniform(1120., 1680.), _uniform(9985., 12045.)]
    bounds = _quantile_bounds(marginals)


class BoreholeCorr(BoreholeGUM):
    """borehole_gum with dependent inputs (research plan T2.5): a Gaussian
    copula with correlation 0.5 between the transmissivities Tu and Tl and
    between the heads Hu and Hl. ppf maps independent uniforms U to the
    dependent inputs (U -> standard normal -> correlated by the Cholesky
    factor -> marginal quantiles); cdf is its inverse (the Rosenblatt
    transform of the Gaussian copula), so designs, sparse PCE and the
    propagation all work in the independent space."""
    name = "borehole_corr"
    R = np.eye(8)
    R[2, 4] = R[4, 2] = 0.5                     # Tu, Tl
    R[3, 5] = R[5, 3] = 0.5                     # Hu, Hl
    L = np.linalg.cholesky(R)

    def ppf(self, U):
        Z = stats.norm.ppf(np.clip(U, 1e-15, 1 - 1e-15)) @ self.L.T
        return super().ppf(stats.norm.cdf(Z))

    def cdf(self, X):
        Z = stats.norm.ppf(np.clip(super().cdf(X), 1e-15, 1 - 1e-15))
        return stats.norm.cdf(np.linalg.solve(self.L, Z.T).T)


class PlumeGUM(Plume):
    name = "plume_gum"
    marginals = [stats.lognorm(s=np.log(2.0), scale=500.0),
                 Truncated(stats.weibull_min(2.0, scale=4.0), 0.5, 15.0),
                 _uniform(1., 20.), _uniform(200., 1000.),
                 stats.norm(0.0, 15.0)]
    bounds = _quantile_bounds(marginals)


class Flood(_Bench):
    name = "flood"
    marginals = [Truncated(stats.gumbel_r(1013.0, 558.0), 500.0, 3000.0),  # Q
                 Truncated(stats.norm(30.0, 8.0), 15.0, np.inf),          # Ks
                 _triangular(49.0, 51.0, 50.0),                           # Zv
                 _triangular(54.0, 56.0, 55.0),                           # Zm
                 _uniform(7.0, 9.0),                                      # Hd
                 _triangular(55.0, 56.0, 55.5),                           # Cb
                 _triangular(4990.0, 5010.0, 5000.0),                     # L
                 _triangular(295.0, 305.0, 300.0)]                        # B
    bounds = _quantile_bounds(marginals)
    tail = "lower"

    def evaluate(self, X):
        Q, Ks, Zv, Zm, Hd, Cb, L, B = X.T
        Hw = (Q / (B * Ks * np.sqrt((Zm - Zv) / L))) ** 0.6
        return Cb + Hd - Zv - Hw


BENCHMARKS = {c.name: c for c in (Physics, Plume, Borehole, Ishigami, OTL,
                                  Piston, BoreholeGUM, PlumeGUM, Flood,
                                  WingWeight, OTL20, Bump, BoreholeCorr)}

# CFD dispersion case (OpenFOAM; runs cached in results/cfd_runs/)
from cfd_benchmark import CFDDispersion            # noqa: E402
BENCHMARKS[CFDDispersion.name] = CFDDispersion
