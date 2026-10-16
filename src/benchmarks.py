"""
benchmarks.py — test models for the generalisation study.

Each class exposes the interface framework_core.ActiveLearner expects:
  dim, bounds (list of (lo, hi)), evaluate(X), noisy(X)
plus
  name, lower_bound  (known physical lower bound of the output, or None;
                      used by the soft physics penalty)
Inputs are uniform over `bounds` (rectangular PDFs, as in GUM S1 examples).
The new models are deterministic computer codes, so noisy() adds no noise.

Models
------
physics   f = sin2x1 cos2x2 + exp(-x3^2) + 0.2||x||^2 on [-200,200]^3
          (the original FYP benchmark; noise SD 0.1 as before). Its dominant
          quadratic term lies exactly in the degree-2 PCE member's class.
plume     Gaussian plume ground-level concentration of a continuous ammonia
          release (mg m^-3), Briggs rural dispersion coefficients for neutral
          stability (class D), ground reflection, receptor at height 0:
            C = 1e3 Q / (2 pi u sy sz) exp(-y^2/(2 sy^2)) 2 exp(-H^2/(2 sz^2))
            sy = 0.08 x (1+1e-4 x)^-1/2,  sz = 0.06 x (1+1.5e-3 x)^-1/2
          Inputs: Q emission rate 100-2000 g/s, u wind speed 1-6 m/s,
          H release height 1-20 m, x downwind distance 200-1000 m,
          y crosswind offset -30-30 m.  Output spans three orders of
          magnitude; no committee member matches its functional form.
borehole  Water flow through a borehole (m^3/yr), 8 inputs, standard ranges
          (Harper & Gupta 1983; Surjanovic & Bingham test-function library).
ishigami  sin x1 + 7 sin^2 x2 + 0.1 x3^4 sin x1 on [-pi, pi]^3.
"""

import numpy as np
from framework_core import ToyProblem


class _Bench:
    noise_sd = 0.0
    lower_bound = None

    def __init__(self):
        self.dim = len(self.bounds)

    def noisy(self, X):
        y = self.evaluate(X)
        return y + np.random.randn(len(X)) * self.noise_sd if self.noise_sd else y


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


BENCHMARKS = {c.name: c for c in (Physics, Plume, Borehole, Ishigami)}
