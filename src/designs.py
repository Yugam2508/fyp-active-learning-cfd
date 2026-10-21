"""
designs.py — space-filling designs on (0, 1)^d for research plan T2.1 (do the
results depend on the design type?). All draw their randomness from the
global numpy RNG, like framework_core.lhs, so np.random.seed(seed) fixes
the design.

  lhs      random Latin hypercube (framework_core.lhs, the main study)
  maximin  the Latin hypercube with the largest smallest pairwise distance
           among 200 random ones
  sobol    scrambled Sobol' sequence (SciPy: linear matrix scrambling with a
           digital shift), first n points
"""

import warnings
import numpy as np
from scipy.stats import qmc
from scipy.spatial.distance import pdist
from framework_core import lhs


def make_design(kind, d, n):
    if kind == "lhs":
        return lhs(d, n)
    seed = int(np.random.randint(0, 2 ** 31 - 1))
    if kind == "maximin":
        rng = np.random.default_rng(seed)
        cands = [qmc.LatinHypercube(d=d, seed=rng).random(n) for _ in range(200)]
        return max(cands, key=lambda U: pdist(U).min())
    if kind == "sobol":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # n need not be a power of 2
            return qmc.Sobol(d, scramble=True, seed=seed).random(n)
    raise ValueError(kind)
