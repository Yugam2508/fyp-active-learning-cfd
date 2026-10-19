"""
cfd_benchmark.py — the OpenFOAM dispersion case (src/cfd_dispersion.py) as a
benchmark with the interface of src/benchmarks.py.

Inputs (independent, uniform):
  U_ref (m/s) 1-6, z0 (m) 0.01-0.3, H (m) 6-15, Sc_t 0.5-1.3,
  x_r (m) 5-120, Q (g s^-1 m^-1) 1-20
Output: concentration (mg m^-3) at 1.5 m height, x_r downstream of the
building. Threshold: 210 mg m^-3 (IDLH of ammonia), as for the plume.

Only the first four inputs need a CFD run: the run gives the concentration
profile downstream of the building for a unit release, and x_r and Q enter
by interpolation and scaling. Every run is cached as one JSON file per flow
state in results/cfd_runs/ (key: the four inputs to 10 significant digits),
so repeated evaluations, other arms and other scripts reuse it. Missing runs
are computed with CFD_WORKERS parallel processes (environment variable,
default 1) in temporary case directories under CFD_WORKDIR (default: the
system temporary directory).

Reference sets (a Monte Carlo reference of 2e5 CFD runs is not affordable):
reference_sets() draws N_FLOW = 800 flow states (RandomState(8)) and, for
each, K = 50 (x_r, Q) pairs, i.e. 40 000 reference points from 800 runs:
  ref    all 40 000 points (reference output distribution)
  test   the first 5 pairs of every flow state (4000 points, nRMSE)
  q      pairs 6-7 of every flow state (1600 points, derived quantities)
"""

import os, json, hashlib, tempfile
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.environ.get("CFD_CACHE", os.path.join(ROOT, "results", "cfd_runs"))
N_FLOW, K_PAIRS = 800, 50


def _key(f):
    return "_".join(f"{v:.10g}" for v in f)


def _path(f):
    return os.path.join(CACHE, hashlib.sha1(_key(f).encode()).hexdigest()[:16] + ".json")


def _run_one(f):
    """Run (or load) the CFD case for flow state f = (U_ref, z0, H, Sc_t)."""
    import cfd_dispersion as cfd
    p = _path(f)
    if os.path.exists(p):
        return json.load(open(p))
    work = tempfile.mkdtemp(prefix="cfd_", dir=os.environ.get("CFD_WORKDIR"))
    out = cfd.run_case(*f, workdir=os.path.join(work, "case"))
    os.rmdir(work)
    rec = {"key": _key(f), "U_ref": f[0], "z0": f[1], "H": f[2], "Sc_t": f[3], **out}
    os.makedirs(CACHE, exist_ok=True)
    tmp = p + f".tmp{os.getpid()}"
    json.dump(rec, open(tmp, "w"))
    os.replace(tmp, p)                       # atomic: safe for parallel runs
    return rec


def run_flows(F, workers=None):
    """Profiles for the flow states in F (m, 4), computing missing ones."""
    workers = workers or int(os.environ.get("CFD_WORKERS", "1"))
    F = [tuple(float(v) for v in f) for f in np.asarray(F)]
    todo = sorted({f for f in F if not os.path.exists(_path(f))})
    if todo:
        if workers > 1 and len(todo) > 1:
            with ProcessPoolExecutor(workers) as ex:
                list(ex.map(_run_one, todo))
        else:
            for f in todo:
                _run_one(f)
    cache, out = {}, []
    for f in F:
        if f not in cache:
            cache[f] = _run_one(f)
        out.append(cache[f])
    return out


class CFDDispersion:
    name = "cfd"
    #          U_ref       z0            H            Sc_t         x_r           Q
    bounds = [(1., 6.), (0.01, 0.3), (6., 15.), (0.5, 1.3), (5., 120.), (1., 20.)]
    noise_sd = 0.0
    lower_bound = 0.0
    positive = True
    marginals = None
    threshold = 210.0                        # IDLH of ammonia, mg m^-3
    tail = "upper"

    def __init__(self):
        self.dim = len(self.bounds)

    @property
    def pos_in(self):
        return np.ones(self.dim, bool)

    def ppf(self, U):
        b = np.array(self.bounds, float)
        return b[:, 0] + U * (b[:, 1] - b[:, 0])

    def cdf(self, X):
        b = np.array(self.bounds, float)
        return (X - b[:, 0]) / (b[:, 1] - b[:, 0])

    def evaluate(self, X):
        X = np.atleast_2d(X)
        recs = run_flows(X[:, :4])
        return np.array([1e3 * q * np.interp(xr, r["x"], r["s"])
                         for r, xr, q in zip(recs, X[:, 4], X[:, 5])])

    def noisy(self, X):
        return self.evaluate(X)

    def reference_flows(self, seed=8):
        rs = np.random.RandomState(seed)
        b = np.array(self.bounds, float)
        F = b[:4, 0] + rs.uniform(0, 1, (N_FLOW, 4)) * (b[:4, 1] - b[:4, 0])
        P = rs.uniform(0, 1, (N_FLOW, K_PAIRS, 2))
        return F, P

    def reference_sets(self, seed=8):
        """The reference (seed 8); seed 18 gives the independent second set
        of research plan T2.6."""
        F, P = self.reference_flows(seed)
        b = np.array(self.bounds, float)
        XR = b[4:, 0] + P * (b[4:, 1] - b[4:, 0])          # (N_FLOW, K, 2)
        X = np.concatenate([np.repeat(F[:, None, :], K_PAIRS, axis=1), XR], axis=2)
        return {"ref": X.reshape(-1, 6), "test": X[:, :5].reshape(-1, 6),
                "q": X[:, 5:7].reshape(-1, 6)}
