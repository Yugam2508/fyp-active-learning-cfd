"""
uq_extension.py
===============
Two extensions to the spectral active learning framework
(drop these cells/classes into FYP_Active_Learning_Final.ipynb after the
ActiveLearner cell — they only depend on the Surrogate .fit/.predict API).

1.  SoftLCS + SoftActiveLearner
    ---------------------------
    Replaces hard, permanent LCS rejection with soft, reversible weights
    computed from k-fold cross-validation error ("weighted average
    surrogate": Goel et al. 2007; Viana et al. 2009):

        w_k  ∝  1 / ( CV-MSE_k · (1 + physics_penalty_k) )

    i.e. inverse-variance weighting, the minimum-variance rule for
    combining estimators.  Design history (kept for the paper's ablation):
    likelihood/Boltzmann weights built on the *training-point* chi-square
    score — w_k ∝ exp(-s_k/2T) — fail on the physics benchmark (RMSE ~2000
    vs ~0.7 for hard LCS) because every surrogate interpolates the sparse
    training data, so the score never separates catastrophic extrapolators
    from good models, and a consensus-deviation probe score punishes the
    one *correct* dissenter (PCE-d2) when the majority prior-reverts
    together.  CV error measures generalisation directly and separates
    committee members by orders of magnitude, which is what soft averaging
    needs to be safe.  Two advantages over hard LCS:

      (a) Reversibility — a DNN that extrapolates wildly at N=8 but becomes
          informative at N=16 regains weight automatically at the next
          update.  Hard LCS kills it forever.
      (b) A physics_penalty hook lets domain knowledge (e.g. concentration
          non-negativity, known output bounds) enter as a *soft* loss in
          the spirit of physics-informed ML (Karniadakis et al., 2021).

    For the spectral acquisition, models whose weight falls below a floor
    (w_k < tau/K) are *transiently muted* from the UEigV prediction graph so
    that pathological extrapolators cannot inject phantom disagreement —
    preserving the thesis result that DNN artefacts corrupt UEigV, while
    never discarding a model permanently.

2.  UQAnalyzer — the actual uncertainty quantification stage
    --------------------------------------------------------
    The active learning loop produces a trained committee; this class uses
    it as a Monte Carlo engine to deliver the UQ outputs that motivated the
    project (surrogate-accelerated MC: Marzouk & Najm 2009; Heidenreich et
    al. 2017), each with *epistemic error bars derived from committee
    disagreement*:

      * output mean / std / quantiles, with per-model spread,
      * full output PDF with an epistemic envelope,
      * CDF probability box (p-box): lower/upper CDF over the committee
        (Ferson & Ginzburg, 1996) and bounded exceedance probabilities
        P(Y > y_crit) — the quantity a safety engineer actually needs,
      * variance decomposition by the law of total variance,

            Var[Y] = Var_x[ mu_w(x) ]   +   E_x[ sigma_epi^2(x) ]
                     `── input-driven ─'     `── model-form (epistemic) ─'

        whose epistemic fraction is a convergence diagnostic: it tells you
        whether the next dollar should buy another CFD run (epistemic
        dominates) or whether the UQ estimate is committee-converged
        (input-driven dominates),
      * Sobol first-order and total sensitivity indices via the
        Saltelli/Jansen pick-and-freeze estimators (Saltelli et al., 2010;
        Jansen, 1999) evaluated on the cheap ensemble — with per-model
        indices providing epistemic error bars on the sensitivity ranking.

All routines are vectorised over the committee: a 10^5-point MC study costs
seconds on the surrogate versus ~10^5 CFD-hours on the true solver.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import KFold



# ═════════════════════════════════════════════════════════════════════════
# 1. Soft model weighting via cross-validation (replaces hard LCS rejection)
# ═════════════════════════════════════════════════════════════════════════
class SoftLCS:
    """
    Cross-validation-based soft weighting of committee members
    ("weighted average surrogate": Goel et al. 2007; Viana et al. 2009).

    Why CV and not the training-point chi-square?
    ---------------------------------------------
    The thesis (Sec. 3.7.4) observes that training-point residuals are
    uninformative because every surrogate interpolates the sparse data.
    We verified empirically that *all* soft schemes built on the
    training-point consistency score fail on the physics benchmark
    (RMSE ~2000 vs ~0.7 for hard LCS): with near-uniform weights, models
    that are catastrophically wrong in extrapolation still contaminate the
    weighted mean.  k-fold CV error measures *generalisation* directly and
    separates committee members by orders of magnitude (GP CV-MSE ~1e7
    under prior reversion vs ~1 for PCE-d2), so inverse-error weights

        w_k  ∝  1 / ( CV-MSE_k · (1 + penalty_k) + eps )

    collapse softly onto the consistent subset.  This is inverse-variance
    weighting — the minimum-variance rule for combining estimators — and
    unlike hard LCS it is fully reversible: a DNN that becomes informative
    once N grows regains weight automatically at the next update.

    Parameters
    ----------
    n_folds : int
        k for k-fold CV (3 is enough to separate failure modes by orders
        of magnitude and keeps the refit cost low).
    min_points : int
        Uniform weights until this many training points exist (same guard
        as hard LCS).
    weight_floor : float
        Models with normalised weight below weight_floor / K are muted
        from the UEigV prediction graph for that iteration (transient).
    physics_penalty : callable or None
        physics_penalty(model, X_probe) -> float >= 0; multiplies the CV
        error as (1 + penalty).  Encodes soft physical constraints
        (see make_bound_penalty) in the spirit of physics-informed ML.
    holdout_families : set of str
        Model families too expensive for k-fold refits (e.g. {"DNN"}) are
        scored with a single 80/20 holdout refit instead.
    model_factory : callable(model) -> fresh untrained clone
        Needed because CV refits must not mutate the live committee.
        Defaults to a name-based factory for the notebook's classes.
    """

    def __init__(self, n_folds=3, min_points=8, weight_floor=0.5,
                 physics_penalty=None, holdout_families=("DNN",),
                 model_factory=None):
        self.k = n_folds
        self.min_points = min_points
        self.weight_floor = weight_floor
        self.physics_penalty = physics_penalty
        self.holdout_families = set(holdout_families)
        self.model_factory = model_factory or default_model_factory

    def _cv_mse(self, model, X, y, dim):
        n = len(y)
        if model.family in self.holdout_families:
            # single holdout: cheap, order-of-magnitude is all we need
            idx = np.random.permutation(n)
            cut = max(2, int(0.8 * n))
            tr, te = idx[:cut], idx[cut:]
            f = self.model_factory(model, dim)
            f.fit(X[tr], y[tr])
            return float(np.mean((y[te] - f.predict(X[te])) ** 2))
        kf = KFold(n_splits=min(self.k, n), shuffle=True, random_state=0)
        mse = 0.0
        for tr, te in kf.split(X):
            f = self.model_factory(model, dim)
            f.fit(X[tr], y[tr])
            mse += np.mean((y[te] - f.predict(X[te])) ** 2)
        return float(mse / kf.get_n_splits())

    def weights(self, models, X, y, X_probe=None):
        """Returns (w, info): normalised weights and diagnostics."""
        K, dim = len(models), X.shape[1]
        if len(y) < self.min_points:
            w = np.full(K, 1.0 / K)
            return w, {"decision": "uniform (insufficient data)",
                       "cv_mse": None, "active": np.ones(K, bool)}

        mse = np.array([self._cv_mse(m, X, y, dim) for m in models])
        pen = np.zeros(K)
        if self.physics_penalty is not None and X_probe is not None:
            pen = np.array([self.physics_penalty(m, X_probe) for m in models])

        w = 1.0 / (mse * (1.0 + pen) + 1e-12)
        w = w / w.sum()

        active = w >= self.weight_floor / K
        if active.sum() < 2:                      # UEigV graph needs >= 2 nodes
            active = np.zeros(K, bool)
            active[np.argsort(-w)[:2]] = True
        return w, {"decision": "cv-inverse-error", "cv_mse": mse,
                   "penalty": pen, "active": active}


def default_model_factory(model, dim):
    """Fresh untrained clone of a notebook surrogate, keyed on its name."""
    from framework_core import (GPSurrogate, PCESurrogate, RFSurrogate,
                                SVRSurrogate, DNNSurrogate)
    name = model.name
    if name.startswith("GP-"):
        return GPSurrogate(name.split("-", 1)[1])
    if name.startswith("PCE-d"):
        return PCESurrogate(int(name[-1]))
    if name.startswith("RF-"):
        return RFSurrogate(int(name.split("-")[1]))
    if name.startswith("SVR"):
        return SVRSurrogate("rbf")
    if "H128" in name:
        return DNNSurrogate(dim, 128)
    return DNNSurrogate(dim, 64)


def make_bound_penalty(lower=None, upper=None, scale=10.0):
    """
    Soft physics constraint: penalise the fraction of probe predictions
    violating known output bounds (e.g. lower=0 for a concentration field;
    lower=-1 for the physics benchmark since f = sin*cos + exp(-z^2)
    + 0.2||x||^2 >= -1 analytically).  Use as
    SoftLCS(physics_penalty=make_bound_penalty(lower=0)).
    """
    def penalty(model, X_probe):
        p = model.predict(X_probe)
        viol = 0.0
        if lower is not None:
            viol += np.mean(p < lower)
        if upper is not None:
            viol += np.mean(p > upper)
        return scale * viol
    return penalty


class SoftActiveLearner:
    """
    Drop-in variant of ActiveLearner using SoftLCS (CV inverse-error
    weights) instead of permanent rejection.  Wraps the original class by
    composition so the notebook's ActiveLearner is untouched:

        agent = SoftActiveLearner(ActiveLearner, problem, domain,
                                  physics_penalty=make_bound_penalty(lower=-1))

    update_every : recompute CV weights every this many steps (CV refits
    are the only added cost; every 3 steps is a good accuracy/cost point).
    """

    def __init__(self, base_cls, problem, domain, n_init=5,
                 n_folds=3, weight_floor=0.5, physics_penalty=None,
                 update_every=3):
        self.base = base_cls(problem, domain, n_init=n_init)
        self.base.rejected = set()            # unused; kept for API compat
        self.soft = SoftLCS(n_folds=n_folds, weight_floor=weight_floor,
                            physics_penalty=physics_penalty)
        K = len(self.base.models)
        self.w = np.full(K, 1.0 / K)
        self.active = np.ones(K, bool)
        self.weight_history = []
        self.update_every = update_every
        self._steps_since_update = 0

    # ---- delegation -------------------------------------------------------
    @property
    def models(self): return self.base.models
    @property
    def X(self): return self.base.X
    @property
    def y(self): return self.base.y
    @property
    def domain(self): return self.base.domain
    @property
    def dim(self): return self.base.dim
    @property
    def problem(self): return self.base.problem

    def train(self, force_weight_update=False):
        for m in self.base.models:            # ALL models always retrain —
            m.fit(self.base.X, self.base.y)   # nothing is ever discarded
        self._steps_since_update += 1
        due = (self._steps_since_update >= self.update_every
               or len(self.weight_history) == 0)
        if due or force_weight_update:
            self._update_weights()
            self._steps_since_update = 0

    def _update_weights(self):
        X_probe = np.random.uniform(self.domain[:, 0], self.domain[:, 1],
                                    size=(200, self.dim))
        self.w, info = self.soft.weights(self.base.models, self.base.X,
                                         self.base.y, X_probe=X_probe)
        self.active = info["active"]
        self.weight_history.append((len(self.base.y), self.w.copy()))

    def predict(self, X):
        """Weighted ensemble prediction (the weighted-average surrogate)."""
        preds = np.array([m.predict(X) for m in self.base.models])
        return np.average(preds, axis=0, weights=self.w)

    def ueigv(self, grid):
        """UEigV over the active (non-muted) sub-committee — identical
        spectral construction to the original, so results are comparable."""
        models = [m for m, a in zip(self.base.models, self.active) if a]
        preds = np.array([m.predict(grid) for m in models])
        if preds.shape[0] < 2:
            return np.ones(len(grid))
        K, N = preds.shape
        scores = np.zeros(N)
        preds_norm = ((preds - preds.mean(axis=0, keepdims=True))
                      / (preds.std(axis=0, keepdims=True) + 1e-8))
        for i in range(N):
            p = preds_norm[:, i:i+1]
            d2 = squareform(pdist(p, metric="sqeuclidean"))
            A = np.exp(-d2 / 2.0)
            Di = np.diag(1.0 / np.sqrt(A.sum(axis=1) + 1e-12))
            L = np.eye(K) - Di @ A @ Di
            scores[i] = np.sum(np.maximum(0.0, 1.0 - np.linalg.eigvalsh(L)))
        return scores

    def step(self):
        grid = np.random.uniform(self.domain[:, 0], self.domain[:, 1],
                                 size=(300, self.dim))
        idx = np.argmax(self.ueigv(grid))
        x_new = grid[idx:idx+1]
        y_new = self.problem.noisy(x_new)
        self.base.X = np.vstack([self.base.X, x_new])
        self.base.y = np.append(self.base.y, y_new)
        return x_new, y_new

# ═════════════════════════════════════════════════════════════════════════
# 2. Forward uncertainty quantification on the trained committee
# ═════════════════════════════════════════════════════════════════════════
def uniform_sampler(bounds):
    """Returns sampler(n) -> (n, d) uniform over the given bounds."""
    b = np.array(bounds, float)
    def sampler(n):
        return b[:, 0] + np.random.uniform(0, 1, (n, b.shape[0])) * (b[:, 1] - b[:, 0])
    return sampler


class UQAnalyzer:
    """
    Forward UQ through a trained surrogate committee.

    Parameters
    ----------
    models : list of trained surrogates (the committee)
    weights : (K,) array or None.  None -> uniform (matches hard-LCS
              ensemble_predict if you pass only the non-rejected models).
    noise_std : observation noise to add to the aleatoric budget (sigma_eps).
    """

    def __init__(self, models, weights=None, noise_std=0.0):
        self.models = list(models)
        K = len(self.models)
        self.w = (np.full(K, 1.0/K) if weights is None
                  else np.asarray(weights, float) / np.sum(weights))
        self.noise_std = float(noise_std)

    # ---- core -------------------------------------------------------------
    def _preds(self, X):
        return np.array([m.predict(X) for m in self.models])     # (K, N)

    def propagate(self, sampler, n_mc=50_000,
                  quantiles=(0.05, 0.25, 0.5, 0.75, 0.95)):
        """
        Monte Carlo propagation of the input distribution through the
        committee.  Returns a dict of UQ outputs, each with epistemic bands.
        """
        X = sampler(n_mc)
        P = self._preds(X)                                       # (K, N)
        mu = np.average(P, axis=0, weights=self.w)               # mixture mean
        s2_epi_x = np.average((P - mu[None, :])**2, axis=0, weights=self.w)

        # --- law of total variance over the input distribution -------------
        V_input = float(np.var(mu))                  # Var_x[ mu_w(x) ]
        V_epi   = float(np.mean(s2_epi_x))           # E_x[ sigma_epi^2(x) ]
        V_noise = self.noise_std**2
        V_total = V_input + V_epi + V_noise

        # --- per-model output statistics (epistemic spread of each stat) ---
        per_model = {
            "mean":  P.mean(axis=1),                              # (K,)
            "std":   P.std(axis=1),
            "quantiles": {q: np.quantile(P, q, axis=1) for q in quantiles},
        }
        stats = {
            "mean":  float(np.average(per_model["mean"], weights=self.w)),
            "std":   float(np.sqrt(V_total)),
            "quantiles": {q: float(np.quantile(mu, q)) for q in quantiles},
            "mean_epistemic_band": (float(per_model["mean"].min()),
                                    float(per_model["mean"].max())),
        }
        return {
            "X": X, "preds": P, "mu": mu,
            "stats": stats, "per_model": per_model,
            "decomposition": {
                "V_input": V_input, "V_epistemic": V_epi, "V_noise": V_noise,
                "V_total": V_total,
                "epistemic_fraction": V_epi / max(V_total, 1e-300),
            },
        }

    # ---- probability box on the CDF / exceedance --------------------------
    def pbox(self, preds, y_grid=None, n_grid=200):
        """
        CDF probability box: at each output level y, the committee's CDFs
        span [F_lower(y), F_upper(y)]; the weighted mixture CDF is the
        central estimate.  Exceedance bounds follow as 1 - F.
        """
        P = preds
        if y_grid is None:
            lo, hi = np.quantile(P, 0.001), np.quantile(P, 0.999)
            y_grid = np.linspace(lo, hi, n_grid)
        F = np.array([[np.mean(p <= y) for y in y_grid] for p in P])  # (K, G)
        F_central = np.average(F, axis=0, weights=self.w)
        return {"y": y_grid, "F_lower": F.min(axis=0), "F_upper": F.max(axis=0),
                "F_central": F_central, "F_models": F}

    def exceedance(self, preds, threshold):
        """P(Y > threshold): central estimate with epistemic bounds."""
        p_k = np.array([np.mean(p > threshold) for p in preds])
        return {"central": float(np.average(p_k, weights=self.w)),
                "lower": float(p_k.min()), "upper": float(p_k.max()),
                "per_model": p_k}

    # ---- Sobol sensitivity indices (Saltelli/Jansen pick-and-freeze) ------
    def sobol(self, sampler, dim, n=4096):
        """
        First-order S_i and total ST_i indices of the weighted ensemble,
        with per-model indices as epistemic error bars.
        Estimators: S_i  = mean( f_B (f_ABi - f_A) ) / V      (Saltelli 2010)
                    ST_i = mean( (f_A - f_ABi)^2 ) / (2 V)    (Jansen 1999)
        Cost: n (2 + dim) surrogate evaluations — free compared to CFD.
        """
        A, B = sampler(n), sampler(n)
        fA = self._preds(A)                                     # (K, n)
        fB = self._preds(B)
        K = len(self.models)
        S  = np.zeros((K, dim)); ST = np.zeros((K, dim))
        V  = np.var(np.concatenate([fA, fB], axis=1), axis=1) + 1e-300  # (K,)
        for i in range(dim):
            ABi = A.copy(); ABi[:, i] = B[:, i]
            fABi = self._preds(ABi)                             # (K, n)
            S[:, i]  = np.mean(fB * (fABi - fA), axis=1) / V
            ST[:, i] = 0.5 * np.mean((fA - fABi)**2, axis=1) / V
        return {
            "S":  np.average(S,  axis=0, weights=self.w),
            "ST": np.average(ST, axis=0, weights=self.w),
            "S_per_model": S, "ST_per_model": ST,
            "S_band":  (S.min(axis=0),  S.max(axis=0)),
            "ST_band": (ST.min(axis=0), ST.max(axis=0)),
        }
