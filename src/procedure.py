"""
procedure.py — the procedure of the paper (flowchart, Section 7) for a GUM S1
evaluation with a surrogate fitted to N runs of an expensive model.

    from procedure import evaluate, budget_table
    res = evaluate(X, y, sample_inputs, bounds=..., positive=True, threshold=210)
    print(budget_table(res))

Steps
  1. Candidates: GP (Matern-5/2, ARD, estimated amplitude, nugget), GP with a
     quadratic trend, sparse PCE; for a positive output also each on the log
     scale (log-transformed positive inputs).
  2. Selection: the candidate with the smallest leave-one-out (LOO) RMSE on
     the original scale of y (every candidate re-fitted N times).
  3. Surrogate contribution: the LOO residuals r_i of the chosen candidate on
     its fitting (latent) scale.
  4. GUM S1 propagation of M input draws through the surrogate, adding to
     every trial a residual drawn with equal probabilities from {r_i}:
     y_k = phi^-1(phi(fhat(x_k)) + r_J).
     With ndig given, the number of trials is chosen by the adaptive Monte
     Carlo procedure of GUM S1 (JCGM 101:2008, 7.9): batches of
     h = max(100/(1-p), 1e4) trials until twice the standard deviation of
     the batch means of y, u(y) and both interval limits is at most the
     numerical tolerance delta = 0.5 10^l, u(y) = c 10^l with ndig
     significant digits in c.
  5. Report: estimate y (mean), standard uncertainty u(y) (SD), the
     probabilistically symmetric 95 % coverage interval, P(Y > T) if a
     threshold is given; the same without the surrogate contribution; 95 %
     jackknife intervals of the reported quantities over the LOO fits; and
     an uncertainty budget of Var(Y) with the surrogate error E treated as an
     additional input quantity: first-order variance shares of the inputs and
     the total variance share of E (Saltelli et al. 2010), the rest being
     interactions between inputs. For comparison, the GUM uncertainty
     framework (law of propagation of uncertainty, first order, through the
     chosen surrogate at the input means, sensitivity coefficients by central
     differences, interval y +- 1.96 u(y)).

Inputs
  X (N, d), y (N,)       the model runs
  sample_inputs(n, rng)  draws n input vectors from the input distribution
                         (independent components, as GUM S1 with independent
                         inputs; needed for the variance shares)
  bounds                 [(lo, hi)] per input, for the PCE mapping of uniform
                         inputs; alternatively unit_map(X) -> (0, 1)^d (input
                         CDFs) for non-uniform inputs
  positive               True if the output is strictly positive (adds the
                         log-scale candidates); pos_in per input likewise
The validation of steps 2-4 against the true models is in the paper.
"""

import numpy as np
from surrogates import GP, SparsePCE

Z95 = 1.959963984540054


def jackknife_plus(mu_loo, r, alpha=0.05):
    """Jackknife+ bounds (Barber et al. 2021) from LOO predictions mu_loo
    (n_runs, m) at m points and LOO residuals r (n_runs,)."""
    n = len(r)
    k_lo, k_hi = int(np.floor(alpha * (n + 1))), int(np.ceil((1 - alpha) * (n + 1)))
    a = np.abs(r)[:, None]
    lo = (np.sort(mu_loo - a, axis=0)[k_lo - 1] if k_lo >= 1
          else np.full(mu_loo.shape[1], -np.inf))
    hi = (np.sort(mu_loo + a, axis=0)[k_hi - 1] if k_hi <= n
          else np.full(mu_loo.shape[1], np.inf))
    return lo, hi


def _candidates(d, positive, pos_in, bounds, unit_map):
    c = {"gp_m52": lambda: GP("m52", d, pos_in), "gp_quad": lambda: GP("quad", d, pos_in),
         "spce": lambda: SparsePCE(d, bounds, pos_in=pos_in, unit_map=unit_map)}
    if positive:
        c.update({"gp_log": lambda: GP("log", d, pos_in),
                  "gp_quadlog": lambda: GP("quadlog", d, pos_in),
                  "spce_log": lambda: SparsePCE(d, bounds, log=True, pos_in=pos_in,
                                                unit_map=unit_map)})
    return c


def _quantities(Y, threshold):
    q = {"mean": Y.mean(-1), "sd": Y.std(-1), "q025": np.quantile(Y, 0.025, axis=-1),
         "q975": np.quantile(Y, 0.975, axis=-1), "q95": np.quantile(Y, 0.95, axis=-1)}
    if threshold is not None:
        q["p_exceed"] = (Y > threshold).mean(-1)
    return q


def evaluate(X, y, sample_inputs, bounds=None, unit_map=None, positive=None,
             pos_in=None, threshold=None, M=100_000, n_jack=None, n_sobol=20_000,
             seed=0, ndig=None):
    X, y = np.asarray(X, float), np.asarray(y, float)
    N, d = X.shape
    positive = bool(np.all(y > 0)) if positive is None else positive
    pos_in = np.all(X > 0, axis=0) if pos_in is None else np.asarray(pos_in, bool)
    if bounds is None:
        bounds = list(zip(X.min(0), X.max(0)))
    rng = np.random.default_rng(seed)
    cands = _candidates(d, positive, pos_in, bounds, unit_map)

    # 1-2. LOO error of every candidate, original scale
    loo = {}
    for name, make in cands.items():
        pred = np.empty(N); fits = []
        for i in range(N):
            keep = np.arange(N) != i
            s = make().fit(X[keep], y[keep])
            pred[i] = s.predict(X[i:i + 1])[0]
            fits.append(s)
        loo[name] = {"rmse": float(np.sqrt(np.mean((y - pred) ** 2))), "fits": fits}
    chosen = min(loo, key=lambda k: loo[k]["rmse"])
    model = cands[chosen]().fit(X, y)
    fits = loo[chosen]["fits"]

    # 3. LOO residuals on the latent scale
    z = model.to_latent(y)
    r = np.array([z[i] - f.latent(X[i:i + 1])[0][0] for i, f in enumerate(fits)])

    # 4. propagation with and without the surrogate contribution
    if ndig is None:
        Xm = sample_inputs(M, rng)
        mu = model.latent(Xm)[0]
        y_plain = model.from_latent(mu)
        y_full = model.from_latent(mu + r[rng.integers(0, N, M)])
        adaptive = None
    else:
        Xm, y_full, adaptive = _adaptive_mc(model, r, sample_inputs, rng, ndig)
        M = len(Xm)
        y_plain = model.from_latent(model.latent(Xm)[0])
    out = {"chosen": chosen, "loo_rmse": {k: v["rmse"] for k, v in loo.items()},
           "loo_residuals_latent": r.tolist(), "M": M, "N": N}
    if adaptive is not None:
        out["adaptive_mc"] = adaptive
    for tag, Y in (("with_surrogate_contribution", y_full), ("surrogate_as_exact", y_plain)):
        out[tag] = {k: float(v) for k, v in _quantities(Y, threshold).items()}

    # 5a. jackknife intervals of the reported quantities (surrogate error only):
    #     the LOO fits evaluated on the same draws (all M, or the first n_jack),
    #     centred on the estimate without the surrogate contribution
    Xj = Xm if n_jack is None else Xm[:n_jack]
    est = _quantities(y_plain if n_jack is None else model.predict(Xj), threshold)
    jk = _quantities(np.array([f.predict(Xj) for f in fits]), threshold)
    out["jackknife_95"] = {}
    for k in est:
        se = np.sqrt((N - 1) / N * np.sum((jk[k] - jk[k].mean()) ** 2))
        out["jackknife_95"][k] = [float(est[k] - Z95 * se), float(est[k] + Z95 * se)]

    # 5b. budget on the full model Y = phi^-1(phi(fhat(X)) + E), with the
    #     surrogate error E treated as one more input quantity: first-order
    #     variance shares of the inputs and the total share of E (estimators
    #     of Saltelli et al. 2010 and Jansen's total-effect formula therein)
    def full(Xs, idx):
        return model.from_latent(model.latent(Xs)[0] + r[idx])
    A, B = sample_inputs(n_sobol, rng), sample_inputs(n_sobol, rng)
    eA, eB = rng.integers(0, N, n_sobol), rng.integers(0, N, n_sobol)
    fA, fB = full(A, eA), full(B, eB)
    V = np.var(np.concatenate([fA, fB]))
    S = []
    for i in range(d):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        S.append(float(np.mean(fB * (full(ABi, eA) - fA)) / V))
    S_E = float(np.mean((fA - full(A, eB)) ** 2) / (2 * V))
    out["budget"] = {"input_share": S, "surrogate_share": S_E,
                     "interaction_share": float(max(0.0, 1 - sum(S) - S_E))}

    # 6. GUM uncertainty framework through the surrogate, for comparison
    Xs = sample_inputs(100_000, rng)
    xbar, ux = Xs.mean(0), Xs.std(0)
    c = np.empty(d)
    for i in range(d):
        dx = np.zeros(d); dx[i] = 1e-3 * ux[i]
        c[i] = (model.predict((xbar + dx)[None])[0]
                - model.predict((xbar - dx)[None])[0]) / (2 * dx[i])
    y0 = float(model.predict(xbar[None])[0])
    u0 = float(np.sqrt(np.sum((c * ux) ** 2)))
    out["gum_framework"] = {"y": y0, "u": u0, "q025": y0 - Z95 * u0, "q975": y0 + Z95 * u0,
                            "sensitivity": c.tolist()}
    return out


def _adaptive_mc(model, r, sample_inputs, rng, ndig, p=0.95, max_batches=1000):
    """GUM S1 7.9: batches until the numerical tolerance is met."""
    h = max(int(np.ceil(100 / (1 - p))), 10_000)
    Xs, Ys, stats = [], [], []
    for _ in range(max_batches):
        Xb = sample_inputs(h, rng)
        Yb = model.from_latent(model.latent(Xb)[0] + r[rng.integers(0, len(r), h)])
        Xs.append(Xb); Ys.append(Yb)
        stats.append([Yb.mean(), Yb.std(), *np.quantile(Yb, [(1 - p) / 2, (1 + p) / 2])])
        if len(stats) < 2:
            continue
        S = np.array(stats)
        Y = np.concatenate(Ys)
        u = Y.std()
        l = int(np.floor(np.log10(u))) - ndig + 1
        delta = 0.5 * 10.0 ** l
        sd = S.std(0, ddof=1) / np.sqrt(len(S))
        if np.all(2 * sd <= delta):
            break
    return (np.vstack(Xs), np.concatenate(Ys),
            {"batches": len(stats), "batch_size": h, "ndig": ndig, "delta": float(delta),
             "two_sd_of_mean": (2 * sd).tolist(), "met": bool(np.all(2 * sd <= delta))})


def budget_table(res, names=None, unit=""):
    """Plain-text summary of an evaluate() result."""
    w, p = res["with_surrogate_contribution"], res["surrogate_as_exact"]
    names = names or []
    lines = [f"surrogate chosen by LOO error: {res['chosen']} (N = {res['N']})",
             f"y = {w['mean']:.4g} {unit}, u(y) = {w['sd']:.3g} {unit} "
             f"(without surrogate contribution {p['sd']:.3g})",
             f"95 % coverage interval [{w['q025']:.4g}, {w['q975']:.4g}] {unit}"]
    if "p_exceed" in w:
        j = res["jackknife_95"]["p_exceed"]
        lines.append(f"P(Y > T) = {w['p_exceed']:.3f} (jackknife 95 %: {j[0]:.3f} to {j[1]:.3f})")
    names = names if len(names) == len(res["budget"]["input_share"]) else \
        [f"x{i + 1}" for i in range(len(res["budget"]["input_share"]))]
    lines.append("variance shares: " + ", ".join(
        f"{n} {100 * s:.1f} %" for n, s in zip(names, res["budget"]["input_share"]))
        + f", interactions {100 * res['budget']['interaction_share']:.1f} %, "
        f"surrogate {100 * res['budget']['surrogate_share']:.1f} %")
    return "\n".join(lines)
