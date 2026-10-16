"""
make_paper_figures.py — print-ready figures for the IMEKO 2027 extended abstract.

Usage:  python make_paper_figures.py [strip] [domain] [pbox]      (default: all)

  strip   per-seed RMSE of every paired ablation arm
          reads results/results_{hard,varacq,soft,nopce,gpvar}.json
  domain  domain-size sweep and Schwefel control
          reads results/results_domain_*.json  (experiments/campaign_domain.py)
  pbox    representative error-weighted run, seed 1000 (the first campaign
          seed, not selected for quality), campaign.py protocol. Runs the
          model, then writes results/representative_run_seed1000.json with
          every number quoted in the text, plus the p-box figure.

Figures go to figures/<name>.{pdf,png}. Styling: one column (82 mm) wide,
8 pt STIX (Times-like) text, words with units in parentheses on axes
(IMEKO template rule), no titles (captions live in LaTeX).
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
RES = _os.path.join(_ROOT, "results")
FIG = _os.path.join(_ROOT, "figures"); _os.makedirs(FIG, exist_ok=True)

import sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_W = 82 / 25.4                       # one IMEKO column, inches
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, BLUE_LIGHT, ORANGE = "#2a78d6", "#b9d3f2", "#eb6834"
THRESH = 200.0                          # success threshold used in the paper

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelcolor": INK, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def _save(fig, name):
    fig.savefig(_os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(_os.path.join(FIG, f"{name}.png"), dpi=300)
    plt.close(fig)
    print("wrote", _os.path.join(FIG, f"{name}.pdf"))


# ─────────────────────────────────────────────────────────────────────────
def fig_strip():
    arms = [("hard",   "UEigV, hard rejection"),
            ("varacq", "Variance, hard rejection"),
            ("soft",   "UEigV, error weighting"),
            ("nopce",  "UEigV, no PCE members"),
            ("gpvar",  "Single GP, posterior SD")]
    # y labels sit outside the axes: shrink the canvas so the tight bbox ~ COL_W
    fig, ax = plt.subplots(figsize=(COL_W - 0.67, 2.15))
    rs = np.random.RandomState(0)                  # jitter only
    for i, (key, _) in enumerate(arms):
        r = np.array([t["rmse"] for t in
                      json.load(open(_os.path.join(RES, f"results_{key}.json")))])
        ax.scatter(r, i + rs.uniform(-0.17, 0.17, len(r)), s=8, color=BLUE,
                   alpha=0.8, linewidths=0, zorder=3)
        med = np.median(r)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=1.3, zorder=4)
        ax.text(1.01, i, f"{int((r < THRESH).sum())}/{len(r)}",
                transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=7, color=INK)
    ax.text(1.01, -0.85, "Success", transform=ax.get_yaxis_transform(),
            va="center", ha="left", fontsize=6.5, color=MUTED)
    ax.axvline(THRESH, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.text(THRESH * 1.2, -0.75, "success threshold", fontsize=6.5,
            color=MUTED, va="center", ha="left")
    ax.set_xscale("log")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([lab for _, lab in arms], color=INK)
    ax.set_ylim(len(arms) - 0.45, -1.1)
    ax.set_xlabel("Test root mean square error (log scale)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.tick_params(axis="y", length=0)
    _clean(ax)
    _save(fig, "fig_ablation_strip")


# ─────────────────────────────────────────────────────────────────────────
def domain_summary():
    """rho per configuration from results/results_domain_*.json."""
    out = {}
    for f in sorted(glob.glob(_os.path.join(RES, "results_domain_*.json"))):
        d = json.load(open(f))
        al = np.array([t["rmse_al"] for t in d])
        gp = np.array([t["rmse_lhsgp"] for t in d])
        out[d[0]["config"]] = {
            "function": d[0]["function"], "half_width": d[0]["half_width"],
            "n": len(d), "rho_mean": float(al.mean() / gp.mean()),
            "rho_median": float(np.median(al) / np.median(gp)),
            "mean_al": float(al.mean()), "mean_lhsgp": float(gp.mean()),
            "median_al": float(np.median(al)), "median_lhsgp": float(np.median(gp)),
            "al_wins": int((al < gp).sum()),
            "output_std": d[0]["output_std"],
        }
    return out


def fig_domain():
    """Median test RMSE normalised by the output SD, both arms, per domain."""
    s = domain_summary()
    phys = sorted((v for v in s.values() if v["function"] == "physics"),
                  key=lambda v: v["half_width"])
    h = np.array([v["half_width"] for v in phys])
    gp = np.array([v["median_lhsgp"] / v["output_std"] for v in phys])
    al = np.array([v["median_al"] / v["output_std"] for v in phys])
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    ax.plot(h, gp, color=ORANGE, lw=1.5, marker="s", ms=4, zorder=3,
            label="Passive GP (20-point LHS)")
    ax.plot(h, al, color=BLUE, lw=1.5, marker="o", ms=4, zorder=3,
            label="Committee, adaptive")
    if "schwefel" in s:
        sw = s["schwefel"]
        ax.plot([sw["half_width"]], [sw["median_lhsgp"] / sw["output_std"]],
                marker="s", ms=4.5, mfc="white", mec=ORANGE, mew=1.2, ls="none",
                zorder=3)
        ax.plot([sw["half_width"]], [sw["median_al"] / sw["output_std"]],
                marker="o", ms=4.5, mfc="white", mec=BLUE, mew=1.2, ls="none",
                zorder=3)
        ax.annotate("Schwefel\ncontrol", (sw["half_width"], 1.0),
                    textcoords="offset points", xytext=(0, -24), ha="center",
                    fontsize=6.5, color=MUTED)
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(14, 1.15, "constant predictor", fontsize=6.5, color=MUTED)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([2, 10, 50, 100, 200, 500])
    ax.set_xticklabels(["2", "10", "50", "100", "200", "500"])
    ax.set_ylim(1e-4, 3)
    ax.set_xlabel("Domain half-width, $h$")
    ax.set_ylabel("Median RMSE / output SD")
    ax.grid(color=GRID, lw=0.5)
    ax.legend(loc="lower left", frameon=False)
    _clean(ax)
    _save(fig, "fig_domain_sweep")


# ─────────────────────────────────────────────────────────────────────────
def fig_pbox(seed=1000):
    try:
        import torch
    except Exception:
        torch = None
    from framework_core import ToyProblem, ActiveLearner
    from uq_extension import (SoftActiveLearner, UQAnalyzer,
                              make_bound_penalty)

    problem = ToyProblem(3, "physics")
    b = np.array(problem.bounds, float)
    fixed = lambda n, rs: b[:, 0] + rs.uniform(0, 1, (n, 3)) * (b[:, 1] - b[:, 0])
    # identical fixed sets to experiments/campaign.py
    X_test = fixed(5000, np.random.RandomState(7)); y_test = problem.evaluate(X_test)
    X_ref = fixed(200_000, np.random.RandomState(8)); y_ref = problem.evaluate(X_ref)
    X_mc = fixed(30_000, np.random.RandomState(9))

    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
    a = SoftActiveLearner(ActiveLearner, ToyProblem(3, "physics"), problem.bounds,
                          n_folds=3, update_every=3,
                          physics_penalty=make_bound_penalty(lower=-1.0))
    a.train()
    traj = []

    def snap():
        uq = UQAnalyzer(a.models, weights=a.w, noise_std=0.1)
        P = uq._preds(X_mc)
        mu = np.average(P, axis=0, weights=uq.w)
        v_in = float(np.var(mu))
        v_epi = float(np.mean(np.average((P - mu) ** 2, axis=0, weights=uq.w)))
        traj.append({"n_runs": int(len(a.y)), "v_input": v_in,
                     "v_model_form": v_epi,
                     "model_form_fraction": v_epi / (v_in + v_epi)})

    snap()
    for i in range(15):
        a.step(); a.train(force_weight_update=(i == 14)); snap()

    uq = UQAnalyzer(a.models, weights=a.w, noise_std=0.1)
    P = uq._preds(X_mc)
    mu = np.average(P, axis=0, weights=uq.w)
    rmse = float(np.sqrt(np.mean(
        (y_test - np.average(uq._preds(X_test), axis=0, weights=uq.w)) ** 2)))
    y_crit = float(np.quantile(y_ref, 0.95))
    exc = uq.exceedance(P, y_crit)
    pb = uq.pbox(P)
    rel = lambda est, tru: float(abs(est - tru) / abs(tru))
    rec = {
        "seed": seed, "protocol": "experiments/campaign.py (soft), N=20",
        "rmse": rmse,
        "weights": {m.name: float(w) for m, w in zip(a.models, uq.w)},
        "mean": {"est": float(mu.mean()), "true": float(y_ref.mean()),
                 "rel_err": rel(mu.mean(), y_ref.mean())},
        "std": {"est": float(mu.std()), "true": float(y_ref.std()),
                "rel_err": rel(mu.std(), y_ref.std())},
        "quantiles": {str(q): {"est": float(np.quantile(mu, q)),
                               "true": float(np.quantile(y_ref, q)),
                               "rel_err": rel(np.quantile(mu, q),
                                              np.quantile(y_ref, q))}
                      for q in (0.05, 0.5, 0.95)},
        "exceedance": {"threshold_q95": y_crit, "central": exc["central"],
                       "lower": exc["lower"], "upper": exc["upper"],
                       "true": float(np.mean(y_ref > y_crit))},
        "trajectory": traj,
    }
    json.dump(rec, open(_os.path.join(RES, f"representative_run_seed{seed}.json"),
                        "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("rmse", "mean", "std", "exceedance")},
                     indent=1))
    print("model-form fraction:", [round(t["model_form_fraction"], 3) for t in traj])

    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    sc = 1e-4                                          # plot in units of 1e4
    ax.fill_between(pb["y"] * sc, pb["F_lower"], pb["F_upper"], color=BLUE_LIGHT,
                    lw=0, label="p-box (all members)")
    ax.plot(pb["y"] * sc, pb["F_central"], color=BLUE, lw=1.4,
            label="Weighted mixture")
    ys = np.sort(y_ref)
    ax.plot(ys * sc, np.linspace(0, 1, len(ys)), color=INK, lw=1.0,
            ls=(0, (4, 2)), label="Reference")
    ax.axvline(y_crit * sc, color=MUTED, lw=0.7, ls=":")
    ax.text(y_crit * sc * 1.04, 0.62,
            f"$y_{{0.95}}$\n$P(Y>y_{{0.95}})$\n$\\in[{exc['lower']:.3f},\\,"
            f"{exc['upper']:.3f}]$\nreference {rec['exceedance']['true']:.3f}",
            ha="left", va="top", fontsize=6.5, color=INK)
    ax.set_xlim(0, float(np.quantile(y_ref, 0.999)) * sc)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Output level, $y$ ($\\times 10^4$)")
    ax.set_ylabel("Cumulative probability")
    ax.grid(color=GRID, lw=0.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=3,
              frameon=False, handlelength=1.6, columnspacing=0.8,
              borderaxespad=0.2, fontsize=6.5)
    _clean(ax)
    _save(fig, "fig_uq_pbox")


if __name__ == "__main__":
    todo = sys.argv[1:] or ["strip", "domain", "pbox"]
    for t in todo:
        {"strip": fig_strip, "domain": fig_domain, "pbox": fig_pbox}[t]()
    if "domain" in todo:
        json.dump(domain_summary(), open(_os.path.join(RES, "summary_domain.json"),
                                         "w"), indent=1)
