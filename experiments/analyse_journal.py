"""
analyse_journal.py — statistics, tables and figures of the journal study.

Reads results/results_gen_*.json (campaign_generalisation.py) and
results/results_cov_*.json (campaign_coverage.py) and writes
results/journal/summary.json plus LaTeX tables (results/journal/tab_*.tex)
and figures (figures/journal/*.pdf).

Statistics
  * Median over the 30 seeds with a 95 % percentile-bootstrap confidence
    interval (10 000 resamples of the seeds, RandomState(0)).
  * Paired comparisons A vs B on the same seeds: effect size = median over
    seeds of the ratio A/B of the metric (nRMSE, or the absolute error of an
    exceedance probability), with a paired bootstrap 95 % CI; two-sided
    Wilcoxon signed-rank test on log ratios (zero differences dropped,
    Pratt); Holm adjustment over every test in the family (all comparisons x
    benchmarks x budgets reported in summary.json).
  * Coverage: share of test points (pointwise) or of the reference
    distribution (GUM S1 interval) inside the 95 % interval, averaged over
    seeds; for derived quantities, the share of seeds whose interval covers
    the true value, with a Wilson 95 % interval.

Usage:  python experiments/analyse_journal.py            (write everything)
        python experiments/analyse_journal.py --check    (recompute and
                compare with the committed summary.json; exit 1 on change)
"""

import os, sys, json, glob
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "journal")
FIG = os.path.join(ROOT, "figures", "journal")
B_BOOT = 10_000

BENCHES = ["physics", "plume", "borehole", "ishigami", "otl", "piston",
           "wingweight", "otl20", "borehole_gum", "plume_gum", "flood", "cfd"]
LABEL = {"physics": "Physics", "plume": "Plume", "borehole": "Borehole",
         "ishigami": "Ishigami", "otl": "OTL circuit", "piston": "Piston",
         "borehole_gum": "Borehole (GUM inputs)", "plume_gum": "Plume (GUM inputs)",
         "flood": "Flood", "cfd": "CFD dispersion", "wingweight": "Wing weight",
         "otl20": "OTL circuit, 20 inputs"}
METHODS = {
    "gp_orig": "GP, original kernel", "gp_amp": "GP + amplitude",
    "gp_m52": "GP Mat\\'ern ARD", "gp_quad": "GP + quadratic trend",
    "gp_log": "GP, log variables", "gp_quadlog": "GP + quad. trend, log",
    "spce": "Sparse PCE", "spce_log": "Sparse PCE, log",
    "pas_ard:cv": "Committee (CV)", "pas_ard:uni": "Committee (uniform)",
    "al_gp": "GP, adaptive (SD)", "al_var:cv": "Committee, adaptive",
    "al_ueigv:cv": "Committee, UEigV", "ak_m52": "GP, AK-MCS",
    "ak_log": "GP log, AK-MCS", "akq_m52": "GP, AK-MCS at y95",
    "akq_log": "GP log, AK-MCS at y95"}
# (A, B, metric, label): effect = median of A/B over paired seeds
COMPARISONS = [
    ("gp_m52", "gp_orig", "nrmse", "kernel: amplitude + ARD vs original"),
    ("gp_log", "gp_m52", "nrmse", "log variables vs raw"),
    ("gp_quad", "gp_m52", "nrmse", "quadratic trend vs none"),
    ("gp_quadlog", "gp_log", "nrmse", "quadratic trend vs none (log)"),
    ("spce", "gp_m52", "nrmse", "sparse PCE vs GP"),
    ("spce_log", "gp_log", "nrmse", "sparse PCE vs GP (log)"),
    ("pas_ard:cv", "gp_quad", "nrmse", "committee vs GP + trend"),
    ("pas_ard:cv", "pas_ard:uni", "nrmse", "CV weights vs uniform"),
    ("al_gp", "gp_m52", "nrmse", "adaptive vs passive GP"),
    ("al_var:cv", "pas_ard:cv", "nrmse", "adaptive vs passive committee"),
    ("al_ueigv:cv", "al_var:cv", "nrmse", "UEigV vs variance"),
    ("ak_m52", "gp_m52", "thr_abserr", "AK-MCS vs passive, exceedance"),
    ("ak_log", "gp_log", "thr_abserr", "AK-MCS vs passive, exceedance (log)"),
    ("akq_m52", "gp_m52", "exc_abserr", "AK-MCS at $y_{0.95}$ vs passive"),
    ("akq_log", "gp_log", "exc_abserr", "AK-MCS at $y_{0.95}$ vs passive (log)"),
    ("gp_log", "gp_m52", "thr_abserr", "log vs raw, exceedance"),
]
BUDGETS = [10, 20, 40, 80]
SWEEP = ["gp_orig", "gp_m52", "gp_quad", "gp_log", "spce", "spce_log",
         "pas_ard:cv", "al_gp", "al_var:cv"]
SWEEP_TESTS = [("al_gp", "gp_m52"), ("al_var:cv", "pas_ard:cv"),
               ("gp_log", "gp_m52"), ("spce", "gp_m52")]


# ── loading ──────────────────────────────────────────────────────────────
def load(bench, method, N=20, metric="nrmse"):
    """{seed: value} for one arm (method "arm" or "arm:uni"/"arm:cv")."""
    arm, _, agg = method.partition(":")
    f = os.path.join(RES, f"results_gen_{bench}_{arm}" + ("" if N == 20 else f"_N{N}") + ".json")
    if not os.path.exists(f):
        return {}
    key = f"{agg}_{metric}" if agg else metric
    return {r["seed"]: r[key] for r in json.load(open(f)) if r.get(key) is not None}


# ── statistics ───────────────────────────────────────────────────────────
def boot_median(v, rs=None):
    v = np.asarray(v, float)
    rs = rs or np.random.RandomState(0)
    idx = rs.randint(0, len(v), (B_BOOT, len(v)))
    m = np.median(v[idx], axis=1)
    return float(np.median(v)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 5:
        return None
    x = np.array([a[s] for s in seeds], float)
    y = np.array([b[s] for s in seeds], float)
    eps = 1e-12                       # exceedance errors can be exactly zero
    r = np.log(np.maximum(x, eps)) - np.log(np.maximum(y, eps))
    med, lo, hi = boot_median(r)
    p = float(wilcoxon(r, zero_method="pratt").pvalue) if np.any(r != 0) else 1.0
    return {"n": len(seeds), "ratio": float(np.exp(med)), "ci": [float(np.exp(lo)), float(np.exp(hi))],
            "wins": int((x < y).sum()), "losses": int((x > y).sum()), "p": p}


def holm(tests):
    """Adds p_holm to every dict in the list (in place)."""
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
    m, running = len(tests), 0.0
    for r, i in enumerate(order):
        running = max(running, min(1.0, (m - r) * tests[i]["p"]))
        tests[i]["p_holm"] = running


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


# ── summaries ────────────────────────────────────────────────────────────
def summarise():
    S = {"n20": {}, "tests": [], "sweep": {}, "coverage": {}, "exceedance": {}}
    for b in BENCHES:
        S["n20"][b] = {}
        for m in METHODS:
            d = load(b, m)
            if d:
                med, lo, hi = boot_median(list(d.values()))
                S["n20"][b][m] = {"n": len(d), "median": med, "ci": [lo, hi]}
        S["exceedance"][b] = {}
        for m in METHODS:
            d = load(b, m, metric="thr_abserr")
            if d:
                med, lo, hi = boot_median(list(d.values()))
                cov = load(b, m, metric="thr_covered")
                S["exceedance"][b][m] = {"n": len(d), "median_abserr": med, "ci": [lo, hi],
                                         "p_true": next(iter(load(b, m, metric="thr_true").values()), None),
                                         "envelope_coverage": (float(np.mean(list(cov.values())))
                                                               if ":" in m and cov else None)}
        S.setdefault("prop", {})[b] = {}
        for m in METHODS:
            row = {}
            for k in ("relerr_mean", "relerr_sd", "relerr_q95", "thr_abserr", "thr_hat"):
                d = load(b, m, metric=k)
                if d:
                    row[k] = float(np.median(list(d.values())))
            if row:
                S["prop"][b][m] = row
        for a, c, metric, label in COMPARISONS:
            t = paired(load(b, a, metric=metric), load(b, c, metric=metric))
            if t:
                S["tests"].append({"bench": b, "N": 20, "a": a, "b": c, "metric": metric,
                                   "label": label, **t})
    for b in BENCHES:
        S["sweep"][b] = {}
        for m in SWEEP:
            row = {}
            for N in BUDGETS:
                d = load(b, m, N)
                if len(d) >= 5:
                    med, lo, hi = boot_median(list(d.values()))
                    row[str(N)] = {"n": len(d), "median": med, "ci": [lo, hi]}
            if row:
                S["sweep"][b][m] = row
        for a, c in SWEEP_TESTS:
            for N in (10, 40, 80):
                t = paired(load(b, a, N), load(b, c, N))
                if t:
                    S["tests"].append({"bench": b, "N": N, "a": a, "b": c, "metric": "nrmse",
                                       "label": f"{METHODS[a]} vs {METHODS[c]}", **t})
    holm(S["tests"])
    S["n_tests"] = len(S["tests"])
    S["coverage"] = coverage_summary()
    S["coverage_ranges"] = coverage_ranges(S)
    S["selection"] = selection_summary()
    return S


def selection_summary():
    """Evaluate the selection step of the procedure: per seed, the candidate
    with the smallest LOO RMSE (results_sel_*.json) is taken and its
    validation errors are read from the campaign records of the same seed."""
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, "results_sel_*.json"))):
        recs = json.load(open(f))
        if not recs or recs[0]["N"] != 20 or len(recs) < 5 or recs[0]["benchmark"] not in BENCHES:
            continue
        b = recs[0]["benchmark"]
        cands = [k[len("loo_rmse_"):] for k in recs[0] if k.startswith("loo_rmse_")]
        data = {c: {m: load(b, c, metric=m) for m in ("nrmse", "relerr_q95", "thr_abserr")}
                for c in cands}
        cov = {}
        for c in cands:
            fc = os.path.join(RES, f"results_cov_{b}_{c}.json")
            cov[c] = {x["seed"]: x for x in json.load(open(fc))} if os.path.exists(fc) else {}
        chosen, rows, covrows = {}, [], []
        for r in recs:
            sd = r["seed"]
            if not all(sd in data[c]["nrmse"] for c in cands):
                continue
            c = min(cands, key=lambda c: r[f"loo_rmse_{c}"])
            chosen[c] = chosen.get(c, 0) + 1
            oracle = min(data[k]["nrmse"][sd] for k in cands)
            rows.append({"seed": sd, "nrmse": data[c]["nrmse"][sd], "oracle": oracle,
                         "gp_m52": data["gp_m52"]["nrmse"][sd],
                         "relerr_q95": data[c]["relerr_q95"][sd],
                         "thr_abserr": data[c]["thr_abserr"][sd]})
            if sd in cov[c]:
                covrows.append(cov[c][sd])
        if not rows:
            continue
        v = lambda k: np.array([x[k] for x in rows])
        med, lo, hi = boot_median(np.log(v("nrmse") / v("oracle")))
        out[b] = {"n": len(rows), "chosen": chosen,
                  "nrmse_median": float(np.median(v("nrmse"))),
                  "ratio_to_oracle": float(np.exp(med)), "ratio_to_oracle_ci": [float(np.exp(lo)), float(np.exp(hi))],
                  "share_oracle": float(np.mean(v("nrmse") == v("oracle"))),
                  "ratio_to_gp_m52": float(np.median(v("nrmse") / v("gp_m52"))),
                  "relerr_q95_median": float(np.median(v("relerr_q95"))),
                  "thr_abserr_median": float(np.median(v("thr_abserr"))),
                  "per_seed_nrmse": {str(x["seed"]): x["nrmse"] for x in rows}}
        if len(covrows) == len(rows):           # the whole procedure, end to end
            m = lambda k: float(np.mean([x[k] for x in covrows]))
            kk = lambda k: int(np.sum([x[k] for x in covrows]))
            out[b]["procedure_coverage"] = {
                "ci_resample_cov": m("ci_resample_cov"), "ci_plain_cov": m("ci_plain_cov"),
                "pw_jkplus_cov": m("pw_jkplus_cov"),
                **{f"dq_{q}_jack_cov": m(f"dq_{q}_jack_cov") for q in ("mean", "sd", "q95", "p_thr")},
                **{f"dq_{q}_jack_wilson": wilson(kk(f"dq_{q}_jack_cov"), len(covrows))
                   for q in ("mean", "q95", "p_thr")}}
    return out


def coverage_summary():
    C = {}
    for f in sorted(glob.glob(os.path.join(RES, "results_cov_*.json"))):
        recs = json.load(open(f))
        if not recs:
            continue
        b, s, N = recs[0]["benchmark"], recs[0]["surrogate"], recs[0]["N"]
        if N != 20 or "design" in recs[0] or b not in BENCHES:   # main study only
            continue
        row = {"n": len(recs)}
        common = set.intersection(*(set(r) for r in recs))   # records being augmented
        for k in recs[0]:            # may hold new fields in some seeds only
            if k not in common or k == "loo_rmse_orig" or any(
                    t in k for t in ("_gpt", "_gpmc", "_gpcal", "gpt_", "gpmc_", "gpcal_",
                                     "augment_")):
                continue             # research-plan fields: analyse_completion.py
            v = [r[k] for r in recs]
            if k.startswith(("pw_", "ci_")) and isinstance(v[0], float):
                row[k] = float(np.mean(v))
            elif k.endswith("_cov") and k.startswith("dq_"):
                row[k] = float(np.mean(v)); row[k + "_wilson"] = wilson(int(np.sum(v)), len(v))
        row["nrmse_median"] = float(np.median([r["nrmse"] for r in recs]))
        C.setdefault(b, {})[s] = row
    return C


# ── LaTeX tables ─────────────────────────────────────────────────────────
def fmt(x):
    return f"{x:.2g}" if x < 0.01 else (f"{x:.3f}" if x < 1 else f"{x:.3g}")


def tab_n20(S, methods):
    """Median nRMSE, benchmark x method, best per row in bold. A method
    "ak" stands for AK-MCS on the recommended scale (log if positive)."""
    lines = []
    for b in BENCHES:
        row = S["n20"].get(b, {})
        if not row:
            continue
        pick = lambda m: ("ak_log" if recommended(b) == "gp_log" else "ak_m52") if m == "ak" else m
        vals = {m: row[pick(m)]["median"] for m in methods if pick(m) in row}
        best = min(vals.values())
        cells = []
        for m in methods:
            if m not in vals:
                cells.append("--"); continue
            t = fmt(vals[m])
            cells.append(f"\\textbf{{{t}}}" if vals[m] == best else t)
        lines.append(f"{LABEL[b]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def ptxt(p):
    if p < 1e-4:
        return "$<10^{-4}$"
    return f"{p:.1g}" if p < 0.01 else f"{p:.2f}"


SHORT = {"physics": "Phys.", "plume": "Plume", "borehole": "Bore.", "ishigami": "Ishi.",
         "otl": "OTL", "piston": "Pist.", "borehole_gum": "Bore.G", "plume_gum": "Plume.G",
         "flood": "Flood", "cfd": "CFD", "wingweight": "Wing", "otl20": "OTL20"}


def tab_tests(S, N=20):
    """Matrix comparison x model: median paired ratio; bold if the Holm-
    adjusted p < 0.05 (two-sided Wilcoxon). Below 1 favours the first."""
    lines = []
    for a, c, metric, label in COMPARISONS:
        cells, any_ = [], False
        for b in BENCHES:
            t = next((t for t in S["tests"] if (t["a"], t["b"], t["metric"], t["N"], t["bench"])
                      == (a, c, metric, N, b)), None)
            if t is None:
                cells.append("--"); continue
            any_ = True
            x = t["ratio"]
            r = f"{x:.2g}" if x < 0.1 else (f"{x:.3f}" if abs(x - 1) < 0.01 else f"{x:.2f}")
            cells.append(f"\\textbf{{{r}}}" if t["p_holm"] < 0.05 else r)
        if any_:
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def tab_tests_long(S, N=20):
    """Supplementary table: every paired comparison with CI, wins, Holm p."""
    lines = []
    for a, c, metric, label in COMPARISONS:
        for t in S["tests"]:
            if (t["a"], t["b"], t["metric"], t["N"]) == (a, c, metric, N):
                lines.append(f"{label} & {LABEL[t['bench']]} & {t['ratio']:.3g} "
                             f"[{t['ci'][0]:.3g}, {t['ci'][1]:.3g}] & {t['wins']}/{t['n']} & "
                             f"{ptxt(t['p_holm'])} \\\\")
    return "\n".join(lines) + "\n"


POSITIVE = {"plume", "borehole", "otl", "piston", "borehole_gum", "plume_gum", "cfd",
            "wingweight", "otl20"}


def recommended(b):
    """The GP the paper recommends for a model: log variables if positive."""
    return "gp_log" if b in POSITIVE else "gp_m52"


def tab_prop(S):
    """Validation error of reported quantities, median over seeds, for the
    recommended GP (passive and AK-MCS) and the CV-weighted committee:
    relative error (%) of mean, SD, y_0.95; absolute error (x100) of the
    exceedance probability at the model's threshold."""
    lines = []
    for b in BENCHES:
        g = recommended(b); ak = "ak_log" if g == "gp_log" else "ak_m52"
        P = S["prop"].get(b, {})
        if g not in P:
            continue
        pct = lambda m, k: (f"{100 * P[m][k]:.2f}" if 100 * P[m][k] < 10 else f"{100 * P[m][k]:.1f}") \
            if m in P and k in P[m] else "--"
        lines.append(f"{LABEL[b]} & {pct(g, 'relerr_mean')} & {pct(g, 'relerr_sd')} & "
                     f"{pct(g, 'relerr_q95')} & {pct(g, 'thr_abserr')} & {pct(ak, 'thr_abserr')} & "
                     f"{pct('pas_ard:cv', 'relerr_q95')} & {pct('pas_ard:cv', 'thr_abserr')} \\\\")
    return "\n".join(lines) + "\n"


def tab_coverage(S):
    """Coverage (%) of nominal 95 % intervals for the recommended GP of each
    model: pointwise (GP, CV-normal, jackknife+), GUM S1 coverage interval
    (surrogate alone, + LOO normal error, + LOO residuals, + GP SD), and the
    share of seeds whose interval for y_0.95 and for P(Y>T) contains the
    true value (GP joint posterior paths, jackknife)."""
    keys = ["pw_gp_cov", "pw_cvnorm_cov", "pw_jkplus_cov", "ci_plain_cov",
            "ci_normal_cov", "ci_resample_cov", "ci_gp_cov",
            "dq_q95_gp_cov", "dq_q95_jack_cov", "dq_p_thr_gp_cov", "dq_p_thr_jack_cov"]
    lines = []
    for b in BENCHES:
        s = recommended(b)
        r = S["coverage"].get(b, {}).get(s)
        if not r:
            continue
        cells = [f"{100 * r[k]:.0f}" if k in r else "--" for k in keys]
        name = LABEL[b] + (" (log)" if s == "gp_log" else "")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


def coverage_ranges(S, exclude=("cfd",)):
    """Range over the model x surrogate combinations of each coverage key
    (analytical models; the CFD case is reported separately)."""
    out = {}
    for b, row in S["coverage"].items():
        if b in exclude:
            continue
        for s, r in row.items():
            for k, v in r.items():
                if k.endswith("_cov") and isinstance(v, float):
                    out.setdefault(k, []).append(v)
    return {k: {"min": float(min(v)), "median": float(np.median(v)), "max": float(max(v)),
                "n": len(v)} for k, v in out.items()}


# ── figures ──────────────────────────────────────────────────────────────
def plain(m):
    return METHODS[m].replace("Mat\\'ern", "Mat\u00e9rn")


def fig_sweep(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    benches = [b for b in BENCHES
               if any(len(r) >= 2 for r in S["sweep"].get(b, {}).values())]
    if not benches:
        return
    from matplotlib.ticker import FuncFormatter, NullFormatter
    style = {"gp_orig": ("0.6", ":", "x"), "gp_m52": ("C0", "-", "o"),
             "gp_quad": ("C1", "-", "s"), "gp_log": ("C2", "-", "^"),
             "spce": ("C3", "--", "D"), "spce_log": ("C3", ":", "d"),
             "pas_ard:cv": ("C4", "-", "v"), "al_gp": ("C0", "--", "o"),
             "al_var:cv": ("C4", "--", "v")}
    n = len(benches); cols = 3; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.2, 2.1 * rows), squeeze=False)
    for ax, b in zip(axes.flat, benches):
        for m, row in S["sweep"][b].items():
            Ns = sorted(int(k) for k in row)
            med = [row[str(k)]["median"] for k in Ns]
            lo = [row[str(k)]["ci"][0] for k in Ns]; hi = [row[str(k)]["ci"][1] for k in Ns]
            c, ls, mk = style[m]
            ax.plot(Ns, med, color=c, ls=ls, marker=mk, ms=3, lw=1, label=plain(m))
            ax.fill_between(Ns, lo, hi, color=c, alpha=0.12, lw=0)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xticks(BUDGETS); ax.set_xticklabels(BUDGETS)
        ax.set_title(LABEL[b], fontsize=8); ax.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=0.3, lw=0.5)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("model runs $N$", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("nRMSE", fontsize=8)
    handles = {}
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    order = [plain(m) for m in SWEEP if plain(m) in handles]
    fig.legend([handles[l] for l in order], order, loc="lower center", ncol=5,
               fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig_budget_sweep.pdf"))
    plt.close(fig)


def fig_coverage(S):
    """Coverage of nominal 95 % intervals over every model x surrogate."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("(a) pointwise prediction interval",
               [("pw_gp_cov", "GP"), ("pw_cvnorm_cov", "CV-normal"), ("pw_jkplus_cov", "jackknife+")]),
              ("(b) GUM S1 95 % coverage interval",
               [("ci_plain_cov", "surrogate\nonly"), ("ci_gp_cov", "+ GP SD"),
                ("ci_normal_cov", "+ LOO\nnormal"), ("ci_resample_cov", "+ LOO\nresiduals")]),
              ("(c) interval for $y_{0.95}$",
               [("dq_q95_gp_cov", "GP\nposterior"), ("dq_q95_jack_cov", "jackknife")])]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [3, 4, 2]})
    rs = np.random.RandomState(1)
    for ax, (title, keys) in zip(axes, panels):
        for j, (k, lab) in enumerate(keys):
            for b, row in S["coverage"].items():
                for s, r in row.items():
                    if k in r:
                        c = "C3" if s == "gp_orig" else ("C0" if s.startswith("gp") else "C2")
                        mk, ms = ("D", 3.2) if b == "cfd" else ("o", 2.2)
                        ax.plot(j + rs.uniform(-0.18, 0.18), r[k], mk, ms=ms, color=c, alpha=0.7,
                                mec="k" if b == "cfd" else c, mew=0.4)
        ax.axhline(0.95, color="k", lw=0.7, ls="--")
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys], fontsize=6.5)
        ax.set_title(title, fontsize=7.5); ax.tick_params(labelsize=7)
        ax.set_ylim(0, 1.02); ax.grid(axis="y", alpha=0.3, lw=0.5)
    axes[0].set_ylabel("coverage", fontsize=8)
    axes[2].set_ylabel("share of seeds covered", fontsize=7)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, ms=3, label=l) for c, l in
         [("C0", "GP (Matérn, quad. trend, log)"), ("C3", "GP, original kernel"), ("C2", "sparse PCE")]]
    h.append(Line2D([], [], marker="D", ls="", color="0.6", mec="k", mew=0.4, ms=3.5, label="CFD case"))
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig_coverage.pdf"))
    plt.close(fig)


def fig_n20(S):
    """nRMSE relative to the GP Matern ARD baseline, all benchmarks, with the
    candidate chosen by LOO error (the proposed procedure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["gp_orig", "gp_quad", "gp_log", "gp_quadlog", "spce", "spce_log",
            "pas_ard:cv", "al_gp", "al_var:cv", "loo"]
    benches = [b for b in BENCHES if "gp_m52" in S["n20"].get(b, {})]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    w = 0.84 / len(show)
    for j, m in enumerate(show):
        xs, ys, lo, hi = [], [], [], []
        for i, b in enumerate(benches):
            if m == "loo":
                ps = S["selection"].get(b, {}).get("per_seed_nrmse")
                a = {int(k): v for k, v in ps.items()} if ps else {}
            else:
                a = load(b, m)
            t = paired(a, load(b, "gp_m52")) if a else None
            if t:
                xs.append(i + (j - len(show) / 2 + 0.5) * w); ys.append(t["ratio"])
                lo.append(t["ratio"] - t["ci"][0]); hi.append(t["ci"][1] - t["ratio"])
        lab = "LOO-selected (procedure)" if m == "loo" else plain(m)
        kw = dict(color="k", marker="*", ms=6) if m == "loo" else dict(marker="o", ms=3)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=kw.pop("marker"), lw=0.8,
                    capsize=1.5, label=lab, **kw)
    for i in range(1, len(benches)):
        ax.axvline(i - 0.5, color="0.85", lw=0.5)
    ax.axhline(1, color="k", lw=0.6)
    ax.set_yscale("log"); ax.set_xticks(range(len(benches))); ax.set_xlim(-0.5, len(benches) - 0.5)
    ax.set_xticklabels([LABEL[b] for b in benches], rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("nRMSE / nRMSE(GP Matérn ARD)", fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.legend(fontsize=6.5, ncol=4, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.32))
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "fig_relative_nrmse.pdf"))
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────
def _close(a, b, path="summary"):
    if isinstance(a, dict):
        ok = a.keys() == b.keys()
        return ok and all(_close(a[k], b[k], f"{path}.{k}") for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y, path) for x, y in zip(a, b))
    if isinstance(a, float):
        return bool(np.isclose(a, b, rtol=1e-6, atol=1e-12, equal_nan=True))
    return a == b


if __name__ == "__main__":
    S = summarise()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "summary.json")
    if "--check" in sys.argv:
        old = json.load(open(path))
        same = _close(json.loads(json.dumps(S)), old)
        print("summary.json reproduced" if same else "summary.json DIFFERS from the records")
        sys.exit(0 if same else 1)
    json.dump(S, open(path, "w"), indent=1)
    main_methods = ["gp_orig", "gp_m52", "gp_quad", "gp_log", "gp_quadlog", "spce",
                    "spce_log", "pas_ard:cv", "al_gp", "al_var:cv", "ak"]
    open(os.path.join(OUT, "tab_n20.tex"), "w").write(tab_n20(S, main_methods))
    open(os.path.join(OUT, "tab_tests.tex"), "w").write(tab_tests(S))
    open(os.path.join(OUT, "tab_tests_long.tex"), "w").write(tab_tests_long(S))
    open(os.path.join(OUT, "tab_coverage.tex"), "w").write(tab_coverage(S))
    open(os.path.join(OUT, "tab_prop.tex"), "w").write(tab_prop(S))
    fig_sweep(S); fig_n20(S); fig_coverage(S)
    print(f"{S['n_tests']} paired tests; significant after Holm: "
          f"{sum(t['p_holm'] < 0.05 for t in S['tests'])}")
    for t in S["tests"]:
        print(f"N={t['N']:2d} {t['bench']:12s} {t['a']:12s} vs {t['b']:12s} [{t['metric']}] "
              f"ratio {t['ratio']:.3g} [{t['ci'][0]:.3g},{t['ci'][1]:.3g}] "
              f"wins {t['wins']}/{t['n']} p_holm {t['p_holm']:.2g}")
