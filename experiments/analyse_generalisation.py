"""
analyse_generalisation.py — summary tables and paired tests for
campaign_generalisation.py.  Writes results/summary_generalisation.json and
prints a readable report.

Methods (column keys): an arm name, or <committee arm>:uni / :cv.
Paired comparisons use the two-sided Wilcoxon signed-rank test on log nRMSE
over the 30 shared seeds; "wins" counts seeds where the first method has the
lower nRMSE.
"""

import os, json, glob
import numpy as np
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
BENCHES = ["physics", "plume", "borehole", "ishigami"]
COMMITTEES = ("pas_orig", "pas_ard", "al_var", "al_ueigv")
METHODS = ["gp_orig", "gp_amp", "gp_ardnoamp", "gp_ard", "gp_m52", "gp_log", "al_gp",
           "pas_orig:uni", "pas_orig:cv", "pas_ard:uni", "pas_ard:cv",
           "al_var:cv", "al_ueigv:cv"]
METRICS = ["nrmse", "relerr_mean", "relerr_sd", "relerr_q95", "exc_abserr",
           "exc_covered", "max_weight"]
COMPARISONS = [
    ("gp_m52", "gp_orig", "GP specification: Matern-5/2 ARD + amplitude vs original"),
    ("gp_amp", "gp_orig", "amplitude hyperparameter alone"),
    ("pas_orig:cv", "pas_orig:uni", "aggregation, original members"),
    ("pas_ard:cv", "pas_ard:uni", "aggregation, corrected members"),
    ("pas_ard:cv", "gp_m52", "CV-weighted committee vs best single GP (same design)"),
    ("al_gp", "gp_m52", "adaptive vs passive, single GP"),
    ("al_var:cv", "pas_ard:cv", "adaptive vs passive, committee"),
    ("al_ueigv:cv", "al_var:cv", "spectral vs variance acquisition"),
    ("gp_log", "gp_m52", "log-transformed variables vs raw, single GP"),
    ("gp_log", "pas_ard:cv", "log-transformed GP vs CV-weighted committee"),
]


def load(bench, method):
    arm, _, agg = method.partition(":")
    f = os.path.join(RES, f"results_gen_{bench}_{arm}.json")
    if not os.path.exists(f):
        return None
    out = {}
    for r in json.load(open(f)):
        pre = f"{agg}_" if agg else ""
        out[r["seed"]] = {m: r.get(pre + m) for m in METRICS}
        out[r["seed"]]["top_member"] = r.get(pre + "top_member")
    return out


def summarise():
    S = {"medians": {}, "tests": {}, "n": {}}
    for b in BENCHES:
        S["medians"][b], S["tests"][b], S["n"][b] = {}, {}, {}
        data = {m: load(b, m) for m in METHODS}
        for m, d in data.items():
            if not d:
                continue
            S["n"][b][m] = len(d)
            S["medians"][b][m] = {
                k: (float(np.mean([v[k] for v in d.values()])) if k == "exc_covered"
                    else float(np.median([v[k] for v in d.values()])))
                for k in METRICS if all(v[k] is not None for v in d.values())}
        for a, c, label in COMPARISONS:
            if not data.get(a) or not data.get(c):
                continue
            seeds = sorted(set(data[a]) & set(data[c]))
            x = np.log([data[a][s]["nrmse"] for s in seeds])
            y = np.log([data[c][s]["nrmse"] for s in seeds])
            p = float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
            S["tests"][b][f"{a} vs {c}"] = {
                "label": label, "n": len(seeds), "p": p,
                "wins": int((x < y).sum()),
                "median_ratio": float(np.median(np.exp(x - y)))}
    # Holm adjustment over every paired test reported
    flat = [(b, k) for b in S["tests"] for k in S["tests"][b]]
    order = sorted(flat, key=lambda bk: S["tests"][bk[0]][bk[1]]["p"])
    m, running = len(flat), 0.0
    for r, (b, k) in enumerate(order):
        running = max(running, min(1.0, (m - r) * S["tests"][b][k]["p"]))
        S["tests"][b][k]["p_holm"] = running
    S["n_tests"] = m
    return S


PROP_ROWS = [("gp_orig", "GP, original kernel"),
             ("gp_m52", "GP, Mat\\'ern-5/2 ARD"),
             ("gp_log", "GP, log variables"),
             ("pas_ard:cv", "Committee, CV-weighted"),
             ("al_var:cv", "Committee, adaptive")]


def latex_propagation_table(S):
    """Rows for Table III of the extended abstract (medians over 30 seeds):
    relative error of mean and y95 in %, |P - 0.05| x 100; best per column
    in bold."""
    cols = [(b, k) for b in BENCHES
            for k in ("relerr_mean", "relerr_q95", "exc_abserr")]
    val = {(m, b, k): S["medians"][b].get(m, {}).get(k)
           for m, _ in PROP_ROWS for b, k in cols}
    best = {c: min(v for (m, b, k), v in val.items() if (b, k) == c and v is not None)
            for c in cols}
    lines = []
    for m, label in PROP_ROWS:
        cells = []
        for b, k in cols:
            v = val[(m, b, k)]
            if v is None:
                cells.append("--"); continue
            fmt = lambda x: f"{100 * x:.2f}" if 100 * x < 10 else f"{100 * x:.1f}"
            txt = fmt(v)
            cells.append(f"\\textbf{{{txt}}}" if txt == fmt(best[(b, k)]) else txt)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


NRMSE_ROWS = [("gp_orig", "GP, original kernel (amp. fixed)"),
              ("gp_amp", "GP, + amplitude"),
              ("gp_ardnoamp", "GP, + ARD"),
              ("gp_ard", "GP, + amplitude + ARD"),
              ("gp_m52", "GP, Mat\\'ern-5/2, amp. + ARD"),
              ("gp_log", "GP, Mat\\'ern-5/2, log variables"),
              ("al_gp", "GP, adaptive (posterior SD)"),
              ("pas_orig:uni", "Committee (orig. GPs), uniform"),
              ("pas_orig:cv", "Committee (orig. GPs), CV"),
              ("pas_ard:uni", "Committee, uniform"),
              ("pas_ard:cv", "Committee, CV"),
              ("al_var:cv", "Committee, adaptive (variance)"),
              ("al_ueigv:cv", "Committee, adaptive (UEigV)")]


def latex_nrmse_table(S):
    """Median nRMSE with interquartile range, every strategy x model."""
    lines = []
    for m, label in NRMSE_ROWS:
        cells = []
        for b in BENCHES:
            arm, _, agg = m.partition(":")
            f = os.path.join(RES, f"results_gen_{b}_{arm}.json")
            if not os.path.exists(f):
                cells.append("--"); continue
            key = f"{agg}_nrmse" if agg else "nrmse"
            v = np.array([r[key] for r in json.load(open(f))])
            q1, q2, q3 = np.quantile(v, [0.25, 0.5, 0.75])
            cells.append(f"{q2:.3g} ({q1:.2g}--{q3:.2g})")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


TEST_LABELS = {
    "gp_m52 vs gp_orig": "GP Mat\\'ern ARD + amp. vs original",
    "gp_amp vs gp_orig": "GP + amp. vs original",
    "pas_orig:cv vs pas_orig:uni": "CV vs uniform (orig. GPs)",
    "pas_ard:cv vs pas_ard:uni": "CV vs uniform",
    "pas_ard:cv vs gp_m52": "Committee (CV) vs GP Mat\\'ern",
    "al_gp vs gp_m52": "Adaptive vs passive GP",
    "al_var:cv vs pas_ard:cv": "Adaptive vs passive committee",
    "al_ueigv:cv vs al_var:cv": "UEigV vs variance acquisition",
    "gp_log vs gp_m52": "GP log variables vs raw",
    "gp_log vs pas_ard:cv": "GP log variables vs committee",
}


def latex_tests_table(S):
    """Median paired ratio, wins/30 and Holm-adjusted p for each comparison."""
    lines = []
    for a, c, _ in COMPARISONS:
        label = TEST_LABELS[f"{a} vs {c}"]
        key = f"{a} vs {c}"
        cells = []
        for b in BENCHES:
            t = S["tests"][b].get(key)
            if t is None:
                cells.append("--"); continue
            p = t["p_holm"]
            ptxt = "<10^{-6}" if p < 1e-6 else (f"{p:.1e}" if p < 1e-2 else f"{p:.2f}")
            ptxt = ptxt.replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-")
            if "times10" in ptxt:
                ptxt += "}"
            cells.append(f"{t['median_ratio']:.3g}; {t['wins']}; ${ptxt}$")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


if __name__ == "__main__":
    S = summarise()
    open(os.path.join(RES, "table_nrmse.tex"), "w").write(latex_nrmse_table(S) + "\n")
    open(os.path.join(RES, "table_tests.tex"), "w").write(latex_tests_table(S) + "\n")
    tex = latex_propagation_table(S)
    open(os.path.join(RES, "table_propagation.tex"), "w").write(tex + "\n")
    print(tex)
    json.dump(S, open(os.path.join(RES, "summary_generalisation.json"), "w"), indent=1)
    for b in BENCHES:
        print(f"\n=== {b} " + "=" * 60)
        print(f"{'method':14s} {'n':>3s} {'nRMSE':>8s} {'mean':>7s} {'SD':>7s} "
              f"{'q95':>7s} {'|dP|':>7s} {'cov':>5s}")
        for m, v in S["medians"][b].items():
            cov = f"{v['exc_covered']:.0%}" if ":" in m else ""
            print(f"{m:14s} {S['n'][b][m]:3d} {v['nrmse']:8.3g} {v['relerr_mean']:7.2%} "
                  f"{v['relerr_sd']:7.2%} {v['relerr_q95']:7.2%} {v['exc_abserr']:7.4f} {cov:>5s}")
        for k, t in S["tests"][b].items():
            print(f"  {k:28s} ratio {t['median_ratio']:8.3g}  wins {t['wins']:2d}/{t['n']}"
                  f"  p={t['p']:.2g} holm={t['p_holm']:.2g}   ({t['label']})")
