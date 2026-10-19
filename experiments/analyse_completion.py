"""
analyse_completion.py — statistics of the runs of docs/RESEARCH_PLAN.md
(Tiers 1 and 2), with the thresholds fixed in that plan before the runs.

Writes results/journal/completion.json and the LaTeX tables
tab_bayesgp.tex (T1.2) and tab_budgets.tex (T1.1). Sections that have no
records yet are left out, so the script can be run after every stage.

Usage: python experiments/analyse_completion.py [--check]
--check recomputes completion.json and compares it with the stored file.
"""

import os, sys, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import numpy as np
from scipy.stats import spearmanr
from analyse_journal import (RES, OUT, BENCHES, LABEL, POSITIVE, boot_median, paired,
                             holm, wilson, load, _close)

GP_CANDS = ("gp_m52", "gp_quad", "gp_log", "gp_quadlog")
VARIANTS = ("gp", "gpt", "gpmc", "gpcal")             # GP-based intervals
NOMINAL_MIN = 26                                       # of 30 seeds (plan)
MARGIN, MARGIN_STRICT = (0.8, 1.25), (0.9, 1 / 0.9)


def jload(name):
    f = os.path.join(RES, name)
    return json.load(open(f)) if os.path.exists(f) else None


def suffix(N, design="lhs"):
    return ("" if N == 20 else f"_N{N}") + ("" if design == "lhs" else f"_d{design}")


def cands(b):
    return ["gp_m52", "gp_quad", "spce"] + (["gp_log", "gp_quadlog", "spce_log"]
                                            if b in POSITIVE else [])


def trend(b):
    return "gp_quadlog" if b in POSITIVE else "gp_quad"


def cov_records(b, s, N=20, design="lhs"):
    r = jload(f"results_cov_{b}_{s}{suffix(N, design)}.json")
    return {x["seed"]: x for x in r} if r else {}


# ── T1.2 Bayesian and recalibrated GP intervals ──────────────────────────
def bayes_gp():
    out = {}
    for b in BENCHES:
        for s in GP_CANDS:
            R = list(cov_records(b, s).values())
            if not R or "pw_gpt_cov" not in R[0]:
                continue
            row = {"n": len(R)}
            for v in VARIANTS + ("jkplus",):
                k = f"pw_{v}_cov"
                if k in R[0]:
                    row[k] = float(np.mean([x[k] for x in R]))
                    row[f"pw_{v}_width"] = float(np.mean([x[f"pw_{v}_width"] for x in R]))
            for v in VARIANTS + ("resample", "plain"):
                k = f"ci_{v}_cov"
                if k in R[0]:
                    row[k] = float(np.mean([x[k] for x in R]))
            for q in ("mean", "sd", "q95", "p_thr"):
                for v in VARIANTS + ("jack",):
                    k = f"dq_{q}_{v}_cov"
                    if k in R[0]:
                        row[f"{k}_count"] = int(np.sum([x[k] for x in R]))
            if "gpmc_tau" in R[0]:
                row["gpmc_tau_median"] = float(np.median([x["gpmc_tau"] for x in R]))
                row["gpmc_acceptance_median"] = float(np.median([x["gpmc_acceptance"] for x in R]))
            out.setdefault(b, {})[s] = row
    rows = [r for b in out for r in out[b].values()]
    dec = {}
    for v in ("gpt", "gpmc", "gpcal"):
        have = [r for r in rows if f"pw_{v}_cov" in r]
        if not have:
            continue
        pw_ok = all(r[f"pw_{v}_cov"] >= 0.93 for r in have)
        dq_ok = all(r.get(f"dq_{q}_{v}_cov_count", 0) >= NOMINAL_MIN
                    for r in have for q in ("mean", "sd", "q95", "p_thr"))
        dec[v] = {"n_rows": len(have), "pointwise_ge_93_all": pw_ok,
                  "derived_nominal_all": dq_ok, "valid": pw_ok and dq_ok,
                  "pw_cov_range": [min(r[f"pw_{v}_cov"] for r in have),
                                   max(r[f"pw_{v}_cov"] for r in have)],
                  "dq_q95_count_median": float(np.median([r[f"dq_q95_{v}_cov_count"] for r in have])),
                  "dq_q95_count_min": int(min(r[f"dq_q95_{v}_cov_count"] for r in have))}
    return {"rows": out, "decision": dec}


# ── T1.1 budgets: the procedure (selection + LOO contribution) per N ─────
def budgets():
    """At N = 20 the LOO errors come from campaign_selection.py, at other N
    from the coverage records (loo_rmse_orig, same definition)."""
    out = {}
    for N in (10, 20, 40, 80):
        for b in BENCHES:
            cov = {c: cov_records(b, c, N) for c in cands(b)}
            if N == 20:
                sel = jload(f"results_sel_{b}.json")
                cs = [c for c in cands(b) if len(cov[c]) >= 5 and sel and f"loo_rmse_{c}" in sel[0]]
            else:
                cs = [c for c in cands(b) if len(cov[c]) >= 5
                      and all("loo_rmse_orig" in x for x in cov[c].values())]
                seeds = sorted(set.intersection(*[set(cov[c]) for c in cs])) if cs else []
                sel = [{"seed": sd, **{f"loo_rmse_{c}": cov[c][sd]["loo_rmse_orig"] for c in cs}}
                       for sd in seeds]
            if not cs or not sel:
                continue
            gen = {c: load(b, c, N=N) for c in cs}
            rows = []
            for r in sel:
                sd = r["seed"]
                if not all(sd in cov[c] and sd in gen[c] for c in cs):
                    continue
                c = min(cs, key=lambda c: r[f"loo_rmse_{c}"])
                x = cov[c][sd]
                rows.append({"chosen": c, "ci_resample_cov": x["ci_resample_cov"],
                             "ci_plain_cov": x["ci_plain_cov"], "pw_jkplus_cov": x["pw_jkplus_cov"],
                             "pw_jkplus_unbounded": bool(np.isinf(x["pw_jkplus_width"])),
                             "dq_q95_jack_cov": x["dq_q95_jack_cov"],
                             "dq_p_thr_jack_cov": x["dq_p_thr_jack_cov"],
                             "nrmse": gen[c][sd], "oracle": min(gen[k][sd] for k in cs)})
            if len(rows) < 5:
                continue
            v = lambda k: np.array([x[k] for x in rows], float)
            med, lo, hi = boot_median(np.log(v("nrmse") / v("oracle")))
            out.setdefault(str(N), {})[b] = {
                "n": len(rows), "candidates": cs,
                "ci_resample_cov": float(v("ci_resample_cov").mean()),
                "ci_plain_cov": float(v("ci_plain_cov").mean()),
                "pw_jkplus_cov": float(v("pw_jkplus_cov").mean()),
                "pw_jkplus_unbounded_share": float(v("pw_jkplus_unbounded").mean()),
                "dq_q95_jack_count": int(v("dq_q95_jack_cov").sum()),
                "dq_p_thr_jack_count": int(v("dq_p_thr_jack_cov").sum()),
                "share_designs_below_90": float(np.mean(v("ci_resample_cov") < 0.90)),
                "ratio_to_oracle": float(np.exp(med)), "ratio_to_oracle_ci": [float(np.exp(lo)), float(np.exp(hi))]}
    allc = [r["ci_resample_cov"] for n in out.values() for r in n.values()]
    dec = {"holds_all": bool(allc) and min(allc) >= 0.90,
           "min_cov": min(allc) if allc else None,
           "fails": sorted({(N, b) for N, n in out.items() for b, r in n.items()
                            if r["ci_resample_cov"] < 0.90})}
    dec["fails"] = [list(x) for x in dec["fails"]]
    return {"rows": out, "decision": dec}


# ── T1.3 OpenTURNS cross-check and PC-Kriging (T2.3) ─────────────────────
def crosscheck():
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, "results_ot_*.json"))):
        R = json.load(open(f))
        b = R[0]["benchmark"]
        ot = lambda k: {x["seed"]: x[k] for x in R if x.get(k) is not None}
        sel = jload(f"results_sel_{b}.json")
        chosen = {}
        if sel:
            data = {c: load(b, c) for c in cands(b)}
            for r in sel:
                c = min(cands(b), key=lambda c: r[f"loo_rmse_{c}"])
                if r["seed"] in data[c]:
                    chosen[r["seed"]] = data[c][r["seed"]]
        pairs = {"spce_vs_ot_pce": (load(b, "spce"), ot("ot_pce_nrmse")),
                 "gp_m52_vs_ot_gp": (load(b, "gp_m52"), ot("ot_gp_nrmse")),
                 "gp_quad_vs_ot_gp_quad": (load(b, "gp_quad"), ot("ot_gp_quad_nrmse")),
                 "ot_pck_vs_selected": (ot("ot_pck_nrmse"), chosen),
                 "ot_pck_vs_gp_quad": (ot("ot_pck_nrmse"), load(b, "gp_quad"))}
        out[b] = {}
        for k, (a, c) in pairs.items():
            t = paired(a, c)
            if t:
                t["equivalent"] = MARGIN[0] <= t["ci"][0] and t["ci"][1] <= MARGIN[1]
                out[b][k] = t
    return out


# ── T1.4 equivalence of the null results ─────────────────────────────────
def equivalence():
    S = json.load(open(os.path.join(OUT, "summary.json")))
    rows = []
    for t in S["tests"]:
        if t["p_holm"] < 0.05:
            continue
        lo, hi = t["ci"]
        rows.append({k: t[k] for k in ("bench", "a", "b", "metric", "N", "ratio", "ci")}
                    | {"equivalent_20": MARGIN[0] <= lo and hi <= MARGIN[1],
                       "equivalent_10": MARGIN_STRICT[0] <= lo and hi <= MARGIN_STRICT[1]})
    return {"n_null": len(rows), "n_equivalent_20": sum(r["equivalent_20"] for r in rows),
            "n_equivalent_10": sum(r["equivalent_10"] for r in rows), "rows": rows}


# ── T1.5 per-design coverage ─────────────────────────────────────────────
def per_design():
    out = {}
    for b in BENCHES:
        for s in cands(b):
            R = list(cov_records(b, s).values())
            if len(R) < 5:
                continue
            ci = np.array([x["ci_resample_cov"] for x in R])
            pw = np.array([x["pw_jkplus_cov"] for x in R])
            out.setdefault(b, {})[s] = {
                "ci_resample_below_90": float(np.mean(ci < 0.90)), "ci_resample_min": float(ci.min()),
                "ci_resample_q10": float(np.quantile(ci, 0.1)),
                "pw_jkplus_below_90": float(np.mean(pw < 0.90)), "pw_jkplus_min": float(pw.min())}
    allrows = [r for b in out.values() for r in b.values()]
    return {"rows": out,
            "ci_resample_below_90_median": float(np.median([r["ci_resample_below_90"] for r in allrows])),
            "ci_resample_below_90_max": float(max(r["ci_resample_below_90"] for r in allrows)),
            "ci_resample_min": float(min(r["ci_resample_min"] for r in allrows))}


# ── T1.6 bump model ──────────────────────────────────────────────────────
def bump():
    from benchmarks import BENCHMARKS
    from framework_core import lhs
    p = BENCHMARKS["bump"]()
    out = {}
    for N in (20, 40):
        hit = {}
        for sd in range(1000, 1030):
            np.random.seed(sd)
            hit[sd] = bool(np.any(p.in_bump(p.ppf(lhs(p.dim, N)))))
        rs = np.random.RandomState(0)
        miss_mc = np.mean([not np.any(p.in_bump(lhs_draw)) for lhs_draw in
                           (_lhs(rs, p.dim, N) for _ in range(4000))])
        row = {"designs_hitting": int(sum(hit.values())), "p_miss_lhs_mc": float(miss_mc)}
        for s in ("gp_m52", "gp_quad", "spce"):
            R = cov_records("bump", s, N)
            if len(R) < 5:
                continue
            for tag, sel in (("hit", True), ("miss", False)):
                X = [x for sd, x in R.items() if hit[sd] == sel]
                if not X:
                    continue
                row[f"{s}_{tag}"] = {
                    "n": len(X), "nrmse_median": float(np.median([x["nrmse"] for x in X])),
                    "ci_resample_cov": float(np.mean([x["ci_resample_cov"] for x in X])),
                    "pw_jkplus_cov": float(np.mean([x["pw_jkplus_cov"] for x in X])),
                    "dq_q95_jack_count": int(np.sum([x["dq_q95_jack_cov"] for x in X])),
                    "dq_p_thr_jack_count": int(np.sum([x["dq_p_thr_jack_cov"] for x in X]))}
        for a in ("gp_m52", "ak_m52", "al_gp"):
            g = load("bump", a, N=N, metric="thr_abserr")
            if g:
                row[f"{a}_thr_abserr_median"] = float(np.median(list(g.values())))
        out[str(N)] = row
    return out


def _lhs(rs, d, n):
    u = (np.argsort(rs.uniform(size=(n, d)), axis=0) + rs.uniform(size=(n, d))) / n
    return u


# ── worked example (T1.7) ────────────────────────────────────────────────
def worked():
    W = jload("worked_example_cfd.json")
    if not W or "gum_framework" not in W:
        return None
    return {k: W[k] for k in ("M", "adaptive_mc", "gum_framework",
                              "reference_share_in_interval", "reference_share_in_gum_interval")
            if k in W}


# ── Tier 2 ───────────────────────────────────────────────────────────────
def designs():
    out = {}
    for b in BENCHES:
        for D in ("maximin", "sobol"):
            for c in cands(b):
                g = jload(f"results_gen_{b}_{c}{suffix(20, D)}.json")
                if not g:
                    continue
                t = paired({x["seed"]: x["nrmse"] for x in g}, load(b, c))
                if t:
                    t["equivalent"] = MARGIN[0] <= t["ci"][0] and t["ci"][1] <= MARGIN[1]
                    out.setdefault(b, {})[f"{c}_{D}_vs_lhs"] = t
            R = cov_records(b, trend(b), 20, D)
            if R:
                out.setdefault(b, {})[f"ci_resample_cov_{D}"] = float(
                    np.mean([x["ci_resample_cov"] for x in R.values()]))
    return out


def imse():
    tests = []
    for b in BENCHES:
        for N in (20, 40):
            for ref in ("al_gp", "gp_m52"):
                t = paired(load(b, "al_imse", N=N), load(b, ref, N=N))
                if t:
                    tests.append({"bench": b, "N": N, "vs": ref, **t})
    holm(tests)
    return tests


def stacking():
    out = {}
    for b in BENCHES + ["borehole_corr"]:
        R = jload(f"results_stack_{b}.json")
        if not R or len(R) < 5:
            continue
        st = {x["seed"]: x["stack_nrmse"] for x in R}
        se = {x["seed"]: x["selected_nrmse"] for x in R}
        orc = {x["seed"]: x["oracle_nrmse"] for x in R}
        out[b] = {"stack_vs_selected": paired(st, se), "stack_vs_oracle": paired(st, orc)}
    return out


def vector():
    R = jload("results_vector_plume.json")
    if not R:
        return None
    keys = [k for k in R[0] if k.endswith(("_cov", "_vol"))]
    return {"n": len(R), **{k: float(np.mean([x[k] for x in R])) for k in keys},
            "loo_corr_median": float(np.median([np.mean(x["loo_corr"]) for x in R]))}


def correlated():
    b = "borehole_corr"
    out = {"nrmse_median": {}, "coverage": {}}
    for c in cands(b):
        g = load(b, c)
        if g:
            out["nrmse_median"][c] = float(np.median(list(g.values())))
        R = cov_records(b, c)
        if R:
            out["coverage"][c] = {k: float(np.mean([x[k] for x in R.values()]))
                                  for k in ("ci_resample_cov", "ci_plain_cov", "pw_jkplus_cov")}
    sel = jload(f"results_sel_{b}.json")
    if sel and out["nrmse_median"]:
        data = {c: load(b, c) for c in cands(b)}
        r_ = [data[min(cands(b), key=lambda c: r[f"loo_rmse_{c}"])][r["seed"]]
              / min(data[c][r["seed"]] for c in cands(b)) for r in sel
              if all(r["seed"] in data[c] for c in cands(b))]
        out["selection_ratio_to_oracle_median"] = float(np.median(r_))
    return out


def cfd_numerics():
    out = {}
    G = jload("cfd_gci.json")
    if G:
        p = [v for c in G["cases"] for v in c["p"] if v is not None]
        e = [abs(v) for c in G["cases"] for v in c["err_standard"] if v is not None]
        out["gci"] = {"n_monotone": len(p), "n_total": sum(len(c["p"]) for c in G["cases"]),
                      "p_range": [min(p), max(p)] if p else None,
                      "p_median": float(np.median(p)) if p else None,
                      "err_standard_median": float(np.median(e)) if e else None,
                      "err_standard_max": float(max(e)) if e else None}
    R2 = jload("cfd_reference2.json")
    if R2:
        out["reference2"] = R2
    return out


def stopping():
    """Estimated surrogate share 1 - (u_plain / u_with_E)^2 against the
    actual errors, over all models, candidates and budgets."""
    est, nr, uerr = [], [], []
    for f in glob.glob(os.path.join(RES, "results_cov_*.json")):
        R = json.load(open(f))
        if not R or "design" in R[0] or R[0]["benchmark"] not in BENCHES:
            continue
        for x in R:
            if x["ci_resample_u"] > 0:
                est.append(1 - (x["ci_plain_u"] / x["ci_resample_u"]) ** 2)
                nr.append(x["nrmse"]); uerr.append(abs(x["ci_resample_u"] - 1))
    est, nr, uerr = map(np.array, (est, nr, uerr))
    bins = [(-np.inf, 0.01), (0.01, 0.05), (0.05, np.inf)]
    return {"n": int(len(est)), "spearman_share_vs_nrmse": float(spearmanr(est, nr)[0]),
            "bins": [{"share": [float(lo), float(hi)], "n": int(np.sum((est >= lo) & (est < hi))),
                      "nrmse_median": float(np.median(nr[(est >= lo) & (est < hi)])),
                      "u_relerr_median": float(np.median(uerr[(est >= lo) & (est < hi)]))}
                     for lo, hi in bins if np.any((est >= lo) & (est < hi))]}


def cost():
    out = {}
    for b in BENCHES:
        for a in ("gp_m52", "gp_quad", "gp_log", "gp_quadlog", "spce", "spce_log",
                  "pas_ard", "al_gp", "al_var"):
            g = jload(f"results_gen_{b}_{a}.json")
            if g:
                out.setdefault(a, []).extend(x["runtime_s"] for x in g if "runtime_s" in x)
    res = {a: float(np.median(v)) for a, v in out.items() if v}
    grid = jload("cfd_grid_check.json")
    if grid:
        res["cfd_run_s"] = float(np.median([c["refine1"]["runtime_s"] for c in grid["cases"]]))
    return res


def summarise():
    C = {"bayes_gp": bayes_gp(), "budgets": budgets(), "crosscheck": crosscheck(),
         "equivalence": equivalence(), "per_design": per_design(), "bump": bump(),
         "worked": worked(), "designs": designs(), "imse": imse(), "stacking": stacking(),
         "vector": vector(), "correlated": correlated(), "cfd_numerics": cfd_numerics(),
         "stopping": stopping(), "cost": cost()}
    return json.loads(json.dumps(C, default=float))


# ── tables ───────────────────────────────────────────────────────────────
def tab_bayesgp(C):
    """Per model, the GP with trend on the recommended scale: pointwise
    coverage (%) and seeds (of 30) whose interval for y95 contains the truth."""
    lines = []
    for b in BENCHES:
        r = C["bayes_gp"]["rows"].get(b, {}).get(trend(b))
        if not r:
            continue
        pw = [f"{100 * r[f'pw_{v}_cov']:.0f}" if f"pw_{v}_cov" in r else "--"
              for v in ("gp", "gpt", "gpmc", "gpcal", "jkplus")]
        dq = [str(r[f"dq_q95_{v}_cov_count"]) if f"dq_q95_{v}_cov_count" in r else "--"
              for v in ("gp", "gpt", "gpmc", "gpcal", "jack")]
        lines.append(f"{LABEL[b]} & " + " & ".join(pw + dq) + r" \\")
    return "\n".join(lines) + "\n"


def tab_budgets(C):
    """Per model and N: GUM S1 interval coverage (%) of the procedure."""
    lines = []
    R = C["budgets"]["rows"]
    for b in BENCHES:
        cells = []
        for N in ("10", "20", "40", "80"):
            r = R.get(N, {}).get(b)
            cells.append(f"{100 * r['ci_resample_cov']:.1f}" if r else "--")
        if any(c != "--" for c in cells):
            lines.append(f"{LABEL[b]} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    C = summarise()
    path = os.path.join(OUT, "completion.json")
    if "--check" in sys.argv:
        same = _close(C, json.load(open(path)))
        print("completion.json reproduced" if same else "completion.json DIFFERS from the records")
        sys.exit(0 if same else 1)
    os.makedirs(OUT, exist_ok=True)
    json.dump(C, open(path, "w"), indent=1)
    open(os.path.join(OUT, "tab_bayesgp.tex"), "w").write(tab_bayesgp(C))
    open(os.path.join(OUT, "tab_budgets.tex"), "w").write(tab_budgets(C))
    for k in ("bayes_gp", "budgets"):
        print(k, json.dumps(C[k]["decision"], indent=1))
    print("equivalence:", {k: v for k, v in C["equivalence"].items() if k != "rows"})
    print("per-design:", {k: v for k, v in C["per_design"].items() if k != "rows"})
