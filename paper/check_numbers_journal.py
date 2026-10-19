"""
check_numbers_journal.py — recompute every number quoted in the text of the
journal manuscript (paper/journal_*.tex) from results/journal/summary.json
and the result records, and check that the quoted string is in the text.
Tables are generated, so only numbers typed into the prose are listed here.

Usage: python paper/check_numbers_journal.py      (exit status 1 on any mismatch)
Run python experiments/analyse_journal.py first if the records changed.
"""

import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
S = json.load(open(os.path.join(ROOT, "results", "journal", "summary.json")))
TEXT = {f: open(os.path.join(PAPER, f)).read() for f in os.listdir(PAPER)
        if f.startswith("journal_") and f.endswith(".tex")}


def T(b, a, c, metric="nrmse", N=20):
    return next(t for t in S["tests"] if (t["bench"], t["a"], t["b"], t["metric"], t["N"])
                == (b, a, c, metric, N))


def R(b, a, c, metric="nrmse", N=20):
    return T(b, a, c, metric, N)["ratio"]


def sig(b, a, c, metric="nrmse", N=20):
    return T(b, a, c, metric, N)["p_holm"] < 0.05


med = lambda b, m: S["n20"][b][m]["median"]


def load_pair(b, a, c):
    """Per-seed ratios of nRMSE, arm a over arm c, N = 20 (from the records)."""
    ra = {r["seed"]: r["nrmse"] for r in json.load(open(os.path.join(ROOT, "results", f"results_gen_{b}_{a}.json")))}
    rc = {r["seed"]: r["nrmse"] for r in json.load(open(os.path.join(ROOT, "results", f"results_gen_{b}_{c}.json")))}
    return [ra[k] / rc[k] for k in sorted(ra) if k in rc]
MODELS = [b for b in S["n20"] if S["n20"][b]]            # models with records
prop = lambda b, m, k: S["prop"][b][m][k]
cr = S["coverage_ranges"]
ok = True


def check(fname, quoted, value, fmt="{:.2f}"):
    """quoted: the number exactly as typed in the text; value: recomputed."""
    global ok
    got = fmt.format(value) if isinstance(fmt, str) else fmt(value)
    in_text = quoted in TEXT[fname]
    good = (got == quoted) and in_text
    ok &= good
    if not good:
        print(f"[FAIL] {fname}: quoted '{quoted}', recomputed '{got}'"
              + ("" if in_text else " (quoted string not found in text)"))
    return good


def check_true(desc, cond):
    global ok
    ok &= bool(cond)
    if not cond:
        print(f"[FAIL] {desc}")


pct = lambda x: f"{100 * x:.1f}"
F = "journal_results_main.tex"
# ── 5.1 specification ────────────────────────────────────────────────────
check(F, "factors of 0.14 (physics)", R("physics", "gp_m52", "gp_orig"), "factors of {:.2f} (physics)")
check(F, "to 0.71 (piston)", R("piston", "gp_m52", "gp_orig"), "to {:.2f} (piston)")
check(F, "median nRMSE from 0.59 to 0.033, eighteen-fold", med("physics", "gp_orig"),
      lambda v: f"median nRMSE from {v:.2f} to {med('physics', 'gp_amp'):.3f}, "
                + {18: "eighteen"}.get(round(v / med("physics", "gp_amp")), "?") + "-fold")
spec_sig = [b for b in MODELS if b != "cfd" and any(t["bench"] == b and t["a"] == "gp_m52"
            and t["b"] == "gp_orig" and t["N"] == 20 and t["p_holm"] < 0.05 for t in S["tests"])]
check_true("kernel spec significant on eight models", len(spec_sig) == 8)
check_true("kernel spec not significant on plume, plume_gum, ishigami",
           not any(sig(b, "gp_m52", "gp_orig") for b in ("plume", "plume_gum", "ishigami")))
check(F, "0.11\n(borehole)", R("borehole", "gp_log", "gp_m52"), "{:.2f}\n(borehole)")
check(F, "to 0.75 (piston)", R("piston", "gp_log", "gp_m52"), "to {:.2f} (piston)")
check_true("log significant on all eight positive analytical models",
           all(sig(b, "gp_log", "gp_m52") for b in ("plume", "borehole", "otl", "piston", "wingweight", "otl20", "borehole_gum", "plume_gum")))
check_true("log-scale factors span the quoted range",
           all(0.105 <= R(b, "gp_log", "gp_m52") <= 0.755 for b in ("plume", "borehole", "otl", "piston", "wingweight", "otl20", "borehole_gum", "plume_gum")))
check(F, "by a factor of 400", 1 / R("physics", "gp_quad", "gp_m52"),
      lambda v: f"by a factor of {int(round(v, -2))}")
check_true("trend changes nRMSE by < 0.5 % on nine of the other ten models",
           all(abs(R(b, "gp_quad", "gp_m52") - 1) < 0.005 for b in
               ("plume", "borehole", "ishigami", "otl", "piston", "wingweight", "borehole_gum",
                "plume_gum", "flood")) and abs(R("otl20", "gp_quad", "gp_m52") - 1) >= 0.005)
_q = load_pair("otl20", "gp_quad", "gp_m52")
check(F, "1.6 to 8 times worse in\n12 of the 30 designs", min(v for v in _q if v > 1.5),
      lambda v: f"{v:.1f} to {max(_q):.0f} times worse in\n{sum(x > 1.5 for x in _q)} of the 30 designs")
check(F, "and the wing weight model (ratio 0.50)", R("wingweight", "gp_quadlog", "gp_log"),
      "and the wing weight model (ratio {:.2f})")
check(F, "20-input model worse (1.18)", R("otl20", "gp_quadlog", "gp_log"), "20-input model worse ({:.2f})")
check_true("trend on log scale: wing weight better, otl20 worse, both significant",
           sig("wingweight", "gp_quadlog", "gp_log") and sig("otl20", "gp_quadlog", "gp_log"))
check(F, "plume models (0.42 to 0.35", med("plume", "gp_log"), "plume models ({:.2f} to "
      + f"{med('plume', 'gp_quadlog'):.2f}")
check(F, "0.37 to 0.28 with GUM-type", med("plume_gum", "gp_log"),
      "{:.2f} to " + f"{med('plume_gum', 'gp_quadlog'):.2f} with GUM-type")
# ── 5.2 PCE and committees ───────────────────────────────────────────────
spce = [R(b, "spce", "gp_m52") for b in ("plume", "borehole", "otl", "piston", "wingweight",
                                         "otl20", "borehole_gum", "plume_gum", "flood")]
check(F, "1.2 to 3.8 times worse", min(spce), "{:.1f} to " + f"{max(spce):.1f} times worse")
spl = [R(b, "spce_log", "gp_log") for b in ("plume", "borehole", "otl", "piston", "otl20",
                                            "borehole_gum", "plume_gum")]
check(F, "it was 1.1 to 12 times worse", min(spl), "it was {:.1f} to " + f"{max(spl):.0f} times worse")
check(F, "log-scale GP (ratio 0.72)", R("wingweight", "spce_log", "gp_log"), "log-scale GP (ratio {:.2f})")
check_true("log PCE better than log GP on wing weight", sig("wingweight", "spce_log", "gp_log"))
cm = {b: R(b, "pas_ard:cv", "gp_quad") for b in MODELS if b != "cfd"}
worse = [b for b in cm if sig(b, "pas_ard:cv", "gp_quad") and cm[b] > 1]
check_true("committee never significantly better than trend GP",
           not any(sig(b, "pas_ard:cv", "gp_quad") and cm[b] < 1 for b in cm))
check_true("committee worse on exactly borehole, borehole_gum, otl, wingweight, flood",
           sorted(worse) == ["borehole", "borehole_gum", "flood", "otl", "wingweight"])
check(F, "It tied on six models and was 1.5 to 2.4 times", min(cm[b] for b in worse),
      "It tied on six models and was {:.1f} to " + f"{max(cm[b] for b in worse):.1f} times")
check(F, "reproduces (ratio 1.00)", R("physics", "pas_ard:cv", "gp_quad"), "reproduces (ratio {:.2f})")
cvu = [b for b in cm if sig(b, "pas_ard:cv", "pas_ard:uni")]
check_true("CV weights better on seven models, never worse",
           len(cvu) == 7 and all(R(b, "pas_ard:cv", "pas_ard:uni") < 1 for b in cvu))
# ── 5.3 adaptive and targeted ────────────────────────────────────────────
alg = [R(b, "al_gp", "gp_m52") for b in cm]
check(F, "(ratios 0.86--1.22)", min(alg), "(ratios {:.2f}--" + f"{max(alg):.2f})")
check_true("adaptive GP never significant at N=20", not any(sig(b, "al_gp", "gp_m52") for b in cm))
av = {b: R(b, "al_var:cv", "pas_ard:cv") for b in ("physics", "plume", "borehole", "ishigami", "otl",
                                                   "piston", "wingweight", "otl20")}
avb = [b for b in av if sig(b, "al_var:cv", "pas_ard:cv") and av[b] < 1]
check_true("adaptive committee better on physics, borehole, otl, wingweight, otl20",
           sorted(avb) == ["borehole", "otl", "otl20", "physics", "wingweight"])
check(F, "(0.60--0.80)", min(av[b] for b in avb), "({:.2f}--" + f"{max(av[b] for b in avb):.2f})")
check(F, "plume model (1.25)", av["plume"], "plume model ({:.2f})")
check(F, "median of 0.17, against 0.02", S["exceedance"]["plume"]["al_var:cv"]["median_abserr"],
      "median of {:.2f}, against " + f"{S['exceedance']['plume']['pas_ard:cv']['median_abserr']:.2f}")
ue = {b: R(b, "al_ueigv:cv", "al_var:cv") for b in ("physics", "plume", "borehole", "ishigami", "otl", "piston")}
ue_bad = [b for b in ue if sig(b, "al_ueigv:cv", "al_var:cv")]
check_true("UEigV never better, worse on three", len(ue_bad) == 3 and all(ue[b] > 1 for b in ue_bad))
check(F, "three models (1.26--1.72)", min(ue[b] for b in ue_bad),
      "three models ({:.2f}--" + f"{max(ue[b] for b in ue_bad):.2f})")
ak = [t for t in S["tests"] if t["N"] == 20 and t["a"] in ("ak_m52", "ak_log") and t["bench"] != "cfd"]
aks = [t for t in ak if t["p_holm"] < 0.05]
check(F, "8 of\nthe 19 comparisons", len(aks), lambda v: f"{v} of\nthe {len(ak)} comparisons")
check_true("AK never significantly worse", all(t["ratio"] < 1 for t in aks))
check(F, "median factors of 0.68 (plume) and 0.55", R("plume", "akq_log", "gp_log", "exc_abserr"),
      "median factors of {:.2f} (plume) and " + f"{R('plume_gum', 'akq_log', 'gp_log', 'exc_abserr'):.2f}")
check(F, "no difference (0.95)", R("cfd", "akq_log", "gp_log", "exc_abserr"), "no difference ({:.2f})")
check_true("AK at the reference quantile never significant",
           not any(sig(b, a, c, "exc_abserr") for b, a, c in
                   (("plume", "akq_log", "gp_log"), ("plume_gum", "akq_log", "gp_log"),
                    ("cfd", "akq_log", "gp_log"), ("plume", "akq_m52", "gp_m52"),
                    ("plume_gum", "akq_m52", "gp_m52"))))
check(F, "factors of 0.11 to 0.47", min(t["ratio"] for t in aks),
      "factors of {:.2f} to " + f"{max(t['ratio'] for t in aks):.2f}")
check(F, "piston model (0.0099 to 0.0014", S["exceedance"]["piston"]["gp_log"]["median_abserr"],
      "piston model ({:.4f} to " + f"{S['exceedance']['piston']['ak_log']['median_abserr']:.4f}")
check(F, "borehole model (0.0091 to 0.0009", S["exceedance"]["borehole"]["gp_m52"]["median_abserr"],
      "borehole model ({:.4f} to " + f"{S['exceedance']['borehole']['ak_m52']['median_abserr']:.4f}")
check(F, "from 8.5\\% to 28\\%", prop("ishigami", "gp_m52", "relerr_mean"),
      lambda v: f"from {100 * v:.1f}\\% to {100 * prop('ishigami', 'ak_m52', 'relerr_mean'):.0f}\\%")
# ── 5.4 propagated quantities ────────────────────────────────────────────
rec = lambda b: "gp_log" if b in ("plume", "borehole", "otl", "piston", "borehole_gum", "plume_gum") else "gp_m52"
q95 = {b: prop(b, rec(b), "relerr_q95") for b in cm}
check(F, "within 0.2\\% (OTL, flood) to 16\\%", min(q95.values()),
      lambda v: f"within {100 * v:.1f}\\% (OTL, flood) to {100 * max(q95.values()):.0f}\\%")
thr = {b: prop(b, rec(b), "thr_abserr") for b in cm}
check(F, "within 0.0013 to 0.036", min(thr.values()), "within {:.4f} to " + f"{max(thr.values()):.3f}")
cq = sorted(prop(b, "pas_ard:cv", "relerr_q95") for b in cm)
check(F, "off by 6.3--31\\% on\nseven of the eleven", cq[4],
      lambda v: f"off by {100 * v:.1f}--{100 * cq[-1]:.0f}\\% on\nseven of the eleven")
check_true("committee q95 error below 6 % on four models", cq[3] < 0.06 <= cq[4])
check(F, "$P(C>210\\,\\mathrm{mg\\,m^{-3}})=0.123$", S["exceedance"]["plume"]["gp_log"]["p_true"],
      "$P(C>210\\,\\mathrm{{mg\\,m^{{-3}}}})={:.3f}$")
check(F, "median absolute error of 0.009, against 0.023",
      S["exceedance"]["plume"]["gp_log"]["median_abserr"],
      "median absolute error of {:.3f}, against " + f"{S['exceedance']['plume']['gp_m52']['median_abserr']:.3f}")
# ── 5.5 selection ────────────────────────────────────────────────────────
sel = {b: v for b, v in S["selection"].items() if b != "cfd"}
check(F, "between 1.000 and 1.044", min(v["ratio_to_oracle"] for v in sel.values()),
      "between {:.3f} and " + f"{max(v['ratio_to_oracle'] for v in sel.values()):.3f}")
check(F, "27--77\\% of seeds", min(v["share_oracle"] for v in sel.values()),
      lambda v: f"{100 * v:.0f}--{100 * max(x['share_oracle'] for x in sel.values()):.0f}\\% of seeds")
check(F, "factor of 500 on the physics", 1 / sel["physics"]["ratio_to_gp_m52"],
      lambda v: f"factor of {int(round(v, -2))} on the physics")
pos = [sel[b]["ratio_to_gp_m52"] for b in ("plume", "borehole", "otl", "piston", "wingweight", "otl20", "borehole_gum", "plume_gum")]
check_true("selection never chose a trend GP on otl20",
           not any(c in sel["otl20"]["chosen"] for c in ("gp_quad", "gp_quadlog")))
check(F, "log-scale sparse PCE in 9 of 30 seeds", sel["wingweight"]["chosen"].get("spce_log", 0),
      "log-scale sparse PCE in {} of 30 seeds")
check(F, "factors of 0.12--0.84", min(pos), "factors of {:.2f}--" + f"{max(pos):.2f}")
# ── 5.6 budget ───────────────────────────────────────────────────────────
pos4 = ("plume", "borehole", "otl", "piston", "wingweight", "otl20")
r10 = [R(b, "gp_log", "gp_m52", N=10) for b in pos4]; r80 = [R(b, "gp_log", "gp_m52", N=80) for b in pos4]
check(F, "0.62--1.00 at $N=10$", min(r10), "{:.2f}--" + f"{max(r10):.2f} at $N=10$")
check(F, "0.10--0.48 at $N=80$", min(r80), "{:.2f}--" + f"{max(r80):.2f} at $N=80$")
check(F, "$1.6\\times10^{-4}$, against 0.97", S["sweep"]["physics"]["spce"]["10"]["median"],
      lambda v: f"${v / 1e-4:.1f}\\times10^{{-4}}$, against {S['sweep']['physics']['gp_m52']['10']['median']:.2f}")
check_true("adaptive GP never significant at N=10",
           not any(sig(b, "al_gp", "gp_m52", N=10) for b in ("physics", "plume", "borehole", "ishigami",
                                                            "otl", "piston", "wingweight", "otl20")))
check(F, "wing weight (0.81 at $N=40$)", R("wingweight", "al_gp", "gp_m52", N=40), "wing weight ({:.2f} at $N=40$)")
check(F, "20-input OTL (0.83 at $N=80$)", R("otl20", "al_gp", "gp_m52", N=80), "20-input OTL ({:.2f} at $N=80$)")
check(F, "wing weight model at\n$N=80$ (1.93)", R("wingweight", "al_gp", "gp_m52", N=80),
      "wing weight model at\n$N=80$ ({:.2f})")
check(F, "physics\n(0.22--0.43)", R("physics", "al_gp", "gp_m52", N=80),
      "physics\n({:.2f}--" + f"{R('physics', 'al_gp', 'gp_m52', N=40):.2f})")
check(F, "borehole (0.77)", R("borehole", "al_gp", "gp_m52", N=40), "borehole ({:.2f})")
check(F, "OTL (0.75 at $N=80$)", R("otl", "al_gp", "gp_m52", N=80), "OTL ({:.2f} at $N=80$)")
check(F, "piston (0.81--0.87)", R("piston", "al_gp", "gp_m52", N=40),
      "piston ({:.2f}--" + f"{R('piston', 'al_gp', 'gp_m52', N=80):.2f})")
check(F, "plume model\n(1.19 at $N=40$)", R("plume", "al_gp", "gp_m52", N=40), "plume model\n({:.2f} at $N=40$)")
check(F, "(0.94 and 0.50)", R("physics", "al_var:cv", "pas_ard:cv", N=40),
      "({:.2f} and " + f"{R('borehole', 'al_var:cv', 'pas_ard:cv', N=40):.2f})")
for b, N in (("physics", 40), ("plume", 40), ("borehole", 40), ("borehole", 80), ("otl", 80),
             ("piston", 40), ("piston", 80), ("wingweight", 40), ("wingweight", 80), ("otl20", 80)):
    check_true(f"al_gp vs gp_m52 significant on {b} N={N}", sig(b, "al_gp", "gp_m52", N=N))
# ── coverage ─────────────────────────────────────────────────────────────
F = "journal_results_coverage.tex"
check(F, "between 95.2\\%\nand 99.6\\%", cr["pw_jkplus_cov"]["min"],
      lambda v: f"between {pct(v)}\\%\nand {pct(cr['pw_jkplus_cov']['max'])}\\%")
check(F, "(median 97.4\\%)", cr["pw_jkplus_cov"]["median"], lambda v: f"(median {pct(v)}\\%)")
check(F, "covered 89.7--97.1\\% (median 94.2\\%)", cr["pw_cvnorm_cov"]["min"],
      lambda v: f"covered {pct(v)}--{pct(cr['pw_cvnorm_cov']['max'])}\\% (median {pct(cr['pw_cvnorm_cov']['median'])}\\%)")
check(F, "covered 68--98\\% (median 88\\%)", cr["pw_gp_cov"]["min"],
      lambda v: f"covered {100 * v:.0f}--{100 * cr['pw_gp_cov']['max']:.0f}\\% (median {100 * cr['pw_gp_cov']['median']:.0f}\\%)")
check(F, "ranged from 57\\% to 96\\% (median 94\\%)", cr["ci_plain_cov"]["min"],
      lambda v: f"ranged from {100 * v:.0f}\\% to {100 * cr['ci_plain_cov']['max']:.0f}\\% (median {100 * cr['ci_plain_cov']['median']:.0f}\\%)")
check(F, "to 91.5--97.3\\%\n(median 94.9\\%)", cr["ci_normal_cov"]["min"],
      lambda v: f"to {pct(v)}--{pct(cr['ci_normal_cov']['max'])}\\%\n(median {pct(cr['ci_normal_cov']['median'])}\\%)")
check(F, "to 91.0--97.0\\% (median 94.9\\%)", cr["ci_resample_cov"]["min"],
      lambda v: f"to {pct(v)}--{pct(cr['ci_resample_cov']['max'])}\\% (median {pct(cr['ci_resample_cov']['median'])}\\%)")
check(F, "gave 89.8--96.0\\% (median 94.6\\%)", cr["ci_gp_cov"]["min"],
      lambda v: f"gave {pct(v)}--{pct(cr['ci_gp_cov']['max'])}\\% (median {pct(cr['ci_gp_cov']['median'])}\\%)")
_C = {b: v for b, v in S["coverage"].items() if b != "cfd"}
_rows = [r for v in _C.values() for r in v.values()]
_gp = [r for v in _C.values() for s_, r in v.items() if s_.startswith("gp_")]
check(F, "(68 combinations of model and surrogate on\nthe eleven analytical models", len(_rows),
      "({} combinations of model and surrogate on\nthe eleven analytical models")
check(F, "in 31 of the 49 GP combinations",
      sum(r["pw_gp_score"] <= min(r["pw_cvnorm_score"], r["pw_jkplus_score"]) for r in _gp),
      lambda v: f"in {v} of the {len(_gp)} GP combinations")
_w = [r["pw_jkplus_width"] / r["pw_cvnorm_width"] for r in _rows]
check(F, "a median 1.6 (1.1--2.9)", float(np.median(_w)),
      lambda v: f"a median {v:.1f} ({min(_w):.1f}--{max(_w):.1f})")
check(F, "a median 0.95 times the reference (down to\n0.41)", float(np.median([r["ci_plain_u"] for r in _rows])),
      lambda v: f"a median {v:.2f} times the reference (down to\n{min(r['ci_plain_u'] for r in _rows):.2f})")
check(F, "a median 1.00 (0.78--1.29)", float(np.median([r["ci_resample_u"] for r in _rows])),
      lambda v: f"a median {v:.2f} ({min(r['ci_resample_u'] for r in _rows):.2f}--"
                f"{max(r['ci_resample_u'] for r in _rows):.2f})")
check_true("GP pointwise coverage below 80 % on exactly the six models named",
           sorted({b for b, v in _C.items() for s_, r in v.items() if s_.startswith("gp_")
                   and r["pw_gp_cov"] < 0.8}) == ["ishigami", "otl20", "piston", "plume", "plume_gum",
                                                  "wingweight"])
cv = S["coverage"]
check(F, "rose from 57\\% to 95\\%", cv["ishigami"]["gp_orig"]["ci_plain_cov"],
      lambda v: f"rose from {100 * v:.0f}\\% to {100 * cv['ishigami']['gp_orig']['ci_normal_cov']:.0f}\\%")
check(F, "a median 63\\% of seeds (7--100\\%)", cr["dq_q95_gp_cov"]["median"],
      lambda v: f"a median {100 * v:.0f}\\% of seeds ({100 * cr['dq_q95_gp_cov']['min']:.0f}--{100 * cr['dq_q95_gp_cov']['max']:.0f}\\%)")
check(F, "for $P(Y>T)$ in 63\\%", cr["dq_p_thr_gp_cov"]["median"], lambda v: f"for $P(Y>T)$ in {100 * v:.0f}\\%")
check(F, "contained the true value in a median\n97\\% of seeds", cr["dq_q95_jack_cov"]["median"],
      lambda v: f"contained the true value in a median\n{100 * v:.0f}\\% of seeds")
check_true("jackknife p_thr median equals q95 median (both quantities)",
           round(cr["dq_p_thr_jack_cov"]["median"], 2) == round(cr["dq_q95_jack_cov"]["median"], 2))
# ── CFD case ─────────────────────────────────────────────────────────────
F = "journal_results_cfd.tex"
ref = json.load(open(os.path.join(ROOT, "results", "cfd_reference.json")))
grid = json.load(open(os.path.join(ROOT, "results", "cfd_grid_check.json")))
itc = json.load(open(os.path.join(ROOT, "results", "cfd_iteration_check.json")))
check(F, "Of the 2900 CFD runs", ref["n_runs_cached"], lambda v: f"Of the {v} CFD runs")
check(F, "adaptive designs), 80\\%", ref["share_converged"], lambda v: f"adaptive designs), {100 * v:.0f}\\%")
check(F, "(median 1322\niterations)", ref["iterations_median"], lambda v: f"(median {v:.0f}\niterations)")
check(F, "less than\n0.16\\%", max(x for c in itc["cases"] for x in c["rel_range_2000_4000"]),
      lambda v: f"less than\n{np.ceil(100 * v * 100) / 100:.2f}\\%")
gd = np.abs([x for c in grid["cases"] for x in c["rel_diff"]])
check(F, "0.9--6.5\\% (median 3.7\\%)", gd.min(),
      lambda v: f"{100 * v:.1f}--{100 * gd.max():.1f}\\% (median {100 * np.median(gd):.1f}\\%)")
r_, u_ = ref["reference"], ref["u_reference"]
check(F, "a mean of 434\\,mg", r_["mean"], lambda v: f"a mean of {v:.0f}\\,mg")
check(F, "an SD of 413\\,mg", r_["sd"], lambda v: f"an SD of {v:.0f}\\,mg")
check(F, "of 1252\\,mg", r_["q95"], lambda v: f"of {v:.0f}\\,mg")
check(F, "$P(C>\\mathrm{IDLH})=0.673$", r_["p_idlh"], "$P(C>\\mathrm{{IDLH}})={:.3f}$")
check(F, "2.3\\%,\n3.4\\%, 3.3\\% and 0.007", u_["mean"] / r_["mean"],
      lambda v: f"{100 * v:.1f}\\%,\n{100 * u_['sd'] / r_['sd']:.1f}\\%, "
                f"{100 * u_['q95'] / r_['q95']:.1f}\\% and {u_['p_idlh']:.3f}")
check(F, "(ratio 0.96)", R("cfd", "gp_m52", "gp_orig"), "(ratio {:.2f})")
check(F, "median nRMSE of 0.080 against 0.53", med("cfd", "gp_log"),
      "median nRMSE of {:.3f} against " + f"{med('cfd', 'gp_m52'):.2f}")
check(F, "(ratio 0.17, better\nin all 30 seeds)", R("cfd", "gp_log", "gp_m52"),
      lambda v: f"(ratio {v:.2f}, better\nin all {T('cfd', 'gp_log', 'gp_m52')['wins']} seeds)")
check(F, "to 0.060 (ratio 0.76)", med("cfd", "gp_quadlog"),
      "to {:.3f} (ratio " + f"{R('cfd', 'gp_quadlog', 'gp_log'):.2f})")
check(F, "1.4 times worse than the GP on the raw", R("cfd", "spce", "gp_m52"),
      "{:.1f} times worse than the GP on the raw")
check(F, "scale and 2.3 times on the log scale", R("cfd", "spce_log", "gp_log"),
      "scale and {:.1f} times on the log scale")
check(F, "GP with trend (ratio 1.08)", R("cfd", "pas_ard:cv", "gp_quad"), "GP with trend (ratio {:.2f})")
sc = S["selection"]["cfd"]
check_true("CFD selection chose a log-scale GP in all 30 seeds",
           sum(v for k, v in sc["chosen"].items() if k in ("gp_log", "gp_quadlog")) == 30)
check(F, "within 0.1\\% of the best", sc["ratio_to_oracle_ci"][1] - 1,
      lambda v: "within 0.1\\% of the best" if v < 0.001 else f"within {100 * v:.1f}\\% of the best")
check(F, "within 6.9\\% of the reference", prop("cfd", "gp_log", "relerr_q95"),
      lambda v: f"within {100 * v:.1f}\\% of the reference")
check(F, "within 0.009, against\n18.5\\% and 0.019", prop("cfd", "gp_log", "thr_abserr"),
      lambda v: f"within {v:.3f}, against\n{100 * prop('cfd', 'gp_m52', 'relerr_q95'):.1f}\\% and "
                f"{prop('cfd', 'gp_m52', 'thr_abserr'):.3f}")
check(F, "27\\% and 0.059 for the committee", prop("cfd", "pas_ard:cv", "relerr_q95"),
      lambda v: f"{100 * v:.0f}\\% and {prop('cfd', 'pas_ard:cv', 'thr_abserr'):.3f} for the committee")
check(F, "change the nRMSE (ratio 0.93)", R("cfd", "al_gp", "gp_m52"), "change the nRMSE (ratio {:.2f})")
check(F, "(ratio 0.94)", R("cfd", "ak_log", "gp_log", "thr_abserr"), "(ratio {:.2f})")
check_true("CFD: AK and GP-SD not significant", not sig("cfd", "ak_log", "gp_log", "thr_abserr")
           and not sig("cfd", "al_gp", "gp_m52"))
check(F, "the passive one (ratio 0.84)", R("cfd", "al_var:cv", "pas_ard:cv"), "the passive one (ratio {:.2f})")
check_true("CFD adaptive committee significant", sig("cfd", "al_var:cv", "pas_ard:cv"))
ex = S["exceedance"]["cfd"]
check(F, "median of\n0.11, against 0.06", ex["al_var:cv"]["median_abserr"],
      lambda v: f"median of\n{v:.2f}, against {ex['pas_ard:cv']['median_abserr']:.2f}")
cc = S["coverage"]["cfd"]
gpk = [k for k in cc if k.startswith("gp_")]
check(F, "covered 69--92\\% of\ntest points", min(cc[k]["pw_gp_cov"] for k in gpk),
      lambda v: f"covered {100 * v:.0f}--{100 * max(cc[k]['pw_gp_cov'] for k in gpk):.0f}\\% of\ntest points")
check(F, "intervals 94.8--99.1\\%", min(cc[k]["pw_jkplus_cov"] for k in cc),
      lambda v: f"intervals {100 * v:.1f}--{100 * max(cc[k]['pw_jkplus_cov'] for k in cc):.1f}\\%")
check(F, "of 90.8--94.7\\%; adding the LOO error gave 93.8--95.7\\%",
      min(cc[k]["ci_plain_cov"] for k in cc),
      lambda v: f"of {100 * v:.1f}--{100 * max(cc[k]['ci_plain_cov'] for k in cc):.1f}\\%; adding the LOO error gave "
                f"{100 * min(cc[k]['ci_normal_cov'] for k in cc):.1f}--{100 * max(cc[k]['ci_normal_cov'] for k in cc):.1f}\\%")
check(F, "in only 10\\% of seeds, against 87--93\\%", cc["gp_m52"]["dq_q95_gp_cov"],
      lambda v: f"in only {100 * v:.0f}\\% of seeds, against {100 * cc['gp_m52']['dq_q95_jack_cov']:.0f}--"
                f"{100 * cc['gp_quad']['dq_q95_jack_cov']:.0f}\\%")
check_true("CFD raw GPs both 10 %", round(cc["gp_quad"]["dq_q95_gp_cov"], 2) == 0.10)
check(F, "were 80\\% and 97\\%", cc["gp_log"]["dq_q95_gp_cov"],
      lambda v: f"were {100 * v:.0f}\\% and {100 * cc['gp_log']['dq_q95_jack_cov']:.0f}\\%")
# ── worked example (results/worked_example_cfd.json) ─────────────────────
F = "journal_discussion.tex"
W = json.load(open(os.path.join(ROOT, "results", "worked_example_cfd.json")))
wl, wp, wj, wr, wu = (W["loo_rmse"], W["with_surrogate_contribution"], W["jackknife_95"],
                      W["reference"], W["u_reference"])
check_true("worked example chose gp_quadlog", W["chosen"] == "gp_quadlog")
check(F, "LOO RMSE (23\\,mg\\,m$^{-3}$)", wl["gp_quadlog"], "LOO RMSE ({:.0f}\\,mg\\,m$^{{-3}}$)")
check(F, "log-scale GP (31), the log-scale sparse PCE (93) and the raw-scale\ncandidates (157--260)",
      wl["gp_log"], lambda v: f"log-scale GP ({v:.0f}), the log-scale sparse PCE ({wl['spce_log']:.0f}) "
      f"and the raw-scale\ncandidates ({min(wl[k] for k in ('gp_m52', 'gp_quad', 'spce')):.0f}--"
      f"{max(wl[k] for k in ('gp_m52', 'gp_quad', 'spce')):.0f})")
W0 = W["surrogate_as_exact"]
wchg = {k: wp[k] / W0[k] - 1 for k in wp}
check(F, "changed the reported quantities by at most 1.2\\%", max(abs(v) for v in wchg.values()),
      lambda v: f"changed the reported quantities by at most {100 * v:.1f}\\%")
check_true("worked example: residuals lowered every quantity", all(v < 0 for v in wchg.values()))
rl = np.array(W["loo_residuals_latent"])
check(F, "negative mean ($-0.013$", rl.mean(), "negative mean (${:.3f}$")
check_true("worked example: one residual dominates the mean", abs(rl.min()) > abs(rl.sum()) / 2)
check(F, "contains\n94.3\\% of the reference", W["reference_share_in_interval"],
      lambda v: f"contains\n{100 * v:.1f}\\% of the reference")
inside = {k: wj[k][0] <= wr[k] <= wj[k][1] for k in wj}
check_true("worked example: jackknife contains reference for mean and P only",
           {k for k, v in inside.items() if v} == {"mean", "p_exceed", "q025"})
gap = {k: wr[k] / wj[k][1] - 1 for k in ("sd", "q95", "q975")}
check(F, "the reference lies\n0.9--1.5\\% above the upper end", min(gap.values()),
      lambda v: f"the reference lies\n{100 * v:.1f}--{100 * max(gap.values()):.1f}\\% above the upper end")
check_true("worked example: gaps below half the reference uncertainty",
           all(wr[k] - wj[k][1] < 0.5 * wu[k] for k in gap))
bs = W["budget"]; sh = dict(zip(W["input_names"], bs["input_share"]))
check(F, "wind speed contributed 36\\%", sh["U_ref"], lambda v: f"wind speed contributed {100 * v:.0f}\\%")
check(F, "rate 31\\%, the receptor distance 5.8\\%, the turbulent Schmidt number 3.9\\%",
      sh["Q"], lambda v: f"rate {100 * v:.0f}\\%, the receptor distance {100 * sh['x_r']:.1f}\\%, "
                         f"the turbulent Schmidt number {100 * sh['Sc_t']:.1f}\\%")
check(F, "building height 3.1\\% and the roughness length 1.0\\%", sh["H_b"],
      lambda v: f"building height {100 * v:.1f}\\% and the roughness length {100 * sh['z0']:.1f}\\%")
check(F, "inputs 19\\% and the surrogate 0.3\\%", bs["interaction_share"],
      lambda v: f"inputs {100 * v:.0f}\\% and the surrogate {100 * bs['surrogate_share']:.1f}\\%")
# ── discussion: adaptive committee on four of seven models ───────────────
avb = [b for b in S["n20"] if "al_var:cv" in S["n20"][b]]
check_true("adaptive committee better on six of nine models (incl. CFD)",
           len(avb) == 9 and sum(sig(b, "al_var:cv", "pas_ard:cv") and R(b, "al_var:cv", "pas_ard:cv") < 1
                                 for b in avb) == 6)
# ── abstract ─────────────────────────────────────────────────────────────
F = "journal_abstract.tex"
logs = [R(b, "gp_log", "gp_m52") for b in ("plume", "borehole", "otl", "piston", "borehole_gum", "plume_gum", "cfd")]
check(F, "to 18, 400 and 9", med("physics", "gp_orig") / med("physics", "gp_amp"),
      lambda v: f"to {v:.0f}, {int(round(1 / R('physics', 'gp_quad', 'gp_m52'), -2))} and {1 / min(logs):.0f}")
check(F, "up\nto ninefold", 1 / min(t["ratio"] for t in aks),
      lambda v: "up\nto ninefold" if round(v) == 9 else f"up\nto {v:.0f}-fold")
allc = {k: [r[k] for row in S["coverage"].values() for r in row.values() if k in r]
        for k in ("ci_normal_cov", "pw_jkplus_cov", "pw_gp_cov")}
check(F, "to 91--97\\% on all models", min(allc["ci_normal_cov"]),
      lambda v: f"to {100 * v:.0f}--{100 * max(allc['ci_normal_cov']):.0f}\\% on all models")
check(F, "covered 95--100\\% of test points", min(allc["pw_jkplus_cov"]),
      lambda v: f"covered {100 * v:.0f}--{100 * max(allc['pw_jkplus_cov']):.0f}\\% of test points")
check(F, "as few as 68\\%", min(allc["pw_gp_cov"]), lambda v: f"as few as {100 * v:.0f}\\%")
check(F, "median of\nonly 63\\%", cr["dq_q95_gp_cov"]["median"], lambda v: f"median of\nonly {100 * v:.0f}\\%")
print("all numbers reproduced" if ok else "MISMATCHES FOUND")
sys.exit(0 if ok else 1)
