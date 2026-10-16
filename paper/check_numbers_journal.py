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
spec_sig = [b for b in S["n20"] if b != "cfd" and any(t["bench"] == b and t["a"] == "gp_m52"
            and t["b"] == "gp_orig" and t["N"] == 20 and t["p_holm"] < 0.05 for t in S["tests"])]
check_true("kernel spec significant on six models", len(spec_sig) == 6)
check_true("kernel spec not significant on plume, plume_gum, ishigami",
           not any(sig(b, "gp_m52", "gp_orig") for b in ("plume", "plume_gum", "ishigami")))
check(F, "0.11\n(borehole)", R("borehole", "gp_log", "gp_m52"), "{:.2f}\n(borehole)")
check(F, "to 0.75 (piston)", R("piston", "gp_log", "gp_m52"), "to {:.2f} (piston)")
check_true("log significant on all six positive models",
           all(sig(b, "gp_log", "gp_m52") for b in ("plume", "borehole", "otl", "piston",
                                                    "borehole_gum", "plume_gum")))
check(F, "by a factor of 400", 1 / R("physics", "gp_quad", "gp_m52"),
      lambda v: f"by a factor of {int(round(v, -2))}")
check_true("trend changes nRMSE by < 0.5 % elsewhere",
           all(abs(R(b, "gp_quad", "gp_m52") - 1) < 0.005 for b in
               ("plume", "borehole", "ishigami", "otl", "piston", "borehole_gum", "plume_gum", "flood")))
check(F, "plume models (0.42 to 0.35", med("plume", "gp_log"), "plume models ({:.2f} to "
      + f"{med('plume', 'gp_quadlog'):.2f}")
check(F, "0.37 to 0.28 with GUM-type", med("plume_gum", "gp_log"),
      "{:.2f} to " + f"{med('plume_gum', 'gp_quadlog'):.2f} with GUM-type")
# ── 5.2 PCE and committees ───────────────────────────────────────────────
spce = [R(b, "spce", "gp_m52") for b in ("plume", "borehole", "otl", "piston", "borehole_gum",
                                         "plume_gum", "flood")]
check(F, "1.2 to 3.8 times worse", min(spce), "{:.1f} to " + f"{max(spce):.1f} times worse")
spl = [R(b, "spce_log", "gp_log") for b in ("plume", "borehole", "otl", "piston", "borehole_gum", "plume_gum")]
check(F, "(1.1 to 12 times on the log scale)", min(spl), "({:.1f} to " + f"{max(spl):.0f} times on the log scale)")
cm = {b: R(b, "pas_ard:cv", "gp_quad") for b in S["n20"] if b != "cfd"}
worse = [b for b in cm if sig(b, "pas_ard:cv", "gp_quad") and cm[b] > 1]
check_true("committee never significantly better than trend GP",
           not any(sig(b, "pas_ard:cv", "gp_quad") and cm[b] < 1 for b in cm))
check_true("committee worse on exactly borehole, borehole_gum, otl, flood",
           sorted(worse) == ["borehole", "borehole_gum", "flood", "otl"])
check(F, "It tied on five models and was 1.7 to 2.4 times", min(cm[b] for b in worse),
      "It tied on five models and was {:.1f} to " + f"{max(cm[b] for b in worse):.1f} times")
check(F, "reproduces (ratio 1.00)", R("physics", "pas_ard:cv", "gp_quad"), "reproduces (ratio {:.2f})")
cvu = [b for b in cm if sig(b, "pas_ard:cv", "pas_ard:uni")]
check_true("CV weights better on five models, never worse",
           len(cvu) == 5 and all(R(b, "pas_ard:cv", "pas_ard:uni") < 1 for b in cvu))
# ── 5.3 adaptive and targeted ────────────────────────────────────────────
alg = [R(b, "al_gp", "gp_m52") for b in cm]
check(F, "(ratios 0.86--1.22)", min(alg), "(ratios {:.2f}--" + f"{max(alg):.2f})")
check_true("adaptive GP never significant at N=20", not any(sig(b, "al_gp", "gp_m52") for b in cm))
av = {b: R(b, "al_var:cv", "pas_ard:cv") for b in ("physics", "plume", "borehole", "ishigami", "otl", "piston")}
check(F, "(0.61--0.80)", min(av["physics"], av["borehole"], av["otl"]),
      "({:.2f}--" + f"{max(av['physics'], av['borehole'], av['otl']):.2f})")
check(F, "plume model (1.25)", av["plume"], "plume model ({:.2f})")
check(F, "median of 0.17, against 0.02", S["exceedance"]["plume"]["al_var:cv"]["median_abserr"],
      "median of {:.2f}, against " + f"{S['exceedance']['plume']['pas_ard:cv']['median_abserr']:.2f}")
ue = {b: R(b, "al_ueigv:cv", "al_var:cv") for b in av}
ue_bad = [b for b in ue if sig(b, "al_ueigv:cv", "al_var:cv")]
check_true("UEigV never better, worse on three", len(ue_bad) == 3 and all(ue[b] > 1 for b in ue_bad))
check(F, "three models (1.26--1.72)", min(ue[b] for b in ue_bad),
      "three models ({:.2f}--" + f"{max(ue[b] for b in ue_bad):.2f})")
ak = [t for t in S["tests"] if t["N"] == 20 and t["a"] in ("ak_m52", "ak_log") and t["bench"] != "cfd"]
aks = [t for t in ak if t["p_holm"] < 0.05]
check(F, "8 of\nthe 15 comparisons", len(aks), lambda v: f"{v} of\nthe {len(ak)} comparisons")
check_true("AK never significantly worse", all(t["ratio"] < 1 for t in aks))
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
check(F, "off by 6.9--31\\% on\nsix of the nine", cq[3],
      lambda v: f"off by {100 * v:.1f}--{100 * cq[-1]:.0f}\\% on\nsix of the nine")
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
pos = [sel[b]["ratio_to_gp_m52"] for b in ("plume", "borehole", "otl", "piston", "borehole_gum", "plume_gum")]
check(F, "factors of 0.12--0.84", min(pos), "factors of {:.2f}--" + f"{max(pos):.2f}")
# ── 5.6 budget ───────────────────────────────────────────────────────────
pos4 = ("plume", "borehole", "otl", "piston")
r10 = [R(b, "gp_log", "gp_m52", N=10) for b in pos4]; r80 = [R(b, "gp_log", "gp_m52", N=80) for b in pos4]
check(F, "0.62--0.92 at $N=10$", min(r10), "{:.2f}--" + f"{max(r10):.2f} at $N=10$")
check(F, "0.10--0.48 at $N=80$", min(r80), "{:.2f}--" + f"{max(r80):.2f} at $N=80$")
check(F, "$1.6\\times10^{-4}$, against 0.97", S["sweep"]["physics"]["spce"]["10"]["median"],
      lambda v: f"${v / 1e-4:.1f}\\times10^{{-4}}$, against {S['sweep']['physics']['gp_m52']['10']['median']:.2f}")
check_true("adaptive GP never significant at N=10",
           not any(sig(b, "al_gp", "gp_m52", N=10) for b in ("physics", "plume", "borehole", "ishigami", "otl", "piston")))
check(F, "physics\n(0.22--0.43)", R("physics", "al_gp", "gp_m52", N=80),
      "physics\n({:.2f}--" + f"{R('physics', 'al_gp', 'gp_m52', N=40):.2f})")
check(F, "borehole (0.77)", R("borehole", "al_gp", "gp_m52", N=40), "borehole ({:.2f})")
check(F, "OTL (0.75 at $N=80$)", R("otl", "al_gp", "gp_m52", N=80), "OTL ({:.2f} at $N=80$)")
check(F, "piston\n(0.81--0.87)", R("piston", "al_gp", "gp_m52", N=40),
      "piston\n({:.2f}--" + f"{R('piston', 'al_gp', 'gp_m52', N=80):.2f})")
check(F, "plume model (1.19 at $N=40$)", R("plume", "al_gp", "gp_m52", N=40), "plume model ({:.2f} at $N=40$)")
check(F, "(0.94 and 0.50)", R("physics", "al_var:cv", "pas_ard:cv", N=40),
      "({:.2f} and " + f"{R('borehole', 'al_var:cv', 'pas_ard:cv', N=40):.2f})")
for b, N in (("physics", 40), ("plume", 40), ("borehole", 40), ("borehole", 80), ("otl", 80),
             ("piston", 40), ("piston", 80)):
    check_true(f"al_gp vs gp_m52 significant on {b} N={N}", sig(b, "al_gp", "gp_m52", N=N))
# ── coverage ─────────────────────────────────────────────────────────────
F = "journal_results_coverage.tex"
check(F, "between 95.2\\%\nand 99.0\\%", cr["pw_jkplus_cov"]["min"],
      lambda v: f"between {pct(v)}\\%\nand {pct(cr['pw_jkplus_cov']['max'])}\\%")
check(F, "(median 97.4\\%)", cr["pw_jkplus_cov"]["median"], lambda v: f"(median {pct(v)}\\%)")
check(F, "covered 89.7--97.0\\% (median 94.2\\%)", cr["pw_cvnorm_cov"]["min"],
      lambda v: f"covered {pct(v)}--{pct(cr['pw_cvnorm_cov']['max'])}\\% (median {pct(cr['pw_cvnorm_cov']['median'])}\\%)")
check(F, "covered 68--98\\% (median 89\\%)", cr["pw_gp_cov"]["min"],
      lambda v: f"covered {100 * v:.0f}--{100 * cr['pw_gp_cov']['max']:.0f}\\% (median {100 * cr['pw_gp_cov']['median']:.0f}\\%)")
check(F, "ranged from 57\\% to 96\\% (median 94\\%)", cr["ci_plain_cov"]["min"],
      lambda v: f"ranged from {100 * v:.0f}\\% to {100 * cr['ci_plain_cov']['max']:.0f}\\% (median {100 * cr['ci_plain_cov']['median']:.0f}\\%)")
check(F, "to 91.5--97.3\\%\n(median 95.0\\%)", cr["ci_normal_cov"]["min"],
      lambda v: f"to {pct(v)}--{pct(cr['ci_normal_cov']['max'])}\\%\n(median {pct(cr['ci_normal_cov']['median'])}\\%)")
check(F, "to 91.0--97.0\\% (median 94.9\\%)", cr["ci_resample_cov"]["min"],
      lambda v: f"to {pct(v)}--{pct(cr['ci_resample_cov']['max'])}\\% (median {pct(cr['ci_resample_cov']['median'])}\\%)")
check(F, "gave 89.8--96.0\\% (median 94.5\\%)", cr["ci_gp_cov"]["min"],
      lambda v: f"gave {pct(v)}--{pct(cr['ci_gp_cov']['max'])}\\% (median {pct(cr['ci_gp_cov']['median'])}\\%)")
cv = S["coverage"]
check(F, "rose from 57\\% to 95\\%", cv["ishigami"]["gp_orig"]["ci_plain_cov"],
      lambda v: f"rose from {100 * v:.0f}\\% to {100 * cv['ishigami']['gp_orig']['ci_normal_cov']:.0f}\\%")
check(F, "a median 53\\% of seeds (7--100\\%)", cr["dq_q95_gp_cov"]["median"],
      lambda v: f"a median {100 * v:.0f}\\% of seeds ({100 * cr['dq_q95_gp_cov']['min']:.0f}--{100 * cr['dq_q95_gp_cov']['max']:.0f}\\%)")
check(F, "for $P(Y>T)$ in 57\\%", cr["dq_p_thr_gp_cov"]["median"], lambda v: f"for $P(Y>T)$ in {100 * v:.0f}\\%")
check(F, "contained the true value in a median\n97\\% of seeds", cr["dq_q95_jack_cov"]["median"],
      lambda v: f"contained the true value in a median\n{100 * v:.0f}\\% of seeds")
check_true("jackknife p_thr median equals q95 median (both quantities)",
           round(cr["dq_p_thr_jack_cov"]["median"], 2) == round(cr["dq_q95_jack_cov"]["median"], 2))
print("all numbers reproduced" if ok else "MISMATCHES FOUND")
sys.exit(0 if ok else 1)
