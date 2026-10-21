"""
check_numbers.py — recompute every number quoted in extended_abstract.tex
from the raw result records. See NUMBERS.md for the mapping.

Usage:  python paper/imeko2027/check_numbers.py        (from anywhere)
"""

import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from analyse_generalisation import summarise, latex_propagation_table   # noqa: E402

RES = os.path.join(ROOT, "results")
S = summarise()
M, T = S["medians"], S["tests"]


def head(t):
    print(f"\n== {t} " + "=" * max(0, 60 - len(t)))


head("Seeds per strategy")
print({b: sorted(set(S["n"][b].values())) for b in S["n"]})

head("Plume model facts (Section II.A)")
from benchmarks import Plume                                          # noqa: E402
p = Plume(); dom = np.array(p.bounds)
X = dom[:, 0] + np.random.RandomState(8).uniform(0, 1, (200_000, p.dim)) * (dom[:, 1] - dom[:, 0])
c = p.evaluate(X)
print(f"range {c.min():.3g}-{c.max():.4g} mg/m3 (orders: {np.log10(c.max()/c.min()):.1f}); "
      f"P(C > 210 mg/m3) = {np.mean(c > 210):.1%}")

head("Table II: single-GP kernel specification (median nRMSE)")
for arm in ["gp_orig", "gp_amp", "gp_ardnoamp", "gp_ard", "gp_m52", "gp_log"]:
    print(f"{arm:12s} " + "  ".join(
        f"{b}={M[b][arm]['nrmse']:.3g}" if arm in M[b] else f"{b}=--" for b in M))

print(f"number of paired tests (Holm family): {S['n_tests']}")
for b in M:
    for k in ["gp_amp vs gp_orig", "gp_log vs gp_m52"]:
        if k in T[b]:
            print(f"  {b} {k}: holm p = {T[b][k]['p_holm']:.2g}")

head("Section III.A statements")
t = T["physics"]["gp_amp vs gp_orig"]
print(f"physics amplitude alone: 0.591 -> 0.033, wins {t['wins']}/30, p={t['p']:.1g}, "
      f"per-seed median ratio {t['median_ratio']:.3f} (= {1/t['median_ratio']:.0f}-fold)")
print(f"eighteen-fold (medians): {M['physics']['gp_orig']['nrmse']/M['physics']['gp_amp']['nrmse']:.1f}")
for b in ["plume", "borehole"]:
    t = T[b]["gp_log vs gp_m52"]
    print(f"{b} log vs m52: {M[b]['gp_m52']['nrmse']:.3f} -> {M[b]['gp_log']['nrmse']:.3f} "
          f"(medians ratio {M[b]['gp_m52']['nrmse']/M[b]['gp_log']['nrmse']:.1f}), "
          f"wins {t['wins']}/30, p={t['p']:.2g}")
dom_rows = []
for cfg in ["phys10", "phys50", "phys100", "phys200"]:
    d = json.load(open(os.path.join(RES, f"results_domain_{cfg}.json"))); sd = d[0]["output_std"]
    dom_rows.append(f"{cfg}: original {np.median([r['rmse_lhsgp'] for r in d])/sd:.3f}, "
                    f"Matern {np.median([r['rmse_lhsgp_m52'] for r in d])/sd:.3f}")
print("domain sweep (results_domain_*.json):\n  " + "\n  ".join(dom_rows))

head("Section III.B statements (Fig. 1)")
for b in M:
    for k in ["pas_ard:cv vs gp_m52", "pas_ard:cv vs pas_ard:uni", "al_gp vs gp_m52",
              "al_var:cv vs pas_ard:cv", "al_ueigv:cv vs al_var:cv"]:
        if k in T[b]:
            t = T[b][k]
            print(f"{b:9s} {k:26s} ratio {t['median_ratio']:9.4g} wins {t['wins']:2d}/{t['n']} "
                  f"p={t['p']:.2g} holm={t['p_holm']:.2g}")
print("physics committee vs best single GP (gp_amp): "
      f"{M['physics']['gp_amp']['nrmse']/M['physics']['pas_ard:cv']['nrmse']:.0f}x")

head("Table III (median over 30 seeds)")
print(latex_propagation_table(S))

head("Section III.C statements")
for b in M:
    cov = {m: M[b][m]["exc_covered"] for m in ["pas_ard:cv", "al_var:cv", "al_ueigv:cv"] if m in M[b]}
    print(f"{b:9s} envelope coverage of P: " + ", ".join(f"{m} {v:.0%}" for m, v in cov.items()))

head("Fig. 3 / Section III.C plume example (results/plume_cdf_seed1000.json)")
r = json.load(open(os.path.join(RES, "plume_cdf_seed1000.json")))
for k, v in r.items():
    if isinstance(v, dict) and "P_exceed_idlh" in v:
        print(f"{k:28s} P(C > 210) = {v['P_exceed_idlh']:.3f}")

head("Section III.D / Table IV: validating the surrogate contribution (results_cov_*.json)")
for b, a in [("physics", "gp_m52"), ("plume", "gp_log"), ("borehole", "gp_log"),
             ("ishigami", "gp_m52")]:
    d = json.load(open(os.path.join(RES, f"results_cov_{b}_{a}.json")))
    m = lambda k: 100 * np.mean([r[k] for r in d])
    print(f"{b:9s} {a:7s} n={len(d)} pointwise GP {m('pw_gp_cov'):.0f} J+ {m('pw_jkplus_cov'):.0f} | "
          f"GUM S1 exact {m('ci_plain_cov'):.0f} +LOO {m('ci_normal_cov'):.0f} | "
          f"y95 GP {m('dq_q95_gp_cov'):.0f} J {m('dq_q95_jack_cov'):.0f}")

head("Section III.E: CFD dispersion case (results_gen_cfd_*.json, cfd_reference.json)")
g = {a: json.load(open(os.path.join(RES, f"results_gen_cfd_{a}.json")))
     for a in ("gp_m52", "gp_log", "pas_ard")}
med = lambda a, k: float(np.median([r[k] for r in g[a]]))
sd = {r["seed"]: r["nrmse"] for r in g["gp_m52"]}
print(f"nRMSE: GP log {med('gp_log', 'nrmse'):.3f}, GP Matern {med('gp_m52', 'nrmse'):.2f}; "
      f"log better in {sum(r['nrmse'] < sd[r['seed']] for r in g['gp_log'])}/30 seeds; "
      f"median ratio {np.median([r['nrmse'] / sd[r['seed']] for r in g['gp_log']]):.3f} "
      f"(1/ratio = {1 / np.median([r['nrmse'] / sd[r['seed']] for r in g['gp_log']]):.1f}-fold)")
print(f"y95 rel. error: GP log {100 * med('gp_log', 'relerr_q95'):.1f} %, "
      f"GP Matern {100 * med('gp_m52', 'relerr_q95'):.1f} %, "
      f"committee {100 * float(np.median([r['cv_relerr_q95'] for r in g['pas_ard']])):.0f} %")
print(f"P(C > IDLH) abs. error: GP log {med('gp_log', 'thr_abserr'):.3f}")
ref = json.load(open(os.path.join(RES, "cfd_reference.json")))
print(f"reference standard uncertainty: y95 {100 * ref['u_reference']['q95'] / ref['reference']['q95']:.1f} %, "
      f"P(C > IDLH) {ref['u_reference']['p_idlh']:.3f}; runs {ref['n_runs_cached']}")
