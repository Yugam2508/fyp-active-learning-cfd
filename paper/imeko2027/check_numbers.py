"""
check_numbers.py — recompute every number quoted in extended_abstract.tex
from the raw result records. See NUMBERS.md for the mapping.

Usage:  python paper/imeko2027/check_numbers.py        (from anywhere)
"""

import os, json
import numpy as np
from collections import Counter
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
THRESH = 200.0
load = lambda name: json.load(open(os.path.join(RES, name)))


def head(t):
    print(f"\n== {t} " + "=" * max(0, 60 - len(t)))


# ── Domain sweep ──────────────────────────────────────────────────────────
head("Domain sweep (Fig. 1)")
for c in ["phys2", "phys10", "phys50", "phys100", "phys200", "schwefel"]:
    d = load(f"results_domain_{c}.json")
    al = np.array([t["rmse_al"] for t in d]); gp = np.array([t["rmse_lhsgp"] for t in d])
    sd = d[0]["output_std"]
    print(f"{c:9s} n={len(d)} SD={sd:9.4g} | GP med/SD={np.median(gp)/sd:.3f} "
          f"| committee med={np.median(al):.3f} med/SD={np.median(al)/sd:.3g} "
          f"| rho(mean)={al.mean()/gp.mean():.3f} | GP med={np.median(gp):.4g}")
d10 = load("results_domain_phys10.json")
print("h=10: PCE-d2 sole survivor in",
      sum(t["final_models"] == ["PCE-d2"] for t in d10), "/ 30")
for h in [2, 10, 50, 100, 200]:
    X = np.random.RandomState(7).uniform(-h, h, (200_000, 3))
    q = 0.2 * (X ** 2).sum(1)
    r = np.sin(2 * X[:, 0]) * np.cos(2 * X[:, 1]) + np.exp(-X[:, 2] ** 2)
    print(f"h={h:3d}: quadratic variance share={q.var()/(q + r).var():.3f}, "
          f"RMS of non-quadratic terms={r.std():.3f}")

# ── Ablations ─────────────────────────────────────────────────────────────
head("Ablations (Table I)")
arms = {k: {t["seed"]: t["rmse"] for t in load(f"results_{k}.json")}
        for k in ["hard", "varacq", "nopce", "gpvar", "soft"]}
for k, v in arms.items():
    r = np.array(list(v.values()))
    line = f"{k:7s} median={np.median(r):8.4g} success={(r < THRESH).sum()}/{len(r)}"
    if k != "hard":
        s = sorted(set(v) & set(arms["hard"]))
        a = np.log([v[i] for i in s]); b = np.log([arms["hard"][i] for i in s])
        line += f" | p vs hard={wilcoxon(a, b).pvalue:.3g} | lower in {(a < b).sum()}/{len(s)}"
    print(line)

head("Decomposition (collapse of hard rejection)")
dec = load("results_decompose.json")
coll = [t for t in dec if t["n_survivors"] == 1]
print(f"collapse to one member: {len(coll)}/{len(dec)}")
print("median collapse iteration:",
      np.median([t["collapse_iter"] for t in dec if t["collapse_iter"] is not None]))
rq = sum(t["random_queries_after_collapse"] for t in dec)
print(f"random queries: {rq}/{len(dec) * 15} = {rq / (len(dec) * 15):.1%}")
print("sole survivor:", Counter(t["survivor_names"][0] for t in coll))
a = np.log([t["rmse_A_hard_reported"] for t in dec])
b = np.log([t["rmse_B_hard_traj_soft_predictor"] for t in dec])
print(f"predictor swap: p={wilcoxon(a, b).pvalue:.3g}, "
      f"within 1%: {(np.abs(np.exp(a) - np.exp(b)) / np.exp(a) < 0.01).sum()}/{len(dec)}")

# ── Aggregation ───────────────────────────────────────────────────────────
head("Aggregation (Table II)")
hm = load("results_hardmin3.json")
u = np.array([t["rmse_uniform"] for t in hm]); w = np.array([t["rmse_cvweighted"] for t in hm])
has = np.array(["PCE-d2" in t["survivor_names"] for t in hm])
print(f"K_min=3 uniform:     median={np.median(u):.4g} success={(u < THRESH).sum()}/{len(u)}")
print(f"K_min=3 CV-weighted: median={np.median(w):.4g} success={(w < THRESH).sum()}/{len(w)}")
print(f"PCE-d2 survives in {has.sum()}/{len(hm)}; weighted success when present: "
      f"{(w[has] < THRESH).sum()}/{has.sum()}, when absent: {(w[~has] < THRESH).sum()}/{(~has).sum()}")
print(f"paired p (uniform vs weighted) = {wilcoxon(np.log(u), np.log(w)).pvalue:.2g}")

# ── Propagation ───────────────────────────────────────────────────────────
head("Propagation (Table III)")
for k in ["soft", "hard"]:
    d = [t for t in load(f"results_{k}.json") if t["rmse"] < THRESH]
    q = {qq: np.median([t["q_relerr"][qq] for t in d]) for qq in ["0.05", "0.5", "0.95"]}
    print(f"{k}: n={len(d)} q05={q['0.05']:.2%} q50={q['0.5']:.2%} q95={q['0.95']:.2%} "
          f"|P-0.05|={np.median([abs(t['exc_central'] - t['exc_true']) for t in d]):.4f} "
          f"width={np.median([t['exc_up'] - t['exc_lo'] for t in d]):.3f} "
          f"cov mean={np.mean([t['mean_covered'] for t in d]):.0%} "
          f"cov exc={np.mean([t['exc_covered'] for t in d]):.0%} "
          f"pbox={np.mean([t['pbox_coverage'] for t in d]):.0%}")

head("Representative run (Fig. 3)")
r = load("representative_run_seed1000.json")
print(f"mean {r['mean']['rel_err']:.2%}, SD {r['std']['rel_err']:.2%}, quantiles "
      + ", ".join(f"{k}: {v['rel_err']:.2%}" for k, v in r["quantiles"].items()))
tr = r["trajectory"]
print(f"model-form share {tr[0]['model_form_fraction']:.2f} (N={tr[0]['n_runs']}) -> "
      f"{tr[-1]['model_form_fraction']:.1g} (N={tr[-1]['n_runs']})")
print(f"RMSE^2/Var[Y] = {r['rmse'] ** 2 / r['std']['true'] ** 2:.1g}")
print(f"exceedance interval [{r['exceedance']['lower']:.3f}, {r['exceedance']['upper']:.3f}], "
      f"reference {r['exceedance']['true']:.3f}")
