"""
demo_uq.py — end-to-end demonstration on the physics benchmark.

Produces:
  fig_uq_pdf.pdf        output PDF with epistemic envelope vs ground truth
  fig_uq_pbox.pdf       CDF probability box + bounded exceedance probability
  fig_uq_decomp.pdf     variance decomposition / epistemic fraction vs N_t
  fig_uq_sobol.pdf      Sobol indices with epistemic error bars vs truth
  fig_uq_weights.pdf    soft-LCS weight trajectories (recovery behaviour)
  comparison printout   hard vs soft LCS RMSE (3 trials each, smoke test)
"""

import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _os.path.join(_ROOT, "src"))
RESULTS_DIR = _os.path.join(_ROOT, "results"); _os.makedirs(RESULTS_DIR, exist_ok=True)
FIGURES_DIR = _os.path.join(_ROOT, "figures"); _os.makedirs(FIGURES_DIR, exist_ok=True)

import time
import numpy as np
import matplotlib.pyplot as plt

from framework_core import (ToyProblem, ActiveLearner, lhs)
from uq_extension import (SoftActiveLearner, UQAnalyzer,
                          uniform_sampler, make_bound_penalty)

np.random.seed(42)
try:
    import torch; torch.manual_seed(42)
except Exception:
    pass

plt.style.use("seaborn-v0_8-whitegrid")
FIGDIR = FIGURES_DIR
import os; os.makedirs(FIGDIR, exist_ok=True)

problem = ToyProblem(dim=3, name="physics")
domain  = problem.bounds
sampler = uniform_sampler(domain)
N_INIT, MAX_STEPS = 5, 15

# Fixed probe set for the convergence diagnostic (same every iteration)
X_probe = sampler(4000)
y_probe_true = problem.evaluate(X_probe)

# ═══════════════════════════════════════════════════════════════════════
# 1. Train soft-weighted active learner, tracking UQ convergence
# ═══════════════════════════════════════════════════════════════════════
print("Training SoftActiveLearner on physics [-200,200]^3 ...")
t0 = time.time()
agent = SoftActiveLearner(
    ActiveLearner, problem, domain, n_init=N_INIT,
    n_folds=3, update_every=3,
    physics_penalty=make_bound_penalty(lower=-1.0, scale=10.0),  # f >= -1 analytically
)
agent.train()

traj = []   # (N_t, rmse, V_input, V_epi, eps_fraction)
def snapshot():
    uq = UQAnalyzer(agent.models, weights=agent.w, noise_std=0.1)
    out = uq.propagate(lambda n: X_probe, n_mc=len(X_probe))  # reuse fixed probe
    d = out["decomposition"]
    rmse = float(np.sqrt(np.mean((y_probe_true - out["mu"])**2)))
    traj.append((len(agent.y), rmse, d["V_input"], d["V_epistemic"],
                 d["epistemic_fraction"]))
    return rmse

snapshot()
for t in range(MAX_STEPS):
    agent.step()
    agent.train(force_weight_update=(t == MAX_STEPS - 1))
    rmse = snapshot()
    print(f"  step {t+1:2d} | N={len(agent.y):2d} | RMSE={rmse:10.3f} | "
          f"eps-frac={traj[-1][4]:.3f}")
print(f"AL run: {time.time()-t0:.0f}s, final RMSE={traj[-1][1]:.3f}")

# ═══════════════════════════════════════════════════════════════════════
# 2. The UQ deliverables (what the framework was built to produce)
# ═══════════════════════════════════════════════════════════════════════
uq  = UQAnalyzer(agent.models, weights=agent.w, noise_std=0.1)
out = uq.propagate(sampler, n_mc=50_000)
P, mu = out["preds"], out["mu"]

# Ground truth MC (only possible because benchmark is synthetic)
X_ref = sampler(200_000)
y_ref = problem.evaluate(X_ref)

print("\n── UQ summary (50k MC on surrogate vs 200k MC on truth) ──")
print(f" mean : {out['stats']['mean']:10.2f}   "
      f"epistemic band {out['stats']['mean_epistemic_band']}   "
      f"truth {y_ref.mean():10.2f}")
print(f" std  : {out['stats']['std']:10.2f}   truth {y_ref.std():10.2f}")
for q in (0.05, 0.5, 0.95):
    print(f" q{int(q*100):02d}  : {out['stats']['quantiles'][q]:10.2f}   "
          f"truth {np.quantile(y_ref, q):10.2f}")
d = out["decomposition"]
print(f" variance decomposition: input-driven {d['V_input']:.3e} | "
      f"epistemic {d['V_epistemic']:.3e} | fraction {d['epistemic_fraction']:.4f}")

# --- Fig A: PDF with epistemic envelope ---------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
bins = np.linspace(np.quantile(y_ref, 0.001), np.quantile(y_ref, 0.999), 80)
hists = np.array([np.histogram(p, bins=bins, density=True)[0] for p in P])
ctr = 0.5*(bins[1:]+bins[:-1])
h_central = np.average(hists, axis=0, weights=uq.w)
ax.hist(y_ref, bins=bins, density=True, alpha=0.35, color="grey",
        label="Ground truth (200k MC on $f$)")
ax.fill_between(ctr, hists.min(axis=0), hists.max(axis=0),
                color="tab:blue", alpha=0.25,
                label="Epistemic envelope (committee)")
ax.plot(ctr, h_central, color="tab:blue", lw=2,
        label="Weighted ensemble PDF")
ax.set_xlabel("Output $Y = f(X)$"); ax.set_ylabel("Density")
ax.set_title("Output PDF of the black box — surrogate UQ vs truth\n"
             "(physics $[-200,200]^3$, 20 evaluations of $f$)")
ax.legend(); fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig_uq_pdf.pdf"); fig.savefig(f"{FIGDIR}/fig_uq_pdf.png", dpi=140)

# --- Fig B: p-box CDF + exceedance --------------------------------------
pb = uq.pbox(P)
y_crit = float(np.quantile(y_ref, 0.95))          # a 'safety threshold'
exc = uq.exceedance(P, y_crit)
exc_true = float(np.mean(y_ref > y_crit))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.fill_between(pb["y"], pb["F_lower"], pb["F_upper"], color="tab:orange",
                alpha=0.3, label="CDF probability box (committee)")
ax.plot(pb["y"], pb["F_central"], color="tab:orange", lw=2,
        label="Weighted mixture CDF")
ys = np.sort(y_ref); ax.plot(ys, np.linspace(0, 1, len(ys)), "k--", lw=1.2,
        label="Ground-truth CDF")
ax.axvline(y_crit, color="red", ls=":", lw=1.5)
ax.annotate(f"$P(Y>y_c)$ = {exc['central']:.3f}\n"
            f"bounds [{exc['lower']:.3f}, {exc['upper']:.3f}]\n"
            f"truth {exc_true:.3f}",
            xy=(y_crit, 0.5), xytext=(y_crit*0.55, 0.35), fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="red", alpha=0.9))
ax.set_xlabel("Output level $y$"); ax.set_ylabel("$F_Y(y)$")
ax.set_title("Probability box and bounded exceedance probability")
ax.legend(loc="lower right"); fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig_uq_pbox.pdf"); fig.savefig(f"{FIGDIR}/fig_uq_pbox.png", dpi=140)

# --- Fig C: variance decomposition trajectory ---------------------------
tr = np.array(traj)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.semilogy(tr[:, 0], tr[:, 2], "o-", label="$V_{input}$ (input-driven)")
ax1.semilogy(tr[:, 0], tr[:, 3], "s-", label="$V_{epistemic}$ (model-form)")
ax1.set_xlabel("Training samples $N_t$"); ax1.set_ylabel("Variance")
ax1.set_title("Law-of-total-variance decomposition"); ax1.legend()
ax2.plot(tr[:, 0], tr[:, 4], "d-", color="tab:red")
ax2.set_xlabel("Training samples $N_t$")
ax2.set_ylabel("Epistemic fraction $V_{epi}/V_{tot}$")
ax2.set_title("UQ convergence diagnostic\n(stop sampling when this is small)")
fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig_uq_decomp.pdf"); fig.savefig(f"{FIGDIR}/fig_uq_decomp.png", dpi=140)

# --- Fig D: Sobol indices with epistemic bars ----------------------------
sob = uq.sobol(sampler, dim=3, n=4096)

def sobol_truth(n=200_000):       # same estimator on the true function
    A, B = sampler(n), sampler(n)
    fA, fB = problem.evaluate(A), problem.evaluate(B)
    V = np.var(np.concatenate([fA, fB]))
    S, ST = [], []
    for i in range(3):
        ABi = A.copy(); ABi[:, i] = B[:, i]
        fABi = problem.evaluate(ABi)
        S.append(np.mean(fB*(fABi-fA))/V); ST.append(0.5*np.mean((fA-fABi)**2)/V)
    return np.array(S), np.array(ST)

S_true, ST_true = sobol_truth()
x = np.arange(3); wdt = 0.35
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.bar(x - wdt/2, sob["S"], wdt, yerr=[sob["S"]-sob["S_band"][0],
       sob["S_band"][1]-sob["S"]], capsize=4, label="$S_i$ (ensemble)",
       color="tab:blue", alpha=0.85)
ax.bar(x + wdt/2, sob["ST"], wdt, yerr=[sob["ST"]-sob["ST_band"][0],
       sob["ST_band"][1]-sob["ST"]], capsize=4, label="$S_T$ (ensemble)",
       color="tab:green", alpha=0.85)
ax.plot(x - wdt/2, S_true, "k_", ms=22, mew=2.5, label="truth")
ax.plot(x + wdt/2, ST_true, "k_", ms=22, mew=2.5)
ax.set_xticks(x); ax.set_xticklabels([f"$x_{i+1}$" for i in range(3)])
ax.set_ylabel("Sobol index")
ax.set_title("Global sensitivity with epistemic error bars\n"
             "(error bars = committee spread; black ticks = ground truth)")
ax.legend(); fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig_uq_sobol.pdf"); fig.savefig(f"{FIGDIR}/fig_uq_sobol.png", dpi=140)

# --- Fig E: soft weight trajectories -------------------------------------
W = np.array([w for _, w in agent.weight_history])
Ns = [n for n, _ in agent.weight_history]
names = [m.name for m in agent.models]
fig, ax = plt.subplots(figsize=(8, 4.4))
for j, nm in enumerate(names):
    ax.plot(Ns, W[:, j], marker="o", ms=3, label=nm)
ax.axhline(0.5/len(names), color="grey", ls=":", lw=1,
           label="UEigV mute floor")
ax.set_xlabel("Training samples $N_t$"); ax.set_ylabel("Soft weight $w_k$")
ax.set_title("SoftLCS weight trajectories — muting is transient and reversible")
ax.legend(fontsize=7, ncol=2); fig.tight_layout()
fig.savefig(f"{FIGDIR}/fig_uq_weights.pdf"); fig.savefig(f"{FIGDIR}/fig_uq_weights.png", dpi=140)

# ═══════════════════════════════════════════════════════════════════════
# 3. Hard vs soft LCS — smoke comparison (3 trials each)
# ═══════════════════════════════════════════════════════════════════════
print("\n── Hard vs Soft LCS smoke comparison (3 trials, full 30 to be run "
      "by you on the cluster) ──")
X_test = sampler(5000); y_test = problem.evaluate(X_test)

def run_hard():
    a = ActiveLearner(ToyProblem(3, "physics"), domain); a.train()
    for _ in range(MAX_STEPS):
        a.step(); a.train()
    preds = [m.predict(X_test) for m in a.models if m not in a.rejected]
    return float(np.sqrt(np.mean((y_test - np.mean(preds, axis=0))**2)))

def run_soft():
    a = SoftActiveLearner(ActiveLearner, ToyProblem(3, "physics"), domain,
                          n_folds=3, update_every=3,
                          physics_penalty=make_bound_penalty(lower=-1.0))
    a.train()
    for i in range(MAX_STEPS):
        a.step(); a.train(force_weight_update=(i == MAX_STEPS - 1))
    return float(np.sqrt(np.mean((y_test - a.predict(X_test))**2)))

hard = [run_hard() for _ in range(3)]
soft = [run_soft() for _ in range(3)]
print(f" Hard LCS RMSE: {['%.2f' % r for r in hard]}  (mean {np.mean(hard):.2f})")
print(f" Soft LCS RMSE: {['%.2f' % r for r in soft]}  (mean {np.mean(soft):.2f})")
print("\nAll figures saved to ./figures/")
