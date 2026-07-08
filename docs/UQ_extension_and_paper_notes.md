# Completing the UQ Pipeline: Design Notes, Literature Map, and Paper Plan

Companion to `uq_extension.py` and `demo_uq.py`. Drop the two classes
(`SoftLCS`/`SoftActiveLearner` and `UQAnalyzer`) into the notebook after the
`ActiveLearner` cell; they depend only on the `Surrogate.fit/.predict` API.

---

## 1. What was missing and what was added

The thesis pipeline ends at "accurate surrogate at N=20". The project's stated
purpose — *quantify the uncertainty of an expensive black box* — needs one more
stage: use the trained committee as a Monte Carlo engine and report the UQ
quantities a decision-maker actually consumes, each with honest epistemic
error bars derived from committee disagreement. `UQAnalyzer` delivers:

**Output distribution.** Propagate the input distribution through all K
surrogates at once (a 50,000-point MC costs seconds on the surrogate versus
~50,000 CFD-hours on the solver). The weighted mixture gives the central PDF;
the per-model envelope gives the epistemic band on the PDF itself.

**Probability box on the CDF.** At every output level y, the committee's CDFs
span [F_lower(y), F_upper(y)]. This is a p-box in the sense of Ferson &
Ginzburg (1996): the *width* of the box is irreducible-by-this-data model-form
uncertainty. Exceedance probabilities P(Y > y_crit) — the safety-engineering
quantity — come out *bounded*, not just point-estimated. For an ammonia
release, "P(concentration > IDLH) ∈ [0.02, 0.05]" is a defensible statement;
a bare 0.03 from a single GP is not.

**Variance decomposition.** By the law of total variance over the input
distribution:

    Var[Y] = Var_x[ μ_w(x) ]  +  E_x[ σ²_epi(x) ]  ( + σ²_ε )
             input-driven         model-form (epistemic)

The **epistemic fraction** V_epi/V_total is a UQ convergence diagnostic: while
it dominates, the next dollar should buy another CFD run; once it is small,
the UQ estimate is committee-converged and further simulations refine the
*function*, not the *statistics*. This also directly answers the thesis's
open problem of online stopping/failure detection (Sec. 7.5.2) in UQ terms.

**Global sensitivity.** Sobol first-order and total indices via the
Saltelli (2010) / Jansen (1999) pick-and-freeze estimators, evaluated on the
cheap ensemble (cost n(2+d) surrogate calls), with per-model indices as
epistemic error bars on the sensitivity *ranking*. This closes the loop with
Hanna et al. (1993), who identified wind speed and emission rate as dominant
by expensive direct MC — the framework now produces the same ranking from 20
runs, with uncertainty on the ranking itself.

---

## 2. Soft weighting instead of hard rejection — and what we learned

Your proposed extension (weight models rather than kick them) was implemented
and stress-tested. The final design and the *negative results on the way to
it* are both publishable content.

### Final design (`SoftLCS`)

    w_k ∝ 1 / ( CV-MSE_k · (1 + physics_penalty_k) )

k-fold cross-validation error, inverse-error weights — the **weighted average
surrogate** of Goel et al. (2007) and Viana et al. (2009), which is
inverse-variance weighting, the minimum-variance rule for combining
estimators. Models below a weight floor are *transiently muted* from the
UEigV graph (so pathological extrapolators cannot inject phantom
disagreement) but keep retraining and recover automatically — fixing hard
LCS's irreversibility. A `physics_penalty` hook (see `make_bound_penalty`)
soft-penalises violations of known physics (e.g. concentration ≥ 0),
in the spirit of physics-informed ML (Karniadakis et al., 2021) and
constrained GPs (Swiler et al., 2020), without hard-constraining any model.

### Negative results worth reporting (paper ablation / defence Q&A)

1. **Boltzmann weights on the training-point chi-square score fail**
   (w ∝ exp(−s_k/2T): physics-benchmark RMSE ≈ 2000 vs ≈ 0.7 for hard LCS).
   Reason: exactly the thesis's own Sec. 3.7.4 argument — every surrogate
   interpolates 20 points, so training scores never separate catastrophic
   extrapolators from good models; near-uniform weights let models wrong by
   thousands contaminate the mixture. Hard rejection tolerates a weak score
   because it only needs the *ranking*; soft averaging needs *magnitude*
   separation.

2. **Consensus-deviation probe scores fall into the spurious-consensus trap.**
   Penalising deviation from the (weighted) committee consensus at probe
   points down-weights the one *correct* dissenter — PCE-d2 — precisely
   because the GP/DNN majority prior-reverts together. This is the
   failure-trial mechanism of thesis Sec. 6.5.2 resurfacing inside the model
   manager. CV error avoids it by scoring each model against *data*, never
   against the committee.

3. **Diagnostic on the hard-LCS runs (important for the defence):**
   instrumented runs show hard LCS sequentially rejects all eight non-PCE-d2
   models, the final "ensemble" is PCE-d2 alone, and **no query lands in the
   oscillatory interior (r < 30)**; final RMSE ≈ 0.7 ≈ the RMS of the
   unresolved sin·cos term. So on this codebase the success mechanism is the
   **anchor-model funnel** (thesis Sec. 6.8) — consistency-based model
   selection rescuing a misspecified majority — rather than a UEigV interior
   discovery event. The thesis already flags the PCE-d2 anchor caveat
   (Secs. 6.10, 7.4); the paper should *lead* with this framing rather than
   the discovery-event narrative, because a referee who reruns the code will
   find the same thing. It is a cleaner and equally strong story: at equal
   budget, model selection by consistency beats any fixed model when the
   majority is misspecified, and the spectral metric supplies the
   disagreement signal that makes the selection identifiable.

---

## 3. Literature map (claim → citations to add)

| Component / claim | Anchor references |
|---|---|
| Surrogate-accelerated Monte Carlo UQ (the `UQAnalyzer` stage) | Marzouk & Najm, *JCP* 2009; Heidenreich et al., *Metrologia* 2017 (already cited in thesis) |
| Probability boxes / bounded CDFs from model-form uncertainty | Ferson & Ginzburg, *Reliab. Eng. Syst. Saf.* 54:133–144, 1996; Beer, Ferson & Kreinovich, *Mech. Syst. Signal Process.* 2013 (imprecise probabilities review) |
| Law-of-total-variance epistemic/aleatoric split in ensembles | Hüllermeier & Waegeman 2021 (already cited); Depeweg et al., ICML 2018 (decomposition in BNN ensembles) |
| Sobol estimators on surrogates | Saltelli et al., *Comput. Phys. Commun.* 181:259–270, 2010; Jansen, *Comput. Phys. Commun.* 117:35–43, 1999; Sudret, *Reliab. Eng. Syst. Saf.* 93:964–979, 2008 (PCE-based Sobol — cite as the analytic alternative) |
| CV/PRESS inverse-error weighted surrogate ensembles | Goel, Haftka, Shyy & Queipo, *Struct. Multidisc. Optim.* 33:199–216, 2007; Viana, Haftka & Steffen, *Struct. Multidisc. Optim.* 39:439–457, 2009; Acar & Rais-Rohani, *Struct. Multidisc. Optim.* 37:279–294, 2009 |
| Bayesian model averaging (the soft-rejection framing) | Hoeting, Madigan, Raftery & Volinsky, *Stat. Sci.* 14:382–417, 1999; Yao, Vehtari, Simpson & Gelman, *Bayesian Anal.* 2018 (stacking — why BMA over-concentrates, supports CV-based weights) |
| Physics-informed soft constraints | Karniadakis et al., *Nat. Rev. Phys.* 3:422–440, 2021; Swiler et al., *J. Mach. Learn. Model. Comput.* 1, 2020 (survey of constrained GPs) |
| Distribution-free calibrated intervals (future work / reviewer-proofing) | Angelopoulos & Bates, *Found. Trends ML* 2023 (conformal prediction) |

---

## 4. Paper plan

**Venue.** First choice *Reliability Engineering & System Safety* (your
Moustapha/Sudret and Ferson lineage lives there, and they value honest regime
delineation); alternatives *Structural Safety*, *SIAM/ASA J. Uncertainty
Quantification*. ~8,000 words / 30–35 pages double-spaced; this means
compressing the 84-page thesis roughly 4:1.

**Headline framing.** Lead with the two-condition failure criterion (large
domain × high dynamic range, Schwefel control) — it is the falsifiable,
quotable result — then present the framework as the remedy and the UQ
deliverables as the payoff. Reframe the win mechanism as consistency-based
model selection (anchor funnel) with the spectral metric as the disagreement
signal, per Sec. 2.3 above.

**Structure (target lengths).**
1. Introduction — 1.5 pp. Merge thesis Ch. 1–2; keep only the four-gap
   paragraph and the Hanna/Tan motivation.
2. Failure characterisation — 2 pp. GP prior reversion math (Eq. 6.1),
   domain-size ablation figure, Schwefel control. This is the headline.
3. Framework — 2.5 pp. Committee table, UEigV (one equation + Fig. 3.2),
   LCS hard + soft (CV-weighted) as two variants of one model-management
   principle, with the negative-result ablation in 1 paragraph.
4. UQ deliverables — 2 pp. **New.** p-box + exceedance, variance
   decomposition trajectory, Sobol with epistemic bars; validation against
   ground-truth MC on the synthetic benchmark.
5. Cross-benchmark results — 1.5 pp. Table 5.1 + Fig. 5.1 only; one
   paragraph per regime. Cut Figs. 5.3–5.5 to supplementary.
6. Discussion + limitations — 1 p. Anchor-model caveat front and centre.
7. Conclusion — 0.5 p.

**Figures to keep (6–7 max).** Domain ablation (5.6); cross-benchmark ratio
bar (5.1); UEigV heatmap (6.1); RMSE trajectory bands (5.8); plus the three
new UQ figures (p-box, decomposition trajectory, Sobol). Drop slice
reconstructions to supplementary.

**Cut entirely.** Thesis Ch. 2 subsections 2.2.1–2.2.3 (textbook material),
Nomenclature, Appendix A (point to the GitHub repo), the per-benchmark
function definitions (cite Surjanovic & Bingham's test-function library or
the original papers and give a one-row-per-function table).

### 30-trial paired campaign (completed; seeds 1000-1029, this repo's code)

Physics benchmark, N=20 budget, identical seeds/initial designs per pair:
hard LCS median RMSE 0.71, success 27/30 (90%); soft CV-weighted median 0.64,
success 26/30 (87%); Wilcoxon signed-rank on log-RMSE p=0.096 (parity). Only
one failure seed is shared, so failures depend on the stochastic query
trajectory, not just initial LHS coverage. Coverage study: hard LCS collapses
to exactly 1 surviving model in all 30 trials, so every epistemic band is
degenerate (0% coverage, zero-width p-box). Soft retains all 9 members:
mean-band covers the true output mean in 97% of trials, p-box brackets 57% of
the true CDF on average, exceedance interval under-covers in the far tail
(13%) -> conformal calibration is the follow-up. Central accuracy identical
(median q50 relative error 0.24% both). Raw per-trial records:
results_hard.json / results_soft.json; rerun via
`python3 campaign.py <hard|soft> <start> <n>` (resumable).

Note: this campaign ran with the sklearn-MLP DNN backend (identical
architecture/optimiser; torch used automatically when installed). Rerun on
your machine with torch to confirm before submission.

**Before submission, run on the cluster:** (i) the full 30-trial hard-vs-soft
comparison (the 3-trial smoke test here is indicative only); (ii) the UQ
validation table (surrogate-MC vs truth-MC moments/quantiles/exceedance) over
30 seeds so you can report coverage of the epistemic bands — i.e. how often
the p-box brackets the true CDF. If coverage is poor, say so; if it is good,
it becomes the strongest figure in the paper.

---

## 5. Honest limitations of the additions (state these)

- The epistemic bands are committee envelopes, not calibrated probabilities:
  they can be over- or under-confident, and K=9 models give a coarse
  envelope. Conformal prediction is the principled upgrade (future work).
- CV at N≈20 is itself noisy (especially the single-holdout score for DNNs);
  weights should be read as order-of-magnitude statements, which is all
  inverse-error weighting needs.
- The Sobol estimators inherit any surrogate bias; on the physics benchmark
  this is visible in the (validated) error vs truth and should be reported.
- Soft weighting adds k×K refits per update; with `update_every=3` and 3-fold
  CV the overhead is ~2× the base loop on this committee — negligible next
  to one CFD run, but worth a sentence.
