# What actually establishes "soft beats hard"? — and why the claim must change

## The test that was run (campaign.py)
- 30 paired trials, seeds 1000–1029. Same seed ⇒ same initial 5-point LHS design.
- Physics benchmark [-200,200]^3, budget N=20 (5 LHS + 15 AL steps).
- RMSE on a fixed 5,000-point test set (RandomState(7)), identical across all arms.
- Statistic: Wilcoxon signed-rank on paired log-RMSE.
- Result (user's torch run): soft median 0.647 vs hard 0.681, 29/30 vs 27/30 success,
  p = 0.0037. Restricted to the 27 seeds where both succeed: p = 0.0055, soft wins
  19/27 (sign test p = 0.052), ~8% lower RMSE. So it is not driven only by rescuing
  hard's failures — but the effect is modest.

## Why that test cannot support the claim as written
The two arms differ in THREE ways at once:
  1. model manager      permanent rejection    vs  reversible soft weights
  2. prediction rule    uniform mean of        vs  CV inverse-error weighted
                        surviving models           average over all K
  3. acquisition set    UEigV over survivors   vs  UEigV over unmuted models

## Decomposition (decompose_soft_vs_hard.py, 30 seeds)
- **Predictor swap, trajectory held fixed:** hard's own final dataset, predicted with
  the CV-weighted all-9 ensemble instead of the survivor mean. Wilcoxon p = 0.237;
  25/30 pairs identical to within 1%. → the weighting adds nothing *as a predictor*
  here, because both reduce to PCE-d2.
- **Hard LCS collapses to K=1 in 30/30 trials**, at median AL iteration 10 of 15.
  Surviving model: PCE-d2 (26/30), PCE-d3 (2), GP-RBF (2).
- With K<2, `ActiveLearner.ueigv()` returns all-ones ⇒ `argmax` selects candidate 0
  of a uniformly random pool. **≈31% of hard's active-learning budget is silently
  uniform random sampling.** Soft is protected by its `if active.sum() < 2` guard.
  ⇒ the published soft-vs-hard accuracy comparison is against a degenerate baseline.

## Fair baseline (campaign_hardmin.py) — the decisive experiment
Hard LCS with a minimum committee size of 3 (the only change):

| Manager                | Aggregation                | Median RMSE | Success | Diversity |
|------------------------|----------------------------|------------:|--------:|-----------|
| Hard, min=1 (published)| uniform mean (=PCE-d2)     |       0.695 |    93%  | none (K=1)|
| Hard, min=3            | uniform mean of 3          |     1763    |     0%  | yes       |
| Hard, min=3            | CV inverse-error weights   |       0.729 |   100%  | yes       |
| Soft (min 2 active)    | CV inverse-error weights   |       0.647 |    97%  | yes       |

(rows 1–2 and 4: 30 seeds; row 3: 12 seeds. Rows 1–3 sklearn-MLP backend; row 4 is the
user's torch run. Rerun with torch to confirm — but the mechanism is arithmetic, not
stochastic.)

## Corrected interpretation
1. **The causal ingredient is the aggregation rule, not the softness of rejection.**
   Hard rejection + CV inverse-error weighting works (0.729, 100%). Reversibility
   contributes nothing measurable on this benchmark.
2. **Uniform averaging over a heterogeneous committee is catastrophic in the
   prior-reversion regime** (0/30 success, median 1763), because prior-reverted GPs/DNNs
   dominate the mean even when the correct model (PCE-d2) is present.
3. **Hard rejection to K=1 trades away all epistemic information** (zero-width p-box)
   and silently degrades acquisition to random sampling.
4. Therefore the p-box coverage comparison (68% vs 0%) is near-tautological: a
   single-model "ensemble" has zero-width bands by construction.

## What survives as a claim
NOT: "soft weighting is a better model manager."
YES: "In the GP prior-reversion regime, committee aggregation must be error-weighted.
Uniform averaging fails catastrophically; hard rejection to a single model recovers
accuracy only by discarding all epistemic information and degrading the acquisition.
CV inverse-error weighting attains both accuracy and usable epistemic bands."

Caveat: CV inverse-error weighting IS the weighted-average surrogate of
Goel et al. (2007) / Viana et al. (2009) — prior art. So this is a characterisation
result (when and why it is necessary), not a new method.

## Open, unresolved
- All of the above is one synthetic benchmark whose dominant term matches PCE-d2's
  hypothesis class exactly. `nopce` (committee without PCE) fails outright. Nothing
  here generalises until it is repeated on benchmarks with no anchor model.
