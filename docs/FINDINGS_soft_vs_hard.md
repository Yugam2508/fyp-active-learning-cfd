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

---

## Update 2026-10-16 — reruns for the IMEKO WC 2027 extended abstract

All runs below use PyTorch DNNs and the fixed Latin-hypercube seeding
(`docs/REPRODUCIBILITY.md` §3); 30 seeds each. Numbers are recomputed by
`paper/imeko2027/check_numbers.py`.

### Fair baseline, now complete (`results/results_hardmin3.json`)
`campaign_hardmin.py` now scores **both** aggregation rules on the **same**
final survivors and data in every trial, replacing the 12-seed sklearn row.

| K_min = 3 survivors, same data | Median RMSE | Success |
|---|---:|---:|
| uniform mean | 1556 | 0/30 |
| CV inverse-error weights | 0.701 | 27/30 |

PCE-d2 survives in 27/30 trials. Weighting succeeds in all 27 and fails in
exactly the 3 without it (seeds 1006, 1012, 1014); uniform averaging fails in
all 30. Paired Wilcoxon p = 5.6e-9. The aggregation rule is causal, but it
can only exploit an anchor that is present.

### Domain sweep reproduced, interpretation revised (`results/results_domain_*.json`)
The thesis ratios reproduce approximately: ρ = 1.04 at h = 2 (thesis 1.06),
0.51 at h = 10 (0.384), 0.030 at h = 200 (0.044), Schwefel 1.10 (1.09). The
thesis's "peak advantage at h = 50" does not: here ρ falls monotonically up
to h = 200.

Normalised by output SD, however, the **passive GP sits at 0.60 SD for every
h ≥ 10 and at 1.00 SD for Schwefel**: it is close to a constant predictor in
both. The committee's absolute median RMSE stays at 0.62–0.68 (h = 2 and
h ≥ 50), which is the RMS of the sin·cos + exp terms (0.50–0.61) that no
member learns from 20 runs. The quadratic term (44 % of the variance at h = 2,
≈ 100 % for h ≥ 10) is captured exactly by PCE-d2. At h = 10 hard rejection
keeps PCE-d2 in only 15/30 trials (median RMSE 1.99).

**Consequence:** the old "large domain **and** high dynamic range"
criterion is not isolated by the Schwefel control, because there the GP also
fails and no committee member matches the response either. The supported
statement is: *a passive GP fails once run spacing is large relative to the
response length scale; a committee helps only to the extent that a member
represents the dominant component.* The README and the extended abstract
use this wording.

### Representative run (`results/representative_run_seed1000.json`)
Seed 1000 (first seed, not selected): RMSE 0.633; mean 0.21 %, SD 0.14 %,
q05/q50/q95 0.66/0.24/0.41 % from the 2×10⁵ reference. The final weights put
0.999999 on PCE-d2, so the weighted model-form share falls from 0.50 to 7e-8,
consistent with the true surrogate error share RMSE²/Var Y ≈ 2e-8. The p-box
and exceedance interval are **unweighted** member envelopes, so they remain
wide ([0, 0.058] around the true 0.05). The earlier text's "0.76 → ≈ 0" had
no stored record; it is replaced by these values.

### Decomposition rerun with PyTorch (`results/results_decompose.json`)
Confirms the sklearn-backend record in `results/reference_sklearn/`: hard
rejection collapses to one member in 30/30 trials, at median iteration 10 of
15; 139 of 450 queries (31 %) are then random. The sole survivor is PCE-d2 in
27/30 (PCE-d3 2, DNN-H64 1). The predictor swap (hard trajectory, CV-weighted
predictor) changes nothing: p = 0.90, 26/30 pairs within 1 %.

### Extended abstract changes
- The Sobol sentence was removed from the IMEKO abstract: no Sobol result is
  shown or validated there.
- Table III median-quantile error for the weighted committee is 0.23 %
  (0.235 %); earlier drafts rounded it to 0.24 %.
