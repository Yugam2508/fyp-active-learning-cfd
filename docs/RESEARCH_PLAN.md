# Research plan to completion

Written 2026-10-19, before any of the runs below, so that the decision rules
are fixed in the git history before the results exist.

## When the project is complete

No study is ever literally finished. This one is complete when:

1. every claim in the papers has been tested against each alternative
   explanation a reviewer can reasonably raise (Tiers 1 and 2);
2. every remaining question is out of scope for a stated reason and is named
   in the paper as a limitation or future work (Tier 3). Each item there
   would be a new paper, not a gap in this one;
3. every number is regenerated from the records by a script, audited, and
   archived with a DOI (Writing and submission).

Each item below gives the question, what is run, the cost (wall time on the
4-core machine), and a **decision rule**: which result changes which
statement in the papers.

## Status

| Done | In progress (extension campaign, about 4–5 h left) |
|---|---|
| 10 models (d ≤ 8), N = 10–80, 30 paired seeds, Holm-adjusted tests, bootstrap intervals; coverage validation at N = 20; LOO selection; 2-D RANS CFD case with an 800-run reference, grid and iteration checks; procedure module, worked example with budget; unit tests, cross-machine reproducibility CI | wing weight (d = 10) and OTL circuit with 20 inputs; AK-MCS at the reference 95th percentile (plume, plume GUM, CFD); coverage of the log GP with trend; end-to-end coverage of the procedure (selection + contribution) on all models |

## Tier 1: would be a major reviewer comment if missing

**T1.1 Coverage, selection and the procedure across budgets.**
The surrogate contribution and the selection are validated only at N = 20.
- *Run:*
  - coverage campaign at N = 10 and 40 for the six candidates on all 12 models;
  - at N = 80, the three candidates matching the output (GP and GP with trend
    on the raw or log scale, plus the matching sparse PCE) on the 11
    analytical models;
  - the CFD budget sweep and coverage at N = 10 and 40 (about 1500 new CFD
    runs).
- *Note:* below 19 runs a 95 % jackknife+ interval is unbounded (rank
  ⌈0.95(N+1)⌉ > N). This is reported as a result, not hidden.
- *Decision:* if the GUM S1 interval with the LOO contribution covers
  ≥ 90 % at every N on every model, the claim holds across budgets.
  Otherwise the paper states the N below which it fails.
- *Cost:* about 14 h.

**T1.2 Is the GP's overconfidence an artefact of plug-in hyperparameters?**
The GP intervals use maximum-likelihood amplitude and length scales. The
Bayesian emulator literature integrates them out, which widens the
intervals at small N. This is the strongest objection to our central claim.
- *Run:* three further interval types for every GP candidate in the
  coverage campaign:
  - (a) Student-t predictive with the amplitude and constant mean integrated
    out (the standard emulator form);
  - (b) full posterior over all hyperparameters by MCMC (log-normal priors,
    about 200 posterior draws, mixture predictive);
  - (c) the GP SD recalibrated by the LOO standardised residuals
    r_i / s_-i(x_i), a hybrid that keeps the GP's localisation and takes
    its scale from the data.
- *Decision:* if (a) or (b) reaches ≥ 93 % pointwise coverage and nominal
  coverage for derived quantities on all models, the claim becomes "the
  plug-in GP variance is not a valid contribution; the Bayesian GP is", and
  both abstracts change. If (c) is valid, it becomes a recommended
  alternative in the procedure. Otherwise the claim stands, stronger.
- *Cost:* about 3 h.

**T1.3 Baseline fidelity against an independent implementation.**
"Sparse PCE is 1.2–12× worse than the GP" depends on our PCE and GP
implementations being competent.
- *Run:* OpenTURNS (PyPI, 1.27) LARS sparse PCE with hyperbolic truncation
  and corrected LOO, and OpenTURNS kriging (Matérn 5/2, ARD, quadratic
  basis), on the same 30 designs of six models at N = 20, paired with ours.
- *Decision:* agreement within the T1.4 margin → state it in the methods. If
  OpenTURNS is better, use the better implementation's results and revise
  the PCE claim.
- *Cost:* about 2 h.

**T1.4 Equivalence statements for the null results.**
"No consistent gain" (committees, adaptive designs, kernel choice on CFD)
can mean either absence of evidence or evidence of absence.
- *Run:* for every non-significant comparison, check whether the 95 %
  bootstrap interval of the median ratio lies inside a margin fixed now:
  **[0.8, 1.25]**, also reported at [0.9, 1.11]. Analysis only.
- *Decision:* each null claim is rephrased as "equivalent within 20 %" or
  "inconclusive with 30 seeds".

**T1.5 Per-design (conditional) coverage.**
The coverage figures are averages over designs; a practitioner has one
design.
- *Run:* the distribution over seeds of the achieved coverage (GUM S1
  interval, pointwise intervals) and the fraction of designs below 90 %.
  Analysis of existing records.
- *Decision:* added to the coverage section. If many designs are below
  90 %, the limitations say so with the number.

**T1.6 A failure mode: a feature the runs cannot see.**
LOO residuals can only reveal errors that some run exposes.
- *Run:* a new analytical model, a smooth function plus a narrow bump
  carrying about 10 % of Var(Y) in about 5 % of the input volume. Main arms
  at N = 20 and 40, plus coverage; the probability that N runs miss the
  bump is computed.
- *Decision:* the failure is reported with its mechanism. The procedure
  gains the caveat that local features smaller than the design spacing
  need prior knowledge or targeted runs.
- *Cost:* about 1.5 h.

**T1.7 GUM S1 conformance of the procedure.**
- *Run:*
  - the adaptive Monte Carlo procedure of GUM S1 §7.9 in `procedure.py`, so
    that M is chosen for a stated numerical tolerance of u(y) and the
    interval limits;
  - the worked example reports that tolerance;
  - a comparison with the GUM uncertainty framework (law of propagation of
    uncertainty, through the same surrogate) on the plume and CFD cases,
    showing why the Monte Carlo method, and hence a surrogate, is needed
    there.
- *Cost:* under 1 h.

### How the thresholds are computed (added 2026-10-19 04:55 UTC, before any Tier 1 run started)

- **Pointwise coverage** of a model and surrogate is the mean over the 30
  seeds of the share of the 5000 test points inside the interval.
- **Coverage of the GUM S1 interval** is the mean over seeds of the
  reference probability mass inside the interval.
- **For T1.1**, the interval is that of the candidate chosen by the LOO
  selection in each seed, so the selection is part of what is tested.
- **"Nominal" coverage for a derived quantity** (mean, SD, y95, P(Y > T)) means
  at least 26 of 30 seeds cover the true value. That is the smallest count
  not rejected against 95 % by a one-sided exact binomial test at the 5 %
  level.
- **T1.3 and T1.4 equivalence** use the 95 % bootstrap interval of the median
  paired ratio (10 000 resamples), as in the paired-test table. "Equivalent"
  means the whole interval lies in the margin.
- **Results kept apart from the main study:**
  - the bump model (T1.6) and the correlated-input model (T2.5) are
    reported in their own subsections, not pooled into the main twelve;
  - records from other design types carry a `design` field and are
    excluded from the main summaries.

### Finding during T1.3 (2026-10-19, recorded before the write-up)

The OpenTURNS GP was 9 times more accurate than our Matérn GP on the
physics model, and agreed within 5 % on the five other models. The cause:

- The estimated kernel amplitude reached the upper end of its pre-specified
  search range (10³, standardised output units) in nearly every design
  for the isotropic, ARD and Matérn GPs on the physics model.
- Elsewhere the bound was reached only for the log-scale GP on the CFD case,
  in 3 of 6 designs. A check of all models and specifications on 6 designs
  each found no other case.
- Rerun with the bound at 10⁶ (`experiments/sensitivity_amp_bound.py`),
  the median nRMSE on the physics model fell from 0.033 to 0.00097
  (isotropic), from 0.036 to 0.0022 (ARD) and from 0.079 to 0.013 (Matérn).
  On the CFD case it changed from 0.080 to 0.077.

The study's records keep the pre-specified bound. The paper reports the
sensitivity result: the bound understates the benefit of estimating the
amplitude on the physics model; no conclusion is reversed.

## Tier 2: closes the remaining minor comments

| Item | Question | Run | Cost |
|---|---|---|---|
| T2.1 Design type | Do the results depend on random LHS? | maximin LHS and scrambled Sobol' designs for the main GP arms and the procedure at N = 20, all models (CFD: about 1200 new runs) | 3 h |
| T2.2 IMSE design | Is "adaptive gives no gain at N ≤ 20" specific to the max-SD criterion? | integrated-variance (IMSE) sequential design at N = 20 and 40 | 3 h |
| T2.3 Stronger candidates | Would PC-Kriging or stacking change the selection result? | PC-Kriging (GP with the LAR-selected PCE basis as trend) and non-negative LOO stacking of the six candidates, added to selection and coverage | 2 h |
| T2.4 Vector output (GUM S2) | Does run-by-run joint resampling of residual vectors give valid joint coverage? | plume model at three receptors; joint against component-wise resampling | 1 h |
| T2.5 Correlated inputs | Does the procedure hold with dependent inputs? | borehole (GUM) with a Gaussian copula, ρ = 0.5 between two inputs; the budget is either replaced by Shapley effects or flagged as needing independence | 1 h |
| T2.6 CFD numerics | Can the CFD validation resolve errors below 5 %? | 800 more reference runs (halves the reference variance); a third grid level at the six check states for a grid convergence index | 2 h |
| T2.7 Stopping rule | Does the estimated surrogate share predict the true propagated error? | from the T1.1 records: estimated share against actual error across N; a rule "add runs while the share exceeds x %" | analysis |
| T2.8 Cost table | What does each strategy cost next to one CFD run? | from the stored runtimes | analysis |
| T2.9 Theory note | Why should resampled LOO residuals give valid GUM S1 intervals? | a short argument under exchangeability, linked to conformal predictive distributions (references verified as for the rest) | writing |
| T2.10 Long-term reproducibility | Will OpenFOAM v1912 still install in five years? | Dockerfile (Ubuntu, OpenFOAM v1912, Python lockfile); Zenodo archive | writing |

Decision rules for Tier 2: a result that contradicts a paper statement
replaces it. Otherwise the result is one sentence and a supplementary table.

## Tier 3: out of scope (the stop line, written into the paper)

| Not done | Reason |
|---|---|
| 3-D, unsteady or LES CFD; validation of the CFD against wind-tunnel data | concerns the validity of the model, not the propagation; about 100× the cost |
| Multi-fidelity (Gaussian plume as a low-fidelity model of the CFD, co-kriging) | a different method class; the natural follow-up paper |
| d > 20, N > 80 | outside the regime of a few runs of an expensive model |
| Stochastic simulators, field and time-series outputs | different error structure; separate study |
| Neural or deep surrogates at N ≤ 80 | data-starved; NN members already tested inside the committee |
| SUR and other Bayesian exceedance designs | AK-MCS is the reliability standard; with the quantile-targeted variant it covers the targeted case |
| Bayesian inverse problems (scatterometry-type) | inverse, not forward, propagation |
| A real NMC measurement case | needs NMC data; the one addition that would most raise impact for a metrology journal, possible only with Wee Hoe |

Considered and not needed: more seeds (the bootstrap intervals already
resolve every claimed effect, and T1.4 handles the nulls); further kernel
families (the kernel family mattered far less than amplitude, ARD, trend
and log scale); larger analytical references (2 × 10^5 draws).

## Writing and submission

- **W1** Rewrite the results, discussion, procedure and conclusion after
  Tiers 1–2. The abstracts change only where a decision rule above
  triggers.
- **W2** Target journal (your decision; for example Metrologia,
  Measurement, Meas. Sci. Technol., Int. J. Uncertain. Quantif.): template,
  word limit, highlights and graphical abstract where required.
- **W3** Overlap with the IMEKO extended abstract: cite it, declare it in
  the cover letter, check the IMEKO 2027 copyright terms.
- **W4** Wee Hoe: review of both manuscripts, any NMC / A*STAR publication
  clearance, re-approval of the changed extended abstract.
- **W5** Terminology audit against the VIM and GUM (measurand, coverage
  interval, standard uncertainty); notation consistency; colour-blind-safe
  figures; supplementary material (long test table, per-model figures).
- **W6** Final audits: both number audits, unit tests, reproducibility CI,
  references re-verified, PDF compliance check for EDAS.
- **W7** Tagged release with a Zenodo DOI cited in both papers (you enable
  the integration).

## Schedule

| Stage | Contents | Wall time |
|---|---|---|
| 0 | extension campaign (running) | 4–5 h |
| 1 | T1.1, T1.2, T1.3, T1.6, T1.7 | about 20 h |
| 2 | T2.1–T2.6 | about 10 h |
| 3 | analyses T1.4, T1.5, T2.7, T2.8; writing W1, W5, W6, T2.9, T2.10 | my time, no compute |

All campaigns are resumable. Each stage is committed and pushed as it
finishes, so a reclaimed container loses at most one stage.

## What could change the headline claims

| Item | Possible outcome | Effect |
|---|---|---|
| T1.2 | a Bayesian or recalibrated GP is valid | "GP variance invalid" becomes "plug-in GP variance invalid"; both abstracts reworded |
| T1.3 | OpenTURNS PCE is clearly better than ours | PCE claim weakened, numbers replaced |
| T1.1 | coverage fails at N = 10 | recommendation restricted to N ≥ some value |
| T1.6 | (expected) coverage fails on the bump model | new limitation and procedure caveat; no claim withdrawn |
