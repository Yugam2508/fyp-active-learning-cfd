# Surrogate models for GUM S1 uncertainty propagation with few model runs

Monte Carlo propagation of distributions (GUM Supplement 1) through a
surrogate fitted to 10–80 runs of an expensive model, motivated by CFD
modelling of hazardous gas dispersion. The repository started as a final-year
project (heterogeneous surrogate committee + spectral disagreement
acquisition + dynamic model management) and now holds a controlled study of
what actually matters, with every per-seed record and the scripts that
regenerate every table, figure and quoted number.

Authors: Yugam Jinesh Vora (NUS, corresponding, yugam@u.nus.edu) and
Wee Hoe Ng (National Metrology Centre, A*STAR). Cite via `CITATION.cff`.

## Status — read this first

This repository is a **research record, including negative results.**

| Deliverable | Where | State |
|---|---|---|
| IMEKO World Congress 2027 extended abstract (4 models, N = 20, coverage validation, CFD case) | `paper/imeko2027/` | ready to submit (see its README); numbers audited by `check_numbers.py` |
| Journal manuscript (10 models, N = 10–80, coverage validation, CFD case, procedure) | `paper/paper.tex` → `paper/paper.pdf` | complete draft; numbers audited by `paper/check_numbers_journal.py` |

Findings (30 paired seeds per strategy; see the journal manuscript for
effect sizes with bootstrap intervals and Holm-adjusted tests):

| Claim | Status | Evidence |
|---|---|---|
| Passive GP surrogates fail on large domains (earlier analysis) | **withdrawn — artefact** | the original kernel fixed the amplitude; estimating it: physics nRMSE 0.59 → 0.033 |
| GP specification (amplitude, ARD, quadratic trend, log scale for positive outputs) is first-order | supported | trend 400× on physics; log scale 1.3–9× on all six positive analytical models and 6× on the CFD case |
| Choosing the surrogate by leave-one-out (LOO) error works without the true model | supported | median nRMSE within 0–4.4 % of the best candidate in hindsight on all 10 models |
| A committee beats a well-specified GP | **not supported** | never better than a GP with trend; 1.7–2.4× worse on 4 of 10 models |
| CV inverse-error weighting is needed for committees | supported | never worse than uniform, up to 2000× better |
| Variance-driven adaptive designs help | **budget-dependent** | GP-SD design: never at N ≤ 20, at N = 40–80 on smooth models (0.22–0.87×), worse on the plume; adaptive committee better than the passive one on 4 of 7 models (incl. CFD) but worse exceedance estimates on plume and CFD |
| UEigV (spectral) beats variance acquisition | **refuted** | never better; worse on 3 of 6 models |
| Threshold-targeted design (AK-MCS) helps exceedance probabilities | supported for tail thresholds | error reduced 2–9× in 8 of 15 analytical comparisons, never worse; no gain on the CFD case, where the threshold is exceeded in 67 % of cases |
| The GP's predictive variance is a valid surrogate contribution | **not supported** | pointwise coverage down to 68 %; intervals for y95 cover the truth in a median 53 % of seeds |
| LOO residuals added to each Monte Carlo trial give a valid contribution | supported | GUM S1 95 % interval coverage 91–97 % on all models; jackknife+ 95–99 % pointwise |

Limits: up to eight inputs; 2-D steady RANS CFD case with a reference from
800 runs (its own uncertainty ≈ 3 % for y95); adaptive committees only to
N = 40. See the journal manuscript, Section 6.3.

## Layout

```
src/                       library modules (importable)
  framework_core.py          committee, UEigV, hard LCS, ActiveLearner (FYP)
  uq_extension.py            SoftLCS (CV inverse-error weights), UQAnalyzer
  surrogates.py              GP specifications (orig ... quad, log) and sparse PCE (LAR)
  benchmarks.py              13 analytical models (uniform, GUM-type and dependent
                             inputs; a failure-mode model)
  procedure.py               the procedure of the paper: selection, LOO contribution,
                             adaptive Monte Carlo (GUM S1 7.9), jackknife intervals,
                             uncertainty budget, GUM-framework comparison
  gp_bayes.py                Student-t and MCMC GP predictive distributions, IMSE
  designs.py                 random LHS, maximin LHS, scrambled Sobol' designs
  cfd_dispersion.py          parametrised OpenFOAM case (2-D RANS, building wake)
  cfd_benchmark.py           CFD case as a cached 6-input benchmark + reference sets
experiments/               runnable studies, launch from anywhere
  campaign_generalisation.py main comparison: 17 strategies x 10 models x N
  campaign_coverage.py       validity of the surrogate contribution (GP, CV, jackknife+)
  campaign_selection.py      LOO errors of the candidates (selection step)
  run_journal_campaign.sh    all analytical-model jobs of the journal study
  run_coverage_campaign.sh, run_selection_campaign.sh, run_cfd_campaign.sh
  cfd_precompute.py          the 1550 CFD runs (designs + reference), cached
  cfd_grid_check.py, cfd_iteration_check.py, cfd_reference_stats.py, fig_cfd_case.py
  analyse_journal.py         statistics, tables, figures (--check for CI)
  run_extension_campaign.sh, run_completion_campaign.sh
                             higher-dimensional models; the research-plan runs
  crosscheck_openturns.py, campaign_stacking.py, campaign_vector.py,
  cfd_gci.py, cfd_reference2.py, worked_example_cfd.py
  analyse_completion.py      statistics of the research-plan runs
  check_benchmarks.py        formulas and input models vs UQTestFuns
  test_reproducibility.py    regenerate published records and compare
  analyse_generalisation.py, make_paper_figures.py, campaign_domain.py,
  campaign_error_estimation.py   conference-paper scripts
  campaign.py, campaign_ablations.py, campaign_hardmin.py, decompose_soft_vs_hard.py,
  demo_uq.py                     earlier FYP campaigns (physics model)
results/                   per-seed JSON records (resumable), cfd_runs/ cache,
                           journal/ (summary.json + generated LaTeX tables)
figures/                   generated figures (journal/ for the journal manuscript)
paper/                     imeko2027/ (extended abstract), paper.tex + journal_*.tex
tests/                     unit tests (python -m pytest -q)
docs/                      research plan, reproducibility record, reference checks
```

## Setup

```bash
pip install -r requirements-lock.txt   # exact versions (CPU PyTorch: add
                                       # --extra-index-url https://download.pytorch.org/whl/cpu)
sudo apt-get install openfoam          # only to recompute CFD runs (v1912)
```

`requirements.txt` gives minimum versions. Without PyTorch the committee's
neural networks fall back to a scikit-learn MLP, which changes committee
results slightly.

## Reproducing

```bash
python experiments/check_benchmarks.py           # formulas vs UQTestFuns
python experiments/test_reproducibility.py       # published records still reproduced
bash experiments/run_journal_campaign.sh 4       # analytical models, all arms and budgets
bash experiments/run_coverage_campaign.sh 4      # surrogate-contribution validation
bash experiments/run_selection_campaign.sh 2     # LOO selection
CFD_WORKERS=4 python experiments/cfd_precompute.py   # CFD runs (or use results/cfd_runs/)
bash experiments/run_cfd_campaign.sh 4           # CFD case, all arms
python experiments/analyse_journal.py            # tables + figures
python paper/check_numbers_journal.py            # audit the journal prose
python paper/imeko2027/check_numbers.py          # audit the extended abstract
```

All campaigns are resumable (finished seeds are skipped); seeds are
1000–1029, paired across arms. Full details, environment and caveats:
`docs/REPRODUCIBILITY.md`.

## Papers

Build from the paper directory:
`pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`
(and the same with `extended_abstract` in `paper/imeko2027/`). Tables are
`\input` from `results/journal/` and figures read from `figures/`.

## Licence

MIT (see LICENSE).
