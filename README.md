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
| Journal manuscript (12 models, N = 10–80, coverage validation, CFD case, procedure, research-plan checks) | `paper/paper.tex` → `paper/paper.pdf` | complete draft; numbers audited by `paper/check_numbers_journal.py` |
| The same, anonymized for *Measurement* (Elsevier), with title page, cover letter and highlights | `paper/measurement_*.tex` | items marked [CONFIRM] need the authors' decision |

The research plan that defines when the study is complete, with thresholds
fixed before the runs, is `docs/RESEARCH_PLAN.md`.

Findings (30 paired seeds per strategy; see the journal manuscript for
effect sizes with bootstrap intervals and Holm-adjusted tests):

| Claim | Status | Evidence |
|---|---|---|
| Passive GP surrogates fail on large domains (earlier analysis) | **withdrawn — artefact** | the original kernel fixed the amplitude; estimating it: physics nRMSE 0.59 → 0.033 (and lower with a wider search range) |
| GP specification (amplitude, ARD, quadratic trend, log scale for positive outputs) is first-order | supported | trend 400× on physics; log scale up to 9× on positive models, 6× on the CFD case; confirmed against OpenTURNS |
| Choosing the surrogate by leave-one-out (LOO) error works without the true model | supported from 20 runs | within 4.4 % of the best candidate in hindsight at N = 20–80; up to 37 % worse at N = 10 |
| A committee, or stacking of the candidates, beats the selected surrogate | **not supported** | committee never better than a GP with trend; stacking 0.96–1.17 × the selection |
| Variance-driven adaptive designs (GP-SD, IMSE) help | **budget-dependent** | none at N = 20; at N = 40 IMSE better than passive on 6 of 9 smooth models |
| The type of space-filling design (random/maximin LHS, Sobol') matters | **not supported** | 2 of 120 comparisons significant; coverage unchanged |
| UEigV (spectral) beats variance acquisition | **refuted** | never better; worse on 3 of 6 models |
| Threshold-targeted design (AK-MCS) helps exceedance probabilities | supported for tail thresholds | error reduced 2–9×, never worse; no gain where the threshold lies in the bulk |
| The GP's predictive variance is a valid surrogate contribution | **not supported** | pointwise coverage down to 68 %; intervals for y95 nominal in 13 of 42 cases (24 with MCMC hyperparameters) |
| LOO residuals added to each Monte Carlo trial give a valid contribution | supported | GUM S1 95 % interval coverage 93–97 % at N = 10–80, with dependent inputs and (resampled jointly) for vector outputs |
| …for local features no run reaches | **limit** | bump model: coverage of y95 intervals drops when the design misses the bump |

Limits: up to 20 inputs; 2-D steady RANS CFD case (discretisation error
about 6 %, part of the model as defined) with a reference from 800 runs;
adaptive committees only to N = 40. See the journal manuscript, Section 6.

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
# or the whole environment, checks included:
docker build -t fyp-surrogate-gum . && docker run --rm fyp-surrogate-gum
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
bash experiments/run_completion_campaign.sh 4    # research-plan runs (Tiers 1-2)
python experiments/analyse_completion.py         # their statistics
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
