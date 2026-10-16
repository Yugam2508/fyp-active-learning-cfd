# Spectral Active Learning + Forward UQ for Expensive Black-Box Simulators

Surrogate-based uncertainty quantification under severe simulation budgets
(N = 20 evaluations), motivated by CFD modelling of hazardous gas dispersion.
Extends an FYP framework (heterogeneous surrogate committee + spectral
disagreement acquisition + dynamic model management) with a forward-UQ stage
and a set of ablations that isolate what actually drives performance.

## Status — read this first

This repository is a **research record, including negative results.**

- **Current deliverable:** IMEKO World Congress 2027 extended abstract,
  `paper/imeko2027/` (official template, ready to submit; see its README).
  Every number in it is traced to a result file in
  `paper/imeko2027/NUMBERS.md`.
- `paper/paper.tex` is the longer journal draft, reframed as an ablation paper.
  Its domain-sweep and aggregation sections predate the 2026-10-16 reruns and
  must be updated from the extended abstract before submission elsewhere.
  `paper/paper.pdf` is an old, superseded build.
- How results were produced, and a seeding bug fixed on 2026-10-16:
  `docs/REPRODUCIBILITY.md`.

What the experiments currently support (30 paired seeds per arm, N = 20 model
runs; physics, Gaussian plume, borehole and Ishigami models; see
`docs/FINDINGS_soft_vs_hard.md`, update 2026-10-17):

| Claim | Status | Evidence |
|---|---|---|
| Passive GP surrogates fail catastrophically on large domains (prior reversion) | **withdrawn — artefact** | caused by the original GP kernel lacking an amplitude hyperparameter; adding it: physics nRMSE 0.591 → 0.033 (30/30 seeds) |
| GP kernel specification (amplitude, ARD) is a first-order choice | supported | up to 18× (physics), 3.4× (borehole) |
| Log-transforming positive variables helps multiplicative models | supported | borehole 8× (0.089 → 0.011), plume 1.4× |
| A committee beats a well-specified GP | **only with a matching member** | physics > 100× (PCE-d2); borehole 2.3× worse; plume, Ishigami tie |
| CV inverse-error weighting is necessary for committees | supported | never worse than uniform; significantly better on 3 of 4 models |
| Adaptive selection of runs helps at N = 20 | **not supported** | single GP never better; committee mixed (better 2, worse 1, tie 1) |
| UEigV (spectral) beats variance acquisition | **refuted** | generalisation study: worse on 3 of 4 models (borehole 1.7×, p = 3e-8), better only on plume (0.90, p = 0.009) |
| Committee member envelopes are calibrated model-form intervals | **not supported** | coverage 0–100 % |
| Forward-UQ propagation reproduces reference Monte Carlo | supported where the surrogate is accurate | Table III of the extended abstract |

Remaining limits: four test models, uniform inputs, N = 20 only; a CFD-based
dispersion case and calibrated (conformal) intervals are the next steps.

## Layout

```
src/                    library modules (importable)
  framework_core.py       committee, UEigV, hard LCS, ActiveLearner
  uq_extension.py         SoftLCS (CV inverse-error weights), UQAnalyzer
  benchmarks.py           physics, Gaussian plume, borehole, Ishigami
experiments/            runnable studies, launch from anywhere
  demo_uq.py              end-to-end pipeline + presentation UQ figures
  make_paper_figures.py   print-ready paper figures from results/*.json
  campaign_domain.py      domain-size sweep + Schwefel control
  campaign_generalisation.py  controlled comparison on 4 models (11 strategies)
  analyse_generalisation.py   medians, paired tests, LaTeX table
  campaign.py             30-trial paired hard vs soft, with UQ coverage
  campaign_ablations.py   varacq | nopce | gpvar
  campaign_hardmin.py     hard LCS with a minimum committee size (fair baseline)
  decompose_soft_vs_hard.py  isolates predictor vs trajectory effects
results/                JSON records, one per trial (resumable)
  reference_sklearn/      runs made with the sklearn-MLP DNN backend
figures/                generated plots
paper/                  imeko2027/ (extended abstract, submission-ready), paper.tex (journal draft)
docs/                   findings, literature map, reproducibility record, reference checks
```

## Setup

```bash
pip install -r requirements.txt
```

`torch` is optional; without it the DNN surrogates fall back to an
architecturally identical scikit-learn MLP. Results differ slightly between
backends, so record which one you used.

## Reproducing

```bash
python experiments/demo_uq.py                       # pipeline + figures
python experiments/campaign.py hard 0 30            # resumable
python experiments/campaign.py soft 0 30
python experiments/campaign_ablations.py gpvar 0 30
python experiments/campaign_ablations.py varacq 0 30
python experiments/campaign_ablations.py nopce 0 30
python experiments/campaign_hardmin.py 3 0 30
python experiments/decompose_soft_vs_hard.py 0 30
for c in phys2 phys10 phys50 phys100 phys200 schwefel; do
  python experiments/campaign_domain.py $c 0 30; done
for b in physics plume borehole ishigami; do
  for a in gp_orig gp_amp gp_ardnoamp gp_ard gp_m52 al_gp pas_orig pas_ard al_var al_ueigv; do
    python experiments/campaign_generalisation.py $b $a 0 30; done; done
for b in plume borehole; do python experiments/campaign_generalisation.py $b gp_log 0 30; done
python experiments/analyse_generalisation.py
python experiments/make_paper_figures.py            # paper figures + summaries
```

All campaigns are resumable: re-running the same command skips completed
seeds. Seeds are 1000 + trial index, paired across arms.

## Paper

`paper/paper.tex` compiles with `pdflatex paper.tex` (run twice for refs).
It needs these figures in `figures/` (or `paper/figures/`):
`fig_ablation.pdf` and `fig_crossbench.pdf` (from the thesis runs),
`fig_uq_pbox.pdf`, `fig_uq_decomp.pdf`, `fig_uq_sobol.pdf`
(`experiments/demo_uq.py`) and `fig_campaign.pdf` (campaign plot).
Every number in the ablation and aggregation tables is recomputable from the
JSON records in `results/`.

## Licence

MIT (see LICENSE).
