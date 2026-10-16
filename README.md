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

What the experiments currently support (physics benchmark unless stated;
30 seeds per arm):

| Claim | Status | Evidence |
|---|---|---|
| A passively sampled GP reverts towards a constant predictor once run spacing is large (median RMSE 0.60 output SD for every half-width h ≥ 10) | supported | `campaign_domain.py`, `figures/fig_domain_sweep.pdf` |
| The committee helps only through a member that represents the dominant response component (PCE-d2 for the quadratic term) | supported | domain sweep: committee error ≈ RMS of the unrepresentable terms; no-PCE ablation 0/30; Schwefel control (no matching member) ratio 1.10 |
| Original thesis framing: failure needs large domain **and** high dynamic range | **revised** | the Schwefel control does not isolate dynamic range: both passive GP and committee sit at ≈ 1 SD there |
| Uniform averaging over a heterogeneous committee is catastrophic in the prior-reversion regime | supported | `campaign_hardmin.py 3`: 0/30 success (median RMSE 1556) though PCE-d2 survives in 27/30 |
| CV inverse-error weighting is the causal ingredient | supported | same survivors and data: 27/30 success, exactly the 27 trials that retain PCE-d2; paired p < 1e-8 |
| UEigV (spectral) beats plain ensemble-variance acquisition | **refuted** | `campaign_ablations.py varacq`: 30/30 vs 27/30 success, variance better (p = 0.047) |
| Soft weighting beats hard LCS rejection *as a manager* | **confounded** | hard collapses to one model in 30/30 trials (median iteration 10) and spends 31 % of queries at random; predictor swap null (p = 0.90); `results/results_decompose.json` |
| Forward-UQ stage recovers ground-truth MC | supported | `results/representative_run_seed1000.json`: mean 0.21 %, SD 0.14 %, quantiles ≤ 0.66 % |
| Member envelopes (p-box, exceedance interval) are calibrated | **not supported** | exceedance coverage 62 %; conformal calibration is future work |

Nothing here generalises yet: every committee result is on **one** synthetic
benchmark whose dominant term matches PCE-d2's hypothesis class. A benchmark
with no matching member is the next step.

## Layout

```
src/                    library modules (importable)
  framework_core.py       committee, UEigV, hard LCS, ActiveLearner
  uq_extension.py         SoftLCS (CV inverse-error weights), UQAnalyzer
experiments/            runnable studies, launch from anywhere
  demo_uq.py              end-to-end pipeline + presentation UQ figures
  make_paper_figures.py   print-ready paper figures from results/*.json
  campaign_domain.py      domain-size sweep + Schwefel control
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
