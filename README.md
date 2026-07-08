# Spectral Active Learning + Forward UQ for Expensive Black-Box Simulators

Surrogate-based uncertainty quantification under severe simulation budgets
(N = 20 evaluations), motivated by CFD modelling of hazardous gas dispersion.
Extends an FYP framework (heterogeneous surrogate committee + spectral
disagreement acquisition + dynamic model management) with a forward-UQ stage
and a set of ablations that isolate what actually drives performance.

## Status — read this first

This repository is a **research record, including negative results.**
`paper/paper.pdf` is a **superseded draft**: its headline claims predate the
ablations in `docs/FINDINGS_soft_vs_hard.md` and are no longer supported.

What the experiments currently support:

| Claim | Status | Evidence |
|---|---|---|
| Passive GP surrogates fail catastrophically only when the domain is large **and** the output dynamic range is high | supported | domain ablation + Schwefel control |
| Forward-UQ stage (p-box, exceedance bounds, variance decomposition, Sobol with epistemic bars) recovers ground-truth MC | supported | `experiments/demo_uq.py`, within ~1% on all moments/quantiles |
| Uniform averaging over a heterogeneous committee is catastrophic in the prior-reversion regime | supported | `campaign_hardmin.py`: 0/30 success, median RMSE 1763, despite the correct model being present |
| UEigV (spectral) beats plain ensemble-variance acquisition | **refuted** | `campaign_ablations.py varacq`: variance is equal-or-better (30/30 vs 27/30 success, Wilcoxon p = 0.047) |
| Soft weighting beats hard LCS rejection | **confounded** | hard collapses to K=1 in 30/30 trials and silently degenerates ~31% of its queries to random sampling; predictor swap is null (p = 0.237) |
| The framework's win is the spectral metric | **refuted** | `campaign_ablations.py nopce`: removing PCE collapses performance; the PCE-d2 anchor is the mechanism |

Nothing here generalises yet: every result is on **one** synthetic benchmark
whose dominant term coincides exactly with PCE-d2's hypothesis class.
Multi-benchmark validation is the next step.

## Layout

```
src/                    library modules (importable)
  framework_core.py       committee, UEigV, hard LCS, ActiveLearner
  uq_extension.py         SoftLCS (CV inverse-error weights), UQAnalyzer
experiments/            runnable studies, launch from anywhere
  demo_uq.py              end-to-end pipeline + all UQ figures
  campaign.py             30-trial paired hard vs soft, with UQ coverage
  campaign_ablations.py   varacq | nopce | gpvar
  campaign_hardmin.py     hard LCS with a minimum committee size (fair baseline)
  decompose_soft_vs_hard.py  isolates predictor vs trajectory effects
results/                JSON records, one per trial (resumable)
  reference_sklearn/      runs made with the sklearn-MLP DNN backend
figures/                generated plots
paper/                  LaTeX source (superseded draft — see Status)
docs/                   findings notes and literature map
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
```

All campaigns are resumable: re-running the same command skips completed
seeds. Seeds are 1000 + trial index, paired across arms.

## Paper

`paper/paper.tex` compiles with `pdflatex paper.tex` (run twice for refs).
Being revised to match `docs/FINDINGS_soft_vs_hard.md`.

## Licence

MIT (see LICENSE) — add one if you intend to publish this.
