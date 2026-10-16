# Number provenance — journal manuscript (`paper/paper.tex`)

Every number in the manuscript comes from the per-seed result records in
`results/` through one of three routes, and each route can be re-run:

| Route | What | Regenerate / check |
|---|---|---|
| Generated tables | Tables 2–6 (`results/journal/tab_*.tex`) and Figs. 1–4 (`figures/journal/*.pdf`) | `python experiments/analyse_journal.py` (writes them); `--check` verifies `summary.json` |
| Numbers typed into the prose | over 100 statements in `paper/journal_*.tex` (effect sizes, coverage, CFD facts, abstract) | `python paper/check_numbers_journal.py` recomputes each from `results/journal/summary.json` and the CFD side files and checks that the quoted string is in the text (exit 1 on any mismatch) |
| CFD case facts | reference values and uncertainty, run counts, convergence, grid and iteration checks | `results/cfd_reference.json` (`experiments/cfd_reference_stats.py`), `results/cfd_grid_check.json`, `results/cfd_iteration_check.json` |

Definitions used throughout: medians over the 30 seeds 1000–1029; nRMSE =
test RMSE / reference output SD; a paired ratio is the median over seeds of
(error of A) / (error of B); confidence intervals are 95 % percentile
bootstrap (10 000 resamples, `RandomState(0)`); p-values are two-sided
Wilcoxon signed-rank on log ratios, Holm-adjusted over all 173 tests of the
study (`p_holm` in `summary.json`); "significant" means `p_holm < 0.05`.

Sources of the records:

| Records | Produced by |
|---|---|
| `results/results_gen_<model>_<arm>[_N<N>].json` | `experiments/campaign_generalisation.py` (`run_journal_campaign.sh`, `run_cfd_campaign.sh`) |
| `results/results_cov_<model>_<surrogate>.json` | `experiments/campaign_coverage.py` (`run_coverage_campaign.sh`, `run_cfd_campaign.sh`) |
| `results/results_sel_<model>.json` | `experiments/campaign_selection.py` (`run_selection_campaign.sh`) |
| `results/cfd_runs/*.json` | `src/cfd_benchmark.py` via `experiments/cfd_precompute.py` and the adaptive CFD arms |
| `results/results_domain_*.json` (Fig. 5) | `experiments/campaign_domain.py` (conference study) |

The extended abstract has its own audit: `paper/imeko2027/NUMBERS.md` and
`paper/imeko2027/check_numbers.py`.
