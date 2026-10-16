# Number provenance — IMEKO WC 2027 extended abstract (and `paper/paper.tex`)

The journal manuscript `paper/paper.tex` extends this study (ten models,
budgets 10–80, coverage validation, CFD case); its numbers are traced by
`paper/check_numbers_journal.py` and `paper/NUMBERS_journal.md`.

Every quantitative statement in `extended_abstract.tex`, where it comes from,
and how it is computed. Run

```bash
python paper/imeko2027/check_numbers.py
```

to recompute all of them from the raw records (it uses
`experiments/analyse_generalisation.py`). Medians are over the 30 seeds
1000–1029. Normalised RMSE (nRMSE) = test RMSE / reference output SD.
Paired tests: two-sided Wilcoxon signed-rank on log nRMSE; the paper quotes
Holm-adjusted p-values over the 36 tests (`p_holm` in the summary).

## Sources

| Records | Produced by |
|---|---|
| `results/results_gen_<model>_<arm>.json` | `experiments/campaign_generalisation.py <model> <arm> 0 30` |
| `results/summary_generalisation.json`, `results/table_propagation.tex` | `experiments/analyse_generalisation.py` |
| `results/results_domain_<config>.json` (fields `rmse_lhsgp`, `rmse_lhsgp_m52`) | `experiments/campaign_domain.py <config> 0 30 [--add-m52]` |
| `figures/fig_strategies.pdf` | `experiments/make_paper_figures.py strategies` |
| `results/results_err_<model>_<gp_m52|gp_log>.json` | `experiments/campaign_error_estimation.py <model> <arm> 0 30` |

## Statement → value → source

| Section | Statement | Value | Source / computation |
|---|---|---|---|
| Abstract, III.A | "up to eighteen-fold" (kernel specification) | 0.591 → 0.033 on physics (17.9× by medians; 18× per-seed median ratio) | `gp_orig`, `gp_amp`, physics |
| Abstract, III.A | "up to eightfold" (log transformation) | borehole 0.089 → 0.011 (8.0× by medians) | `gp_m52`, `gp_log`, borehole |
| II.A | plume spans three orders of magnitude; 12 % above 210 mg m⁻³ | 1.8–2448 mg m⁻³; 12.3 % | `benchmarks.Plume` on the 2×10⁵ reference sample |
| III.A, Table II | all single-GP medians | see table | `gp_*` arms |
| III.A | amplitude alone: all 30 seeds improve, p < 1e-8 | 30/30, p = 1.9e-9 | test `gp_amp vs gp_orig`, physics |
| III.A | original kernel 0.60 for h = 10…200; Matérn 0.17 at 10, 0.08 for 50…200 | 0.612/0.600/0.600/0.600; 0.173/0.078/0.078/0.080 | `results_domain_phys*.json` |
| III.A | ARD on borehole 0.30 → 0.09 | 0.295 → 0.089 | `gp_orig`, `gp_m52` |
| III.A | log: 30 and 27 of 30 seeds improve | borehole 30/30, plume 27/30 | test `gp_log vs gp_m52` |
| III.A | ten runs per input → 30 runs for Ishigami | d = 3 | Loeppky et al. (2009) |
| III.B | committee beats best single GP only on physics, by > 2 orders | 180× vs `gp_amp`; 440× vs `gp_m52` | medians |
| III.B | borehole committee 2.3× worse, 0/30 better | ratio 2.33, 0/30 | test `pas_ard:cv vs gp_m52` |
| III.B | plume, Ishigami tie | p = 0.21, 1.0 | same test |
| III.B | uniform worse than CV: physics (~2000×), plume, borehole (p ≤ 0.023), never better | ratios 4.7e-4, 0.98, 0.84; Ishigami p = 0.18 | test `pas_ard:cv vs pas_ard:uni` |
| III.B | adaptive GP never better; worse on plume p = 0.008 | p = 0.21, 0.0081, 0.44, 0.21 | test `al_gp vs gp_m52` |
| III.B | adaptive committee: 0.80, 0.61, 1.25, n.s. | as stated | test `al_var:cv vs pas_ard:cv` |
| III.B | spectral worse on three models (borehole 1.7×, p < 1e-7), better only on plume (0.90, p = 0.009) | physics 1.02 p=0.021; plume 0.90 p=0.0087; borehole 1.67 p=2.6e-8; Ishigami 1.07 p=0.0043 | test `al_ueigv:cv vs al_var:cv` |
| Table III | all entries | generated | `results/table_propagation.tex` |
| III.B | uniform worse on physics and borehole (adjusted p ≤ 0.002); plume not significant after adjustment | holm 6.7e-8, 0.0016; plume 0.28 | test `pas_ard:cv vs pas_ard:uni` |
| III.B | adaptive committee hurt plume (adjusted p = 0.005); adaptive GP never significantly better | holm 0.0046; GP min holm 0.12 | tests `al_var:cv vs pas_ard:cv`, `al_gp vs gp_m52` |
| III.B | UEigV never better after adjustment; borehole 1.7× worse (adjusted p < 1e-6) | holm 7e-7; plume 0.12 | test `al_ueigv:cv vs al_var:cv` |
| III.C | fixed threshold: reference P(C > IDLH) = 0.123; median abs. error 0.009 (log GP), 0.023 (raw GP) | as stated | `results_err_plume_*.json` (`p_idlh`) |
| Abstract, III.D, Table IV | coverage (%) of the nominal 95 % intervals: pointwise GP / jackknife+; GUM S1 interval exact / + LOO error; y95 interval GP posterior / jackknife | as tabulated; GP 71 % on plume and Ishigami, J+ 97 % on all four, exact 86–95 %, +LOO 95–97 %, y95 J 83–100 %, GP 73–97 % | `results_cov_<model>_<gp_m52|gp_log>.json` (`experiments/campaign_coverage.py`) |
| Abstract, III.E | CFD: log GP nRMSE 0.080 vs Matérn 0.53, better in 30/30 seeds; "sixfold" = 1 / median paired ratio (0.165) | as stated | `results_gen_cfd_{gp_log,gp_m52}.json` |
| III.E | CFD: y95 within 6.9 % (log GP), 18.5 % (Matérn), 27 % (committee); P(C > IDLH) within 0.009; reference uncertainty 3.3 % and 0.007 from 800 runs | as stated | `results_gen_cfd_*.json`, `results/cfd_reference.json` |
| III.C | best-strategy errors: y95 0.41 / 6.9 / 0.60 / 16 %; P within 0.0015 / 0.0076 / 0.0018 / 0.029 | as stated | Table III (bold entries) |
| III.C | adaptive committee plume mean error 55 % | 55.1 % | `al_var:cv` plume `relerr_mean` |
| III.C | envelope coverage 0–100 % | physics `pas_ard:cv` 0 %, `al_var:cv` 100 % | `exc_covered` |
| III.C, Fig. 3 | P(C > 210 mg m⁻³): reference 0.123, GP log 0.119, GP raw 0.158, committee 0.142 | as stated | `results/plume_cdf_seed1000.json` (`make_paper_figures.py plume`) |
| III.A, Fig. 1 | domain sweep curves | as plotted | `results_domain_*.json` (`make_paper_figures.py domain`) |

## Carried over from earlier work (not used for claims here)

The physics-only campaigns of the earlier analysis (`results/results_{hard,
soft,varacq,nopce,gpvar,hardmin3,decompose}.json`) used committees with the
original GP kernel. They are documented in `docs/FINDINGS_soft_vs_hard.md`
and are superseded for the purposes of this abstract by the controlled
comparison above.
