# Number provenance — IMEKO WC 2027 extended abstract

Every quantitative statement in `extended_abstract.tex`, with the file it
comes from and how it is computed. All paths are relative to the repository
root. "Success" means RMSE < 200. Medians are over the 30 seeds (1000–1029)
unless stated. Recompute everything with `python paper/imeko2027/check_numbers.py`,
which prints each value below from the raw records.

## Protocol section

| Text | Value | Source |
|---|---|---|
| output SD at h = 200 | ≈ 4100 (4073) | `results/summary_domain.json` → `phys200.output_std` |
| prior-reverted GP RMSE | ≈ 2400 (median 2445) | `summary_domain.json` → `phys200.median_lhsgp` |

## Domain sweep (Fig. 1, §III.B) — `results/results_domain_*.json`, summary in `results/summary_domain.json`

| Text | Value | Computation |
|---|---|---|
| passive GP, h = 2 | 1.00 SD | median `rmse_lhsgp` / `output_std` (phys2) |
| passive GP, h ≥ 10 | 0.60 SD | same, phys10–phys200 (0.612, 0.600, 0.600, 0.600) |
| committee, h = 2 | 0.99 SD | median `rmse_al` / `output_std` |
| committee, h = 200 | 1.7e-4 SD | same, phys200 |
| ratio of mean RMSEs | 1.04 → 0.030 | `rho_mean`, phys2 and phys200 |
| quadratic share of variance | 44 % at h = 2; ≈ 100 % for h ≥ 10 | Var(0.2‖x‖²)/Var(f) on 2×10⁵ uniform points (`check_numbers.py`) |
| committee median RMSE at h = 2, h ≥ 50 | 0.62–0.68 | median `rmse_al`: 0.615, 0.655, 0.679, 0.679 |
| RMS of oscillatory + exponential terms | 0.50–0.61 | SD of sin2x₁cos2x₂ + exp(−x₃²) on the same points |
| committee median at h = 10 | 1.99 | median `rmse_al` (phys10) |
| PCE-d2 retained at h = 10 | 15/30 | `final_models == ["PCE-d2"]` (phys10) |
| Schwefel ratio | 1.10 | `rho_mean` (schwefel); both arms ≈ 1 SD |

## Ablations (Table I, Fig. 2) — `results/results_{hard,varacq,nopce,gpvar,soft}.json`

| Text | Value | Computation |
|---|---|---|
| UEigV, hard rejection | 0.682, 27/30 | median `rmse`, success count (`hard`) |
| Variance, hard rejection | 0.660, 30/30, p = 0.047 | `varacq`; Wilcoxon vs `hard` on log RMSE |
| variance lower in 18/30 | 18 | paired count |
| no PCE | 2387, 0/30, p < 1e-4 | `nopce` |
| GP alone | 1772, 0/30, p < 1e-4 | `gpvar` |
| Eq. (2) weighting | 0.646, 29/30, p = 0.004 | `soft` |
| collapse, collapse iteration, random share | see §III.C text | `results/results_decompose.json`: `n_survivors == 1`, median `collapse_iter`, Σ`random_queries_after_collapse` / (30 × 15) |

## Aggregation (Table II) — `results/results_hardmin3.json`, `results/results_hard.json`, `results/results_soft.json`

| Text | Value | Computation |
|---|---|---|
| Hard, K_min = 1 | 0.682, 27/30 | `results_hard.json` |
| Hard, K_min = 3, uniform | 1556, 0/30 | median `rmse_uniform` |
| Hard, K_min = 3, Eq. (2) | 0.701, 27/30 | median `rmse_cvweighted` |
| Weighting only | 0.646, 29/30 | `results_soft.json` |
| PCE-d2 survives | 27/30 | `"PCE-d2" in survivor_names` |
| weighted succeeds in exactly those 27 | yes | the three failures (seeds 1006, 1012, 1014) lack PCE-d2 |
| paired p | < 1e-8 (5.6e-9) | Wilcoxon `rmse_uniform` vs `rmse_cvweighted` |

## Propagation (Table III, Fig. 3)

Table III: medians over successful seeds of `results/results_soft.json` (29)
and `results/results_hard.json` (27): `q_relerr`, |`exc_central` − `exc_true`|,
`exc_up` − `exc_lo`, means of `mean_covered`, `exc_covered`, `pbox_coverage`.

Representative run: `results/representative_run_seed1000.json`
(`experiments/make_paper_figures.py pbox`; seed 1000 is the first seed, not selected).

| Text | Value | Field |
|---|---|---|
| mean | 0.21 % | `mean.rel_err` |
| SD | 0.14 % | `std.rel_err` |
| 5th, 50th, 95th percentiles | 0.66, 0.24, 0.41 % | `quantiles.*.rel_err` |
| abstract "within 0.7 %" | max of the above | |
| model-form share | 0.50 → 7e-8 | `trajectory[0]`, `trajectory[-1]` `.model_form_fraction` |
| actual surrogate error share | ≈ 2e-8 | `rmse`² / `std.true`² |

## Values carried over from earlier records (not rerun)

The ablation arms in Table I and rows 1 and 4 of Table II come from the
committed PyTorch campaigns, produced before the Latin-hypercube seeding fix
(`docs/REPRODUCIBILITY.md` §3). They are reported as recorded.
