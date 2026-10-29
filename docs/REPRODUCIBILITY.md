# Reproducibility record

How every result in this repository was produced, what environment produced
it, and known caveats. Read this before quoting a number.

## 1. Environment

| Item | Committed campaigns (`results/results_{hard,soft,varacq,nopce,gpvar}.json`) | Runs added 2026-10-16 (see §3) |
|---|---|---|
| Machine | author's workstation | cloud container, 4 CPU cores |
| DNN backend | PyTorch (author's run) | PyTorch 2.14 (CPU) |
| SciPy | not recorded | 1.17.1 |
| scikit-learn | not recorded | 1.9.1 |
| NumPy | not recorded | see `pip freeze` at run time |

`results/reference_sklearn/` holds earlier runs made with the scikit-learn
MLP fallback for the DNN members (architecturally identical; see `README.md`).

## 2. Protocol (all campaigns)

- Physics benchmark f(x) = sin 2x₁ cos 2x₂ + exp(−x₃²) + 0.2‖x‖² on
  [−200, 200]³ unless stated; observation noise σ = 0.1.
- Budget N = 20: 5-point Latin hypercube + 15 sequential queries.
- Seeds 1000–1029 (trial index t ↦ seed 1000 + t), shared by all arms.
- RMSE on one fixed 5000-point uniform test set, `RandomState(7)`.
- Reference Monte Carlo for UQ checks: 200 000 points, `RandomState(8)`;
  propagation sample: 30 000 points, `RandomState(9)`.
- Success: RMSE < 200 (prior-reverted GP gives ≈ 2000; the two populations
  are separated by more than two orders of magnitude, see
  `figures/fig_ablation_strip.pdf`).
- Paired comparisons: two-sided Wilcoxon signed-rank test on log RMSE.

## 3. Known issue fixed on 2026-10-16: Latin-hypercube seeding

`src/framework_core.lhs()` created `qmc.LatinHypercube(d=d)` without a seed.
In SciPy ≥ 1.15 an unseeded QMC engine draws its seed from OS entropy, so
`np.random.seed(seed)` no longer fixed the initial design. Consequences:

1. Runs were not bit-reproducible, and two arms with the same seed did not
   share an initial design (which the paired protocol assumes).
2. Under older SciPy the engine fell back to NumPy's global state, which is
   presumably how the committed campaigns were produced. The SciPy version
   used for them was not recorded, so their design pairing cannot be checked
   after the fact. The paired Wilcoxon test stays valid if designs were not
   shared (pairs are then independent), but it loses power.

**Fix:** `lhs()` now draws an integer seed from the global NumPy RNG and
passes it to the engine (`rng=` on SciPy ≥ 1.15, `seed=` before). All runs
made after the fix are bit-reproducible; this was checked by re-running
`make_paper_figures.py pbox` twice and comparing the JSON records
(identical).

Draws differ from the pre-fix code, so new runs do not replicate committed
per-seed values exactly, only statistically.

## 4. Runs added on 2026-10-16 (post-fix, PyTorch backend)

| Output | Command | Purpose |
|---|---|---|
| `results/results_domain_{phys2,phys10,phys50,phys100,phys200,schwefel}.json`, `results/summary_domain.json` | `python experiments/campaign_domain.py <config> 0 30` | domain-size sweep and Schwefel control; replaces thesis numbers that had no record in this repo |
| `results/results_hardmin3.json` | `python experiments/campaign_hardmin.py 3 0 30` | fair hard-rejection baseline; uniform and CV-weighted aggregation scored on the same survivors and data |
| `results/results_decompose.json` | `python experiments/decompose_soft_vs_hard.py 0 30` | collapse of hard rejection and predictor swap, PyTorch backend (the sklearn record stays in `results/reference_sklearn/`) |
| `results/representative_run_seed1000.json` | `python experiments/make_paper_figures.py pbox` | the "representative run" quoted in the paper (seed 1000 = first campaign seed, not selected) |
| `results/results_gen_<model>_<arm>.json`, `results/summary_generalisation.json`, `results/table_propagation.tex` (2026-10-17) | `experiments/campaign_generalisation.py`, `experiments/analyse_generalisation.py` | controlled comparison: 4 models, 11 strategies, 30 seeds |
| `rmse_lhsgp_m52` in `results/results_domain_*.json` (2026-10-17) | `experiments/campaign_domain.py <config> 0 30 --add-m52` | well-specified GP in the domain sweep |
| `figures/fig_*.pdf` | `python experiments/make_paper_figures.py` | paper figures, regenerated from the JSON records |

Runtime: about 10–30 s per trial per arm on one core.

## 4b. GP kernel note (2026-10-17)

`GPSurrogate(spec="orig")`, the default and the kernel of every pre-2026-10-17
campaign, has no amplitude hyperparameter. `spec="ard"` adds amplitude and ARD.
The default was left unchanged (verified bit-identical on the representative
run) so all earlier records remain reproducible.

## 5. Re-running everything

```bash
python experiments/results_pack.py      # unpack results/*.zip first
pip install -r requirements.txt
python experiments/campaign.py hard 0 30
python experiments/campaign.py soft 0 30
python experiments/campaign_ablations.py varacq 0 30
python experiments/campaign_ablations.py nopce 0 30
python experiments/campaign_ablations.py gpvar 0 30
python experiments/campaign_hardmin.py 3 0 30
python experiments/decompose_soft_vs_hard.py 0 30
for c in phys2 phys10 phys50 phys100 phys200 schwefel; do
  python experiments/campaign_domain.py $c 0 30; done
python experiments/make_paper_figures.py
```

Campaigns are resumable: they skip seeds already present in their JSON
file. Delete a file to rerun it from scratch, and run `python
experiments/results_pack.py pack` afterwards: git tracks the ZIP files,
not the single records (section 9).

## 6. Number provenance

Every number in the IMEKO extended abstract is traced to its source file
in `paper/imeko2027/NUMBERS.md`.

## 7. Journal study (2026-10-16 onwards)

Environment: cloud container, Python 3.11.15, 4 CPU cores, single-threaded
BLAS (`OMP_NUM_THREADS=1`), exact package versions in `requirements-lock.txt`;
OpenFOAM v1912 from the Ubuntu package `openfoam` (1912.200626-2build3) for
the CFD case.

| Step | Command | Output |
|---|---|---|
| benchmark formulas vs UQTestFuns | `python experiments/check_benchmarks.py` | console |
| published records still reproduced | `python experiments/test_reproducibility.py` | console |
| all analytical-model jobs (new arms, budgets 10–80, `--augment` of the N = 20 records) | `bash experiments/run_journal_campaign.sh 4` | `results/results_gen_*.json` |
| coverage validation | `bash experiments/run_coverage_campaign.sh 4` | `results/results_cov_*.json` |
| CFD runs for designs and reference | `CFD_WORKERS=4 python experiments/cfd_precompute.py` | `results/cfd_runs/*.json` |
| CFD grid sensitivity | `python experiments/cfd_grid_check.py` | `results/cfd_grid_check.json` |
| all CFD jobs | `bash experiments/run_cfd_campaign.sh 4` | `results/results_{gen,cov}_cfd_*.json` |
| statistics, tables, figures | `python experiments/analyse_journal.py` | `results/journal/`, `figures/journal/` |
| check that summary.json is reproduced | `python experiments/analyse_journal.py --check` | exit status |

Notes.

- `--augment` regenerates every published N = 20 record from its seed,
  checks that all stored fields are reproduced to a relative tolerance of
  1e-9 and only then adds the new fields (`thr_*`, exceedance at the model's
  physical threshold). Mismatches would be reported in `results/logs/` and
  leave the record untouched.
- The CFD reference is 800 CFD runs × 50 draws of (x_r, Q), not 2e5 runs;
  see `src/cfd_benchmark.py`. Every CFD run is cached as one JSON file
  keyed by its four flow inputs, so rerunning a campaign reuses them; delete
  `results/cfd_runs/` to recompute from scratch (about 20 s per run on one
  core). Each record stores the iteration count and whether the residual
  criteria were met within 4000 iterations.
- The OpenFOAM function objects (the usual way to add a passive scalar)
  abort with an `IOstream "sha1"` error in this Ubuntu build of v1912; the
  scalar is therefore solved as the temperature of
  `buoyantBoussinesqSimpleFoam` with zero gravity and zero expansion
  coefficient, which is the same transport equation (diffusivity
  nu/Pr + nu_t/Pr_t with Pr = Sc, Pr_t = Sc_t).
- The CFD cache of the main study ends with 2900 runs: 1550 from
  `cfd_precompute.py` (30 x 25 design runs and 800 reference flow states)
  and 1350 added by the three adaptive CFD arms (30 seeds x 15 sequential
  runs each). The research-plan runs (section 8: budgets 10 and 40, maximin
  and Sobol' designs, the second reference, the adaptive arms at N = 40 and
  the IMSE design) bring it to 8170 runs. The adaptive arms
  were run as nine blocks of 10 seeds in parallel (`RESULTS_DIR=results/parts/...`)
  and merged with `experiments/merge_parts.py`, as `run_cfd_campaign.sh` does.
- `paper/check_numbers_journal.py` recomputes every number typed into the
  journal prose (and `analyse_completion.py --check` verifies that
  `results/journal/completion.json` is reproduced from the records);
  `.github/workflows/ci.yml` runs both together with
  `check_benchmarks.py`, `test_reproducibility.py --quick`, the extended
  abstract's `check_numbers.py` and `analyse_journal.py --check`.
- Cross-machine reproducibility (checked by the CI on GitHub's runners,
  same pinned packages, different processor): the regenerated records
  agree with the published ones to a relative deviation of at most
  8.7e-6 (physics/gp_amp nRMSE); committees 2.8e-6, all other cases
  below 2e-8. On the reference environment the deviation is exactly zero.
  The difference comes from processor-dependent BLAS and PyTorch kernels,
  amplified by the hyperparameter optimisation; it is five orders of
  magnitude below any effect reported in the papers.

## 8. Extension and research-plan runs (2026-10-19 onwards)

Same environment as §7, plus `emcee` 3.1.6 (MCMC over GP hyperparameters)
and OpenTURNS 1.27 (independent cross-check), both pinned in
`requirements-lock.txt`. The plan these runs carry out, and the decision
rules fixed before they started, are in `docs/RESEARCH_PLAN.md`.

| Step | Command | Output |
|---|---|---|
| wing weight, OTL with 20 inputs, AK-MCS at the reference 95th percentile, coverage of every candidate on the new models | `bash experiments/run_extension_campaign.sh 4` | `results/results_{gen,cov,sel}_*.json` |
| worked example (procedure on the CFD case) | `python experiments/worked_example_cfd.py` | `results/worked_example_cfd.json`, `results/journal/tab_worked.tex` |
| all Tier 1 and 2 runs of the research plan, in four stages | `bash experiments/run_completion_campaign.sh 4 [first_stage]` | see below |
| statistics of those runs | `python experiments/analyse_completion.py` (`--check`) | `results/journal/completion.json`, `tab_bayesgp.tex`, `tab_budgets.tex` |

Outputs of `run_completion_campaign.sh`:

| Plan item | Script | Records |
|---|---|---|
| T1.1 coverage, selection, the procedure at N = 10, 40, 80 | `campaign_coverage.py`, `campaign_selection.py`, `campaign_generalisation.py` | `results_{cov,sel,gen}_<model>_*_N<N>.json` |
| T1.2 Student-t, MCMC and LOO-recalibrated GP intervals | `campaign_coverage.py` (`--augment` adds them to the N = 20 records after checking that every stored field is reproduced) | fields `*_gpt_*`, `*_gpmc_*`, `*_gpcal_*` |
| T1.3 OpenTURNS PCE, GP and PC-Kriging | `crosscheck_openturns.py` | `results_ot_<model>.json` |
| T1.6 failure mode (model `bump`) | `campaign_generalisation.py`, `campaign_coverage.py`, `campaign_selection.py` | `results_*_bump_*.json` |
| T2.1 maximin LHS and Sobol' designs | `--design=maximin|sobol` | `*_dmaximin.json`, `*_dsobol.json` (records carry a `design` field) |
| T2.2 IMSE sequential design | arm `al_imse` | `results_gen_<model>_al_imse*.json` |
| T2.3 stacking of the candidates | `campaign_stacking.py` | `results_stack_<model>.json` |
| T2.4 vector output (GUM S2) | `campaign_vector.py` | `results_vector_plume.json` |
| T2.5 dependent inputs (model `borehole_corr`) | as for the other models | `results_*_borehole_corr_*.json` |
| T2.6 grid convergence index; second CFD reference | `cfd_gci.py` (iteration check of its coarse mesh: `cfd_gci_iterations.py`); `cfd_precompute.py --ref2` then `cfd_reference2.py` | `results/cfd_gci.json`, `results/cfd_gci_iterations.json`, `results/cfd_reference2.json` |

Notes.

- The campaign ran from 2026-10-19 07:37 to 2026-10-20 15:36 UTC
  (`results/logs/completion_master.log`). The cloud container was restarted
  several times; each time `experiments/resume_completion.sh` resumed the
  unfinished stage, and finished seeds were skipped. The one script that
  was not resumable, `cfd_gci.py`, re-ran once and reproduced its earlier
  output exactly; it now skips when its output is complete.
- The new GP intervals use their own random stream, RandomState(seed + 2e6),
  so the published fields of the coverage records do not change.
- At N = 10 a 95 % jackknife+ interval is unbounded (its upper rank
  ⌈0.95 (N + 1)⌉ exceeds N); the records then hold infinite bounds, and
  `analyse_completion.py` reports the share of such cases.
- `src/cfd_dispersion.py` accepts a non-integer refinement factor (cell
  counts are rounded) for the coarse mesh of the grid convergence index; the
  standard and fine meshes are unchanged.
- `src/surrogates.py` reads `GP_AMP_MAX` (default 1e3, the bound used for
  every record of the study); only `experiments/sensitivity_amp_bound.py`
  sets it, to 1e6, for the sensitivity check described in
  `docs/RESEARCH_PLAN.md` (results in `results/sensitivity_amp_bound.json`).

## 9. Long-term reproducibility

- `Dockerfile` builds the environment of section 1: Ubuntu 24.04,
  OpenFOAM v1912 from the Ubuntu archive (package 1912.200626-2build3),
  Python 3.11 from the deadsnakes PPA (the study used 3.11.15-1+noble1) and
  `requirements-lock.txt` with the CPU build of PyTorch, plus the LaTeX
  packages for the papers. Its default command runs the CI checks:
  `docker build -t fyp-surrogate-gum . && docker run --rm fyp-surrogate-gum`.
  The workflow `.github/workflows/docker.yml` builds the image and runs
  these checks whenever the Dockerfile or the lockfile changes, and then
  `experiments/cfd_smoke.py`, which reruns one stored CFD case with the
  image's OpenFOAM (relative tolerance 1e-3). Both passed on the first
  build (2026-10-20).
- The release cited in the papers, v1.1, is archived at Zenodo
  (DOI 10.5281/zenodo.23722532; v1.0: DOI 10.5281/zenodo.23661235) with the cached CFD
  runs and every result record, so the analyses can be repeated without
  OpenFOAM even if the package stops being installable.
- From v1.1 the 8170 cached CFD runs and the 926 per-seed records are
  kept in two ZIP files, `results/cfd_runs.zip` and `results/records.zip`,
  whose members are byte-identical to the single files of v1.0;
  `python experiments/results_pack.py` unpacks them (and never overwrites
  a file that differs), `pack` rebuilds them reproducibly and `check`
  compares them with the files. The source archive of a release then has
  about 150 files instead of 9246, below the 1000 entries that Zenodo's
  preview lists (`check` tests this on `git archive` of HEAD; CI runs it).
  The two sklearn reference records `results_hard.json` and
  `results_soft.json` in `results/reference_sklearn/` carry new names
  from v1.1; their content is unchanged.
