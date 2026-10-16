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
| `figures/fig_*.pdf` | `python experiments/make_paper_figures.py` | paper figures, regenerated from the JSON records |

Runtime: about 10–30 s per trial per arm on one core.

## 5. Re-running everything

```bash
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
file. Delete a file to rerun it from scratch.

## 6. Number provenance

Every number in the IMEKO extended abstract is traced to its source file
in `paper/imeko2027/NUMBERS.md`.
