# Are the test models and their ranges realistic?

A check, made on 2026-10-21, of every test model of the journal study
(`src/benchmarks.py`, `src/cfd_benchmark.py`) and of the study settings.
The formulas and input models agree with the independent implementation in
UQTestFuns (`experiments/check_benchmarks.py`); this note is about whether
the inputs and ranges are physically meaningful. The papers state the
caveats below (journal, Section 3; extended abstract, Section III).

## Verdict per model

| Model | Inputs and ranges | Verdict | Notes |
|---|---|---|---|
| Bowl (the FYP's "physics" benchmark) | 3 dimensionless inputs on [−200, 200]³; f = sin 2x₁ cos 2x₂ + exp(−x₃²) + 0.2‖x‖² | **synthetic by design** | No units. The domain is the extreme large-domain case the original FYP chose to provoke the GP failure it reported; there the quadratic term carries all but about 1e−8 of the variance (output 0.6 to 24 000) and the sine terms repeat about 127 times across each axis. Kept so that the earlier claim is re-examined on its own terms. |
| Plume (uniform inputs) | Q 100–2000 g/s ammonia, u 1–6 m/s, release height 1–20 m, x 200–1000 m downwind, y ±30 m crosswind; Briggs rural coefficients, neutral stability | **physically plausible, simplified** | Release rates of 0.1–2 kg/s are those of accidental releases; distances are within the range of the Briggs formulas. Simplifications: neutral stability for all wind speeds, and a passive gas, whereas pressurised liquid ammonia can form a dense cloud near the source. The IDLH threshold, 300 ppm = 210 mg/m³, is that of NIOSH. |
| Plume (GUM inputs) | Q lognormal (median 500 g/s, GSD 2; 90 % between 160 and 1560 g/s); u Weibull (shape 2, scale 4 m/s, mean 3.5 m/s) truncated to [0.5, 15] m/s; y normal (SD 15 m) | **physically plausible, simplified** | A shape-2 Weibull is the usual model of a wind-speed climate. 4.6 % of the wind speeds lie below 1 m/s, where Gaussian plume models become unreliable (stated in the journal). |
| Borehole (uniform and Harper–Gupta inputs) | well radius 0.05–0.15 m, radius of influence 100–50 000 m, transmissivities, heads, length, conductivity of the published benchmark | **realistic (published engineering model)** | Water flow through a borehole between two aquifers (Harper and Gupta 1983); SI-type units, published ranges and distributions. |
| Borehole, dependent inputs | as Borehole (GUM) with correlation 0.5 between the transmissivities and between the heads | **plausible assumption** | The value 0.5 is illustrative, not taken from site data; it tests the method, not a site. |
| OTL circuit | resistances 50–150, 25–70, 0.5–3, 1.2–2.5, 0.25–1.2 kΩ; current gain 50–300 | **realistic** | Midpoint voltage of an output-transformerless push-pull circuit (Ben-Ari and Steinberg 2007); ordinary component values. |
| OTL circuit, 20 inputs | as OTL plus 14 inputs on [100, 200] that do not affect the output | **realistic model, synthetic padding** | The inactive inputs are a standard device to test screening in higher dimension (Moon 2010). |
| Piston | mass 30–60 kg, area 0.005–0.020 m², volume 0.002–0.010 m³, spring 1000–5000 N/m, pressure 90–110 kPa, ambient 290–296 K, gas 340–360 K | **realistic** | Cycle time of a piston (Kenett and Zacks; Ben-Ari and Steinberg 2007); physical SI values. |
| Wing weight | wing area 150–200 ft², fuel weight 220–300 lb, aspect ratio 6–10, sweep ±10°, dynamic pressure 16–45 lb/ft², taper 0.5–1, thickness 0.08–0.18, load factor 2.5–6, gross weight 1700–2500 lb, paint 0.025–0.08 lb/ft² | **realistic** | Raymer-type weight equation of a light aircraft (Forrester et al. 2008); the ranges bracket a Cessna 172-class aircraft. |
| Flood | river discharge truncated Gumbel (mode 1013 m³/s, scale 558), Strickler coefficient normal (30, 8), river and bank levels, dyke 7–9 m, length 5 km, width 300 m | **realistic (published engineering model)** | Simplified river-flood model of Iooss and Lemaître (2015), with the published input distributions. |
| Ishigami | 3 dimensionless inputs on [−π, π]³ | **synthetic by design** | Standard sensitivity-analysis function, chosen for its strong non-linearity and interactions. |
| Bump (failure mode) | 4 dimensionless inputs on [0, 1]⁴ | **synthetic by design** | Built to have a narrow feature that the runs can miss. |
| CFD dispersion | wind speed at 10 m 1–6 m/s, roughness length 0.01–0.3 m, building height 6–15 m, turbulent Schmidt number 0.5–1.3, receptor 5–120 m downstream, release 1–20 g/s per metre | **physically plausible, idealised** | Wind speeds from light air to a moderate breeze; roughness from open to rough terrain; 2–5 storey buildings; the Schmidt-number range of the review of Tominaga and Stathopoulos (2007). Idealisations: a two-dimensional section (infinitely wide building and line source), so concentrations are those of a line release and the wake is longer than behind a building of finite width; steady RANS with the standard k–ε model. The journal states that it is a benchmark, not a prediction for a site. |

## Study settings

| Setting | Value | Verdict |
|---|---|---|
| Budgets | 10–80 runs | Realistic for an expensive simulator; below the common rule of ten runs per input on purpose. |
| Repetitions | 30 paired seeds | Adequate for paired Wilcoxon tests with Holm adjustment; equivalence is reported where the intervals allow it. |
| Monte Carlo sizes | 3 × 10⁴ trials through each surrogate; reference 2 × 10⁵ trials of the analytical models, 4 × 10⁴ points from 800 CFD runs | Adequate for the comparisons; the worked example uses the adaptive procedure of GUM S1 7.9 (1.15 × 10⁶ trials). |
| Observation noise | SD 0.1 on the bowl model only | Negligible against its output SD of about 4100. |
| GP search ranges | amplitude up to 10³ in standardised units | Binding only on the bowl model, where the gain from estimating the amplitude is understated (journal, Section 5.1). |

## Conclusion

Nothing is unrealistic in a way that changes the conclusions. Nine of the
twelve models have physical inputs with published or physically plausible
ranges (the plume and CFD models with stated simplifications), and a tenth,
the 20-input OTL circuit, is a physical model padded with inactive inputs. Four elements are
synthetic by design (bowl, Ishigami, bump, the inactive OTL inputs), as is
usual in studies of surrogate models: they are chosen for their
mathematical character, and the papers now say so. The one choice that is
extreme rather than merely synthetic, the bowl model's [−200, 200]³ domain,
is inherited from the earlier analysis on purpose, so that its claim can be
tested; the domain-size sweep (journal, Fig. on h = 2 … 200) shows how the
results change on smaller domains.
