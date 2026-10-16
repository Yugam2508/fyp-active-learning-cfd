# Reference verification log

For the IMEKO WC 2027 extended abstract (`paper/imeko2027/references.bib`),
checked on 2026-10-16/17. IMEKO requires every author to be listed unless
there are six or more, and titles in sentence case; the template also
encourages citing measurement journals (Metrologia, Measurement, …).

Each entry was checked against a publisher page, DOI record or bibliographic
index (Project Euclid, NASA ADS, IDEAS/RePEc, DBLP, NPL/EURAMET repositories,
arXiv, MLAnthology). A DOI is included only where that check confirmed it.

| Key | Reference | Checked against | DOI in .bib |
|---|---|---|---|
| jcgm100 | JCGM 100:2008 (GUM) | standard | — |
| jcgm101 | JCGM 101:2008 (GUM Supplement 1) | standard | — |
| cox2006 | Cox, Siebert, *Metrologia* 43(4) S178–S188 (2006) | NPL repository | yes |
| harris2014 | Harris, Cox, *Metrologia* 51(4) S176–S182 (2014) | NPL repository | yes |
| tan2017 | Tan, Du, Liu, Su, Liu, *J. Loss Prev. Process Ind.* 47, 129–139 (2017) | ScienceDirect, ADS `2017JLPPI..47..129T` | yes |
| heidenreich2017 | Heidenreich, Gross, Bär, Wright, *Measurement* 97, 79–87 (2017) | EURAMET and NPL repositories | yes |
| wang2022 | Wang, Qiang, Xu, Wu, *Symmetry* 14(6) 1219 (2022) | MDPI | yes |
| ohagan2006 | O'Hagan, *Reliab. Eng. Syst. Saf.* 91(10) 1290–1300 (2006) | IDEAS/RePEc | yes |
| kennedy2001 | Kennedy, O'Hagan, *J. R. Stat. Soc. B* 63(3) 425–464 (2001) | IDEAS/RePEc, Lancaster eprints | yes |
| sacks1989 | Sacks, Welch, Mitchell, Wynn, *Stat. Sci.* 4(4) 409–423 (1989) | Project Euclid | yes |
| loeppky2009 | Loeppky, Sacks, Welch, *Technometrics* 51(4) 366–376 (2009) | UBC statistics, OSU journal-club copy | yes |
| chen2016 | Chen, Loeppky, Sacks, Welch, *Stat. Sci.* 31(1) 40–60 (2016) | UBC statistics (author PDF), Project Euclid DOI | yes |
| rw2006 | Rasmussen, Williams, *Gaussian Processes for Machine Learning*, MIT Press (2006) | standard | — |
| gramacy2009 | Gramacy, Lee, *Technometrics* 51(2) 130–145 (2009) | DBLP, arXiv 0805.4359 | no (not confirmed) |
| xiu2002 | Xiu, Karniadakis, *SIAM J. Sci. Comput.* 24(2) 619–644 (2002) | author PDF (Utah SCI) | yes |
| sudret2008 | Sudret, *Reliab. Eng. Syst. Saf.* 93(7) 964–979 (2008) | bibliographic listings | no (not confirmed) |
| huellermeier2021 | Hüllermeier, Waegeman, *Mach. Learn.* 110(3) 457–506 (2021) | LMU repository, MLAnthology | yes |
| goel2007 | Goel, Haftka, Shyy, Queipo, *Struct. Multidiscip. Optim.* 33(3) 199–216 (2007) | DOI landing page | yes |
| viana2009 | Viana, Haftka, Steffen, *Struct. Multidiscip. Optim.* 39(4) 439–457 (2009) | citing literature | no (not confirmed) |
| lin2024 | Lin, Trivedi, Sun, *Trans. Mach. Learn. Res.* (2024) | arXiv 2305.19187, MLAnthology | arXiv id |
| ferson1996 | Ferson, Ginzburg, *Reliab. Eng. Syst. Saf.* 54, 133–144 (1996) | citing literature with DOI | yes |
| angelopoulos2023 | Angelopoulos, Bates, *Found. Trends Mach. Learn.* 16(4) 494–591 (2023) | now publishers, MLAnthology | yes |
| repo | this repository | — | URL |

## Corrections made relative to earlier drafts

- **heidenreich2017** was cited as *Metrologia* 54 (2017) 553–562, which is
  wrong. Corrected here and in `paper/paper.tex`.
- **tan2017**, **wang2022**: "et al." replaced by full author lists.
- Removed: Saltelli 2010 and Jansen 1999 (the Sobol sentence they supported was
  dropped, since no Sobol result is shown).
- Added (2026-10-17): Cox & Siebert 2006 and Harris & Cox 2014 (Monte Carlo
  method in metrology), O'Hagan 2006 and Kennedy & O'Hagan 2001 (emulators,
  model inadequacy), Gramacy & Lee 2009 (adaptive design of computer
  experiments), Xiu & Karniadakis 2002 and Sudret 2008 (polynomial chaos).
- Added (2026-10-17, revised framing): Loeppky et al. 2009 (ten runs per input
  guideline) and Chen et al. 2016 (assessment of GP analysis choices).
- Considered but not added because the record could not be confirmed:
  Seung, Opper, Sompolinsky, "Query by committee" (COLT 1992);
  Wübbeler, Krystek, Elster (Meas. Sci. Technol. 2008).

## Journal version (`paper/journal.bib`), checked 2026-10-16

Entries added for the journal study (new baselines, benchmarks and the
coverage validation). Direct DOI resolvers (Crossref, doi.org) are blocked by
the network policy of the cloud environment used for this check, so each
record was confirmed through publisher/indexing pages returned by search, or
through the reference lists bundled in UQTestFuns v0.7.0 (whose input models
the benchmarks are cross-checked against).

| Key | Reference | Checked against | DOI in .bib |
|---|---|---|---|
| barber2021 | Barber, Candès, Ramdas, Tibshirani, *Ann. Stat.* 49(1) 486–507 (2021) | Project Euclid, arXiv 1905.02928 | yes |
| echard2011 | Echard, Gayton, Lemaire, *Struct. Saf.* 33(2) 145–154 (2011) | Structurae, UQTestFuns (four_branch, undamped_oscillator) | yes |
| blatman2011 | Blatman, Sudret, *J. Comput. Phys.* 230(6) 2345–2367 (2011) | citing literature (Lüthen et al., arXiv 2009.04800; Sudret group pages) | yes |
| efron2004 | Efron, Hastie, Johnstone, Tibshirani, *Ann. Stat.* 32(2) 407–451 (2004) | Project Euclid, arXiv math/0406456 (407–499 includes the discussion) | yes |
| gneiting2007 | Gneiting, Raftery, *J. Am. Stat. Assoc.* 102(477) 359–378 (2007) | DTIC technical report, citing literature | yes |
| benari2007 | Ben-Ari, Steinberg, *Qual. Eng.* 19, 327–338 (2007) | UQTestFuns (otl_circuit, piston) | yes |
| morris1993 | Morris, Mitchell, Ylvisaker, *Technometrics* 35(3) 243–255 (1993) | UQTestFuns (borehole), arXiv 0802.4381 reference list | yes |
| harper1983 | Harper, Gupta, report BMI/ONWI-516, Battelle (1983) | SFU Virtual Library, UQTestFuns (borehole) | — |
| iooss2015 | Iooss, Lemaître, in Dellino, Meloni (eds.), Springer (2015) 101–122 | HAL hal-00975701, arXiv 1404.2405, UQTestFuns (flood) | yes |
| wicaksono2023 | Wicaksono, Hecht, *J. Open Source Softw.* 8(90) 5671 (2023) | JOSS page | yes |
| vlse | Surjanovic, Bingham, Virtual Library of Simulation Experiments | SFU website | URL |

Formula checks: the OTL circuit, piston, borehole (uniform and Harper & Gupta
input models), flood and Ishigami implementations in `src/benchmarks.py`
agree with UQTestFuns v0.7.0 to a relative difference below 1e-10, and all
non-uniform marginal quantiles agree to 1e-6 (`experiments/check_benchmarks.py`).
