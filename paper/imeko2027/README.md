# IMEKO World Congress 2027 — extended abstract

XXV IMEKO World Congress, Rimini, Italy, 30 Aug – 3 Sep 2027.
Target track: **TC21 Mathematical Tools for Measurements** (alternative: TC7 Measurement Science).

**Status: ready to submit**, subject to the author checks under "Before
uploading" below.

## Requirements (imeko2027.org author instructions and template v2.5)

| Stage | Window | Length |
|---|---|---|
| Extended abstract | 15 Oct 2026 – **15 Jan 2027** | 4–5 pages |
| Notification | 28 Feb 2027 | |
| Revised full paper | 3 – 18 May 2027 | max. 6 pages |
| Final full paper | 25 May – 15 Jun 2027 | max. 6 pages |

Compliance of `extended_abstract.tex`:

| Rule | Status |
|---|---|
| Official class `imeko_wc`, no layout changes | yes (class files unmodified) |
| Sections: title, authors/affiliation, abstract, introduction + state of the art, proposed contribution, preliminary results + potential applications, conclusions, references | yes |
| Abstract ≤ ~150 words, no equations or references | yes (143 words in source, about 149 as typeset) |
| Keywords separated by semicolons | yes |
| British English | yes |
| Units via `siunitx`; leading zeros | yes |
| Table captions above, figure captions below; "Fig." in text; every float cited before it appears | yes |
| Axis labels in words, units in parentheses | yes |
| Acronyms defined at first use in the body | yes |
| BibTeX with `imeko_wc.bst`; all authors listed (< 6) | yes; checked in `docs/REFERENCES_VERIFICATION.md` |
| Compiles without errors, undefined references or overfull boxes | yes |
| Length | 4 pages |

Review criteria: relevance, technical quality and clarity, scientific
relevance, originality.

## Files

| File | Content |
|---|---|
| `extended_abstract.tex` | manuscript |
| `references.bib` | bibliography (verified) |
| `NUMBERS.md` | source and computation of every number in the manuscript |
| `check_numbers.py` | recomputes all of those numbers from `results/*.json` |
| `imeko_wc.cls`, `imeko_wc.bst`, `imeko.png`, `logo_IMEKO2027_v1.png` | official template files, unmodified (`class_iwc2027_v2.5.zip`) |

Figures are read from `../../figures/` and produced by
`experiments/make_paper_figures.py`.

## Build

```bash
cd paper/imeko2027
pdflatex extended_abstract && bibtex extended_abstract
pdflatex extended_abstract && pdflatex extended_abstract
python check_numbers.py            # optional: audit every number
```

The class needs `newtx` (TeX Live: `texlive-fonts-extra`) and the
`algorithms`/`algpseudocode` packages (`texlive-science`).

## Before uploading (author)

1. **Name, affiliation, e-mail.** The paper uses "Vora Yugam Jinesh",
   Department of Engineering Science, NUS, and `yjv2508@gmail.com`. EDAS
   requires the profile to match the paper **exactly**; an institutional
   address may be preferred.
2. **Co-authors.** If a supervisor is a co-author, add them with `\author`
   and add the AUTHORS' CONTRIBUTION section (CRediT roles) as in the template.
3. **Funding.** `\funding{}` is empty; fill it in if any applies.
4. **Submit.** Log in to EDAS → <https://edas.info/newPaper.php?c=35447> →
   enter title and a short abstract → Register Paper → upload the PDF in the
   Extended Abstract row (blue cloud icon).

## Plan for the final paper (May 2027, max. 6 pages)

Stated in the Conclusions; the reviewers will expect them:
- a benchmark with **no** anchor member in the committee;
- a dispersion-type test model;
- conformal calibration of the member envelopes (exceedance coverage is 62 %).

## Built PDF

`extended_abstract.pdf` is the build of the current source (4 pages). This
is the file to upload; rebuild it after any edit.
