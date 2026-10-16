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
| Abstract ≤ ~150 words, no equations or references | yes (149 words as typeset) |
| Keywords separated by semicolons | yes |
| British English | yes |
| Units via `siunitx`; leading zeros | yes |
| Table captions above, figure captions below; "Fig." in text; every float cited before it appears | yes |
| Axis labels in words, units in parentheses | yes |
| Acronyms defined at first use in the body | yes |
| BibTeX with `imeko_wc.bst`; all authors listed (< 6); measurement journals cited | yes; 22 references, each checked in `docs/REFERENCES_VERIFICATION.md` |
| Authors' contribution section (more than one author) | yes |
| Compiles without errors, undefined references or overfull boxes | yes |
| Length | 5 pages (4 pages of text + references) |

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

## Authors (as entered in the paper; EDAS must match exactly)

| | Given names | Surname | Affiliation | E-mail |
|---|---|---|---|---|
| 1 (corresponding) | Yugam Jinesh | Vora | Department of Engineering Science, National University of Singapore, Singapore | yugam@u.nus.edu |
| 2 | Wee Hoe | Ng | National Metrology Centre, Agency for Science, Technology and Research (A*STAR), Singapore | — |

The AUTHORS' CONTRIBUTION section uses CRediT roles: Vora — conceptualisation,
methodology, software, investigation, formal analysis, visualisation, writing
(original draft); Ng — conceptualisation, supervision, writing (review and
editing). Adjust if the roles differ.

## Before uploading (author)

1. **Supervisor's affiliation**: set to A*STAR National Metrology Centre;
   confirm the exact wording he uses (a street address can be added).
2. **Supervisor's approval** of the text and author order.
3. **Funding.** `\funding{}` is empty; fill it in if any applies.
4. **Submit.** Log in to EDAS → <https://edas.info/newPaper.php?c=35447> →
   enter title and a short abstract → Register Paper → upload
   `extended_abstract.pdf` in the Extended Abstract row (blue cloud icon).
   Add Wee Hoe Ng as co-author in EDAS as well.

## Plan for the final paper (May 2027, max. 6 pages)

Stated in the Conclusions; the reviewers will expect them:
- a CFD-based dispersion case (the plume model here is analytical);
- calibrated (conformal) intervals for the surrogate contribution;
- budgets other than N = 20, and non-uniform input distributions.

## Built PDF

`extended_abstract.pdf` is the build of the current source (4 pages). This
is the file to upload; rebuild it after any edit.
