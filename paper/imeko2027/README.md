# IMEKO World Congress 2027 — extended abstract

XXV IMEKO World Congress, Rimini, Italy, 30 Aug – 3 Sep 2027.
Target track: **TC21 Mathematical Tools for Measurements** (alternative: TC7 Measurement Science).

## Requirements (imeko2027.org, author instructions and template v2.5)

| Stage | Window | Length |
|---|---|---|
| Extended abstract | 15 Oct 2026 – **15 Jan 2027** | 4–5 pages |
| Notification | 28 Feb 2027 | |
| Revised full paper | 3 – 18 May 2027 | max. 6 pages |
| Final full paper | 25 May – 15 Jun 2027 | max. 6 pages |

- Required content: title; authors and affiliations; abstract; introduction with
  the problem and state of the art; proposed contribution; preliminary results and
  potential applications; conclusions; references. Very short or very preliminary
  abstracts may be desk-rejected.
- Template rules applied here: `\documentclass{imeko_wc}` with no layout changes;
  British English; abstract of at most about 150 words with no equations or
  references; keywords separated by semicolons; units via `siunitx`
  (`\qty`, `\num`), which the class already loads; captions above tables and below
  figures; "Fig." in references; acronyms defined at first use in the body;
  BibTeX with `imeko_wc.bst`; all authors listed unless there are six or more.
- Review criteria: relevance, technical quality and clarity, scientific relevance,
  originality.
- Submission: EDAS, <https://edas.info/newPaper.php?c=35447>. Register the paper
  (title and short abstract), then upload the PDF in the Extended Abstract row.
  Name, affiliation, country and e-mail in EDAS must **exactly match** the paper.

## Files

- `extended_abstract.tex`, `references.bib`: the manuscript.
- `imeko_wc.cls`, `imeko_wc.bst`, `imeko.png`, `logo_IMEKO2027_v1.png`: unmodified
  from the official package `class_iwc2027_v2.5.zip`.

Build (from this folder):

```bash
pdflatex extended_abstract && bibtex extended_abstract
pdflatex extended_abstract && pdflatex extended_abstract
```

Current output: 4 pages, no errors or overfull boxes; abstract is 148 words.

## To do before submission

1. **Figures.** Each slot shows a labelled box until its file exists in
   `figures/` (repo root) or `paper/imeko2027/figures/`:
   - `fig_ablation.pdf`: domain-size sweep (from the thesis runs)
   - `fig_ablation_strip.pdf`: `python experiments/plot_ablation_strip.py`
   - `fig_uq_pbox.pdf`: `python experiments/demo_uq.py`

   Axis labels must be words with units in parentheses (template rule).
2. **References.** Complete the author lists of the three entries marked
   `VERIFY` in `references.bib` (Tan 2017, Heidenreich 2017, Wang 2022). The
   template also encourages citing measurement journals (Metrologia, Measurement,
   Meas. Sci. Technol., Acta IMEKO). Consider adding relevant ones.
3. **Authors.** Confirm the name order, the affiliation, and the corresponding
   e-mail (an institutional address may be preferable), and whether a supervisor is a
   co-author. With more than one author, add the AUTHORS' CONTRIBUTION section
   (CRediT roles), as shown in the template.
4. **Funding.** `\funding{}` is empty; fill it in if there is any funding to declare.
