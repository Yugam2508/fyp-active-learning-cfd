# IMEKO World Congress 2027 — extended abstract

XXV IMEKO World Congress, Rimini, Italy, 30 Aug – 3 Sep 2027.
Target track: **TC21 Mathematical Tools for Measurements** (alternative: TC7 Measurement Science).

## Requirements (from imeko2027.org, checked 2026-10-16)

| Stage | Window | Length |
|---|---|---|
| Extended abstract | 15 Oct 2026 – **15 Jan 2027** | min. 4 pages (author-instructions page says 4–5) |
| Notification | 28 Feb 2027 | |
| Revised full paper | 3 – 18 May 2027 | max. 6 pages |
| Final full paper | 25 May – 15 Jun 2027 | max. 6 pages |

- The extended abstract must contain: title, authors and affiliations, abstract,
  introduction, proposed contribution, preliminary results, conclusions, references.
- The **official IMEKO 2027 template is mandatory** (Word or LaTeX), from the
  "Initial Author Instructions" page. Submission is through **EDAS** (edas.info);
  the corresponding author needs an EDAS account.
- At least two independent reviewers per abstract.
- Proceedings are open access on the IMEKO site; extended versions can go to
  journals afterwards (Acta IMEKO asks for an extension of at least 30 %).

## Status of `extended_abstract.tex`

- Content is complete and follows the required sections. It compiles to 4 A4
  pages in a **stand-in layout** (two columns, Times, 20 mm margins), because
  the official template could not be downloaded from this environment.
- **To do before submission**
  1. Download the official LaTeX template and move the content into it; re-check
     the page count (4–5 pages).
  2. Supply the three figures (each slot shows a labelled box until the file exists):
     - `fig_ablation.pdf` — domain-size sweep (thesis runs)
     - `fig_ablation_strip.pdf` — `python experiments/plot_ablation_strip.py`
     - `fig_uq_pbox.pdf` — `python experiments/demo_uq.py`
  3. Confirm the affiliation and contact e-mail, and whether a supervisor is a co-author.
  4. Reformat references to the template's style if it differs.

Every number in the tables is recomputable from `results/*.json`.
