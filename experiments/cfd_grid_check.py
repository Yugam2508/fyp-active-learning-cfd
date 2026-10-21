"""
cfd_grid_check.py — grid sensitivity of the CFD dispersion case.

Runs six flow states (the centre of the input box and five spread-out
states) on the standard mesh (7776 cells) and on a mesh refined twice in
each direction (31 104 cells), and compares the receptor concentration at
x_r = 5, 20, 60 and 120 m. Writes results/cfd_grid_check.json.
Usage: python experiments/cfd_grid_check.py
"""

import os, sys, json, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import cfd_dispersion as cfd

FLOWS = [(3.5, 0.155, 10.5, 0.9),          # centre of the input box
         (1.5, 0.05, 7.0, 0.6), (5.5, 0.25, 14.0, 1.2), (2.0, 0.25, 8.0, 1.1),
         (5.0, 0.03, 13.0, 0.7), (1.2, 0.28, 14.5, 1.25)]
XR = [5.0, 20.0, 60.0, 120.0]
out = []
for f in FLOWS:
    row = {"flow": dict(zip(("U_ref", "z0", "H", "Sc_t"), f))}
    for r in (1, 2):
        t0 = time.time()
        with tempfile.TemporaryDirectory(dir=os.environ.get("CFD_WORKDIR")) as tmp:
            res = cfd.run_case(*f, workdir=os.path.join(tmp, "case"), refine=r,
                               max_iter=4000 * r)
        row[f"refine{r}"] = {"C_Q1": [float(cfd.receptor_concentration(res, x, 1.0)) for x in XR],
                             "iterations": res["iterations"], "converged": res["converged"],
                             "runtime_s": round(time.time() - t0, 1)}
    c1, c2 = np.array(row["refine1"]["C_Q1"]), np.array(row["refine2"]["C_Q1"])
    row["rel_diff"] = ((c1 - c2) / c2).tolist()
    out.append(row)
    print(f, "rel. diff (coarse - fine)/fine at x_r =", XR, ":",
          np.round(row["rel_diff"], 3), flush=True)
    json.dump({"x_r": XR, "cases": out}, open(os.path.join(ROOT, "results", "cfd_grid_check.json"), "w"), indent=1)
allrd = np.abs([d for c in out for d in c["rel_diff"]])
print(f"max |rel. diff| {allrd.max():.3f}, median {np.median(allrd):.3f}")
