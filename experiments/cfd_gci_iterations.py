"""
cfd_gci_iterations.py — iteration check of the coarse mesh of cfd_gci.py.
The coarse-mesh runs stop at 4000 iterations without meeting the residual
criteria (in the first case, inspected by hand, the flow residuals fall
below 1e-8 but the initial residual of the scalar stalls near 9e-6, the
criterion being 1e-6). This reruns the six coarse
cases with 1000 and 2000 iterations and records the largest relative change
of the receptor concentration against the 4000-iteration values in
results/cfd_gci.json. Writes results/cfd_gci_iterations.json.
"""
import os, sys, json, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import cfd_dispersion as cfd

G = json.load(open(os.path.join(ROOT, "results", "cfd_gci.json")))
out = []
for case in G["cases"]:
    f = tuple(case["flow"].values())
    ref = np.array(case["C_coarse"])
    row = {"flow": case["flow"]}
    for it in (1000, 2000):
        with tempfile.TemporaryDirectory(dir=os.environ.get("CFD_WORKDIR")) as tmp:
            res = cfd.run_case(*f, workdir=os.path.join(tmp, "case"), refine=0.5, max_iter=it)
        c = np.array([float(cfd.receptor_concentration(res, x, 1.0)) for x in G["x_r"]])
        row[f"max_rel_change_{it}"] = float(np.max(np.abs(c - ref) / ref))
    out.append(row)
    print(row, flush=True)
json.dump({"x_r": G["x_r"], "cases": out,
           "max_rel_change_1000": max(r["max_rel_change_1000"] for r in out),
           "max_rel_change_2000": max(r["max_rel_change_2000"] for r in out)},
          open(os.path.join(ROOT, "results", "cfd_gci_iterations.json"), "w"), indent=1)
