"""
cfd_gci.py — research plan T2.6: grid convergence index of the CFD case.

Adds a third, coarser mesh (half the cells in each direction, refine = 0.5)
to the two meshes of cfd_grid_check.py for the same six flow states, so
that with the constant refinement ratio r = 2 the observed order of
convergence p and the Richardson-extrapolated receptor concentration follow
(Roache 1998; Celik et al. 2008):
  p = ln|(f_c - f_m) / (f_m - f_f)| / ln r,  f_ext = f_f + (f_f - f_m) / (r^p - 1)
with f_c, f_m, f_f the coarse, standard (used in the study) and fine values.
Recorded per state and receptor distance: p, the relative error of the
standard mesh with respect to f_ext, and the GCI of the standard mesh,
1.25 |f_c - f_m| / f_m / (r^p - 1) (only where the three values converge
monotonically; oscillatory cases are reported as such).
Writes results/cfd_gci.json.
"""

import os, sys, json, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import cfd_dispersion as cfd

grid = json.load(open(os.path.join(ROOT, "results", "cfd_grid_check.json")))
OUT = os.path.join(ROOT, "results", "cfd_gci.json")
if os.path.exists(OUT) and len(json.load(open(OUT))["cases"]) == len(grid["cases"]):
    sys.exit(print("results/cfd_gci.json is complete; nothing to do"))
XR, out = grid["x_r"], []
for case in grid["cases"]:
    f = tuple(case["flow"].values())
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=os.environ.get("CFD_WORKDIR")) as tmp:
        res = cfd.run_case(*f, workdir=os.path.join(tmp, "case"), refine=0.5, max_iter=4000)
    fc = np.array([float(cfd.receptor_concentration(res, x, 1.0)) for x in XR])
    fm, ff = np.array(case["refine1"]["C_Q1"]), np.array(case["refine2"]["C_Q1"])
    row = {"flow": case["flow"], "C_coarse": fc.tolist(), "C_standard": fm.tolist(),
           "C_fine": ff.tolist(), "coarse_iterations": res["iterations"],
           "coarse_converged": res["converged"], "runtime_s": round(time.time() - t0, 1),
           "p": [], "err_standard": [], "gci_standard": [], "monotone": []}
    for c_, m_, f_ in zip(fc, fm, ff):
        e21, e32 = f_ - m_, m_ - c_
        mono = bool(e21 * e32 > 0)
        row["monotone"].append(mono)
        if not mono or e21 == 0:
            row["p"].append(None); row["err_standard"].append(None); row["gci_standard"].append(None)
            continue
        p = float(np.log(abs(e32 / e21)) / np.log(2))
        ext = f_ + e21 / (2 ** p - 1)
        row["p"].append(p)
        row["err_standard"].append(float((m_ - ext) / ext))
        row["gci_standard"].append(float(1.25 * abs(e32) / m_ / (2 ** p - 1)))
    out.append(row)
    print(case["flow"], "p", np.round([v if v is not None else np.nan for v in row["p"]], 2),
          "err std mesh", np.round([v if v is not None else np.nan for v in row["err_standard"]], 3),
          flush=True)
    json.dump({"x_r": XR, "cases": out}, open(OUT, "w"), indent=1)
