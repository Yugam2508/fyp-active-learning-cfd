"""
cfd_iteration_check.py — iteration (convergence) error of the CFD case for
runs that stop at the 4000-iteration cap without meeting the residual
criteria (in the cached runs these are mostly tall buildings, whose steady
RANS wake keeps the residuals oscillating at a low level).

Re-runs the first five such cached flow states, writing the solution every
250 iterations from 2000 on, and reports for the receptor concentration at
x_r = 5, 20, 60, 120 m: the relative range (max - min) / mean over
iterations 2000-4000 and the relative change between 3000 and 4000.
Writes results/cfd_iteration_check.json.
"""

import os, sys, json, glob, re, tempfile, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import cfd_dispersion as cfd

XR = [5.0, 20.0, 60.0, 120.0]
recs = sorted((json.load(open(f)) for f in glob.glob(os.path.join(ROOT, "results", "cfd_runs", "*.json"))),
              key=lambda r: r["key"])
flows = [(r["U_ref"], r["z0"], r["H"], r["Sc_t"]) for r in recs if not r["converged"]][:5]
env = dict(os.environ, OMP_NUM_THREADS="1")
env.setdefault("WM_PROJECT_DIR", "/usr/share/openfoam")
out = []
for f in flows:
    with tempfile.TemporaryDirectory(dir=os.environ.get("CFD_WORKDIR")) as tmp:
        d = os.path.join(tmp, "case")
        cfd.write_case(d, *f)
        cd = open(f"{d}/system/controlDict").read()
        cd = cd.replace("writeInterval 4000;", "writeInterval 250;")
        open(f"{d}/system/controlDict", "w").write(cd)
        for cmd in ("blockMesh", "buoyantBoussinesqSimpleFoam"):
            subprocess.run([cmd, "-case", d], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, env=env, check=True)
        C = cfd.cell_centres(d)
        sel = (C[:, 0] > cfd.X_WW + cfd.W_B) & (C[:, 2] < f[2])
        xs, ix = cfd._levels(C[sel, 0]); zs, iz = cfd._levels(C[sel, 2])
        series = {}
        for t in sorted((t for t in os.listdir(d) if re.fullmatch(r"\d+", t) and int(t) >= 2000), key=int):
            s = cfd.read_internal_scalar(os.path.join(d, t, "T"))
            g = np.zeros((len(xs), len(zs))); g[ix, iz] = s[sel]
            prof = {"x": (xs - cfd.X_WW - cfd.W_B).tolist(),
                    "s": [float(np.interp(cfd.Z_REC, zs, col)) for col in g]}
            series[int(t)] = [float(cfd.receptor_concentration(prof, x, 1.0)) for x in XR]
    V = np.array([series[t] for t in sorted(series)])
    rng = ((V.max(0) - V.min(0)) / V.mean(0)).tolist()
    chg = ((np.array(series[4000]) - np.array(series[3000])) / np.array(series[4000])).tolist()
    out.append({"flow": dict(zip(("U_ref", "z0", "H", "Sc_t"), f)), "iterations": sorted(series),
                "C_Q1": V.tolist(), "rel_range_2000_4000": rng, "rel_change_3000_4000": chg})
    print(np.round(f, 3), "rel. range 2000-4000:", np.round(rng, 4), " change 3000->4000:",
          np.round(chg, 4), flush=True)
    json.dump({"x_r": XR, "cases": out},
              open(os.path.join(ROOT, "results", "cfd_iteration_check.json"), "w"), indent=1)
allr = np.abs([v for c in out for v in c["rel_range_2000_4000"]])
print(f"max rel. range {allr.max():.4f}, median {np.median(allr):.4f}")
