"""
cfd_smoke.py — checks that the installed OpenFOAM reproduces a stored CFD
result: the first coarse-mesh case of results/cfd_gci.json (about 10 s on
one core). Used by the Docker workflow; exit status 1 if a receptor
concentration differs by more than the relative tolerance (default 1e-3).
"""
import os, sys, json, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import cfd_dispersion as cfd

rtol = float(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--rtol=")), "1e-3"))
G = json.load(open(os.path.join(ROOT, "results", "cfd_gci.json")))
case = G["cases"][0]
with tempfile.TemporaryDirectory() as tmp:
    res = cfd.run_case(*case["flow"].values(), workdir=os.path.join(tmp, "case"),
                       refine=0.5, max_iter=1000)
c = np.array([float(cfd.receptor_concentration(res, x, 1.0)) for x in G["x_r"]])
err = float(np.max(np.abs(c - case["C_coarse"]) / np.array(case["C_coarse"])))
print("receptor concentrations", np.round(c, 4).tolist(), "max relative difference", err)
sys.exit(0 if err <= rtol else 1)
