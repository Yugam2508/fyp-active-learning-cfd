"""
check_benchmarks.py — cross-check src/benchmarks.py against UQTestFuns.

Compares (i) model outputs on random inputs and (ii) marginal quantiles of
the probabilistic input models, for every model UQTestFuns also implements.
Run: python experiments/check_benchmarks.py   (needs `pip install uqtestfuns`)
Exit status 1 if any check fails.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
import uqtestfuns as uqtf
from uqtestfuns.test_functions import otl_circuit, piston, borehole, flood, ishigami
from benchmarks import BENCHMARKS

ok = True
rs = np.random.RandomState(0)
U = rs.uniform(0.01, 0.99, (2000, 8))


def report(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"[{'ok' if cond else 'FAIL'}] {name} {detail}")


pairs = [("otl", uqtf.OTLCircuit(), otl_circuit.evaluate),
         ("piston", uqtf.Piston(), piston.evaluate),
         ("borehole_gum", uqtf.Borehole(), borehole.evaluate),
         ("flood", uqtf.Flood(), flood.evaluate),
         ("ishigami", uqtf.Ishigami(), lambda X: ishigami.evaluate(X, 7.0, 0.1))]
for key, ref, fref in pairs:
    b = BENCHMARKS[key]()
    X = b.ppf(U[:, :b.dim])
    y_ref = fref(X)
    y = b.evaluate(X)
    report(f"{key}: model output", np.allclose(y, y_ref, rtol=1e-10, atol=1e-12),
           f"max rel diff {np.max(np.abs(y - y_ref) / np.maximum(np.abs(y_ref), 1e-12)):.1e}")
    if b.marginals is not None:
        q = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
        for j, (m, mr) in enumerate(zip(b.marginals, ref.prob_input.marginals)):
            ours = np.array([float(m.ppf(np.array([qq]))[0]) for qq in q])
            theirs = mr.icdf(q)
            report(f"{key}: marginal {mr.name}", np.allclose(ours, theirs, rtol=1e-6),
                   f"max rel diff {np.max(np.abs(ours - theirs) / np.abs(theirs)):.1e}")

# uniform borehole: same formula as UQTestFuns on our (Surjanovic & Bingham) ranges
b = BENCHMARKS["borehole"]()
X = b.ppf(U)
report("borehole: model output", np.allclose(b.evaluate(X), borehole.evaluate(X), rtol=1e-12))
sys.exit(0 if ok else 1)
