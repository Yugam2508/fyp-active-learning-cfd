"""
cfd_precompute.py — run, in parallel, every CFD case the CFD campaigns need
before they start, so that the campaigns themselves only read the cache
(results/cfd_runs/). Adaptive arms add their own runs as they go.

  1. the N = 20 Latin hypercube design and the N_INIT = 5 start design of
     each seed 1000-1029 (the same draws as campaign_generalisation.py)
  2. the 800 reference flow states of src/cfd_benchmark.py

Usage: CFD_WORKERS=4 python experiments/cfd_precompute.py [--designs-only]
Requires OpenFOAM v1912 (Ubuntu/Debian package `openfoam`).
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from framework_core import lhs
from cfd_benchmark import CFDDispersion, run_flows, _path

b = CFDDispersion()
flows = []
for seed in range(1000, 1030):
    for n in (20, 5):
        np.random.seed(seed)
        flows.append(b.ppf(lhs(b.dim, n))[:, :4])
designs = np.vstack(flows)
groups = [("designs", designs)]
if "--designs-only" not in sys.argv:
    groups.append(("reference", b.reference_flows()[0]))
for name, F in groups:
    todo = [f for f in F if not os.path.exists(_path(tuple(float(v) for v in f)))]
    t0 = time.time()
    print(f"{name}: {len(F)} flow states, {len(todo)} to run", flush=True)
    # in chunks, so that progress is visible in the log
    for i in range(0, len(todo), 40):
        run_flows(np.array(todo[i:i + 40]))
        print(f"  {name}: {min(i + 40, len(todo))}/{len(todo)} done, "
              f"{time.time() - t0:.0f} s", flush=True)
print("ALL CFD RUNS DONE")
