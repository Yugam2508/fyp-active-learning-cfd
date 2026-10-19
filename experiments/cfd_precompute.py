"""
cfd_precompute.py — run, in parallel, every CFD case the CFD campaigns need
before they start, so that the campaigns themselves only read the cache
(results/cfd_runs/). Adaptive arms add their own runs as they go.

  1. the N = 20 Latin hypercube design and the N_INIT = 5 start design of
     each seed 1000-1029 (the same draws as campaign_generalisation.py)
  2. the 800 reference flow states of src/cfd_benchmark.py
Options (research plan T1.1, T2.1, T2.6):
  --N=10,40          also the designs of these budgets (the N_INIT start
                     designs of the adaptive arms are among them)
  --design=K         also the N = 20 designs of type K (maximin, sobol)
  --ref2             the 800 flow states of the second reference set (seed 18)
  --reverse          run the outstanding flow states in reverse order (a
                     second instance then works towards the first from the
                     other end; every run checks the cache before starting)

Usage: CFD_WORKERS=4 python experiments/cfd_precompute.py [--designs-only] [options]
Requires OpenFOAM v1912 (Ubuntu/Debian package `openfoam`).
"""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np
from framework_core import lhs
from cfd_benchmark import CFDDispersion, run_flows, _path
from designs import make_design

b = CFDDispersion()
flows = []
for seed in range(1000, 1030):
    for n in (20, 5):
        np.random.seed(seed)
        flows.append(b.ppf(lhs(b.dim, n))[:, :4])
for arg in sys.argv[1:]:
    if arg.startswith("--N="):
        for n in map(int, arg[4:].split(",")):
            for seed in range(1000, 1030):
                np.random.seed(seed)
                flows.append(b.ppf(lhs(b.dim, n))[:, :4])
    if arg.startswith("--design="):
        for seed in range(1000, 1030):
            np.random.seed(seed)
            flows.append(b.ppf(make_design(arg[9:], b.dim, 20))[:, :4])
designs = np.vstack(flows)
groups = [("designs", designs)]
if "--ref2" in sys.argv:
    groups.append(("reference 2", b.reference_flows(18)[0]))
if "--designs-only" not in sys.argv:
    groups.append(("reference", b.reference_flows()[0]))
for name, F in groups:
    todo = [f for f in F if not os.path.exists(_path(tuple(float(v) for v in f)))]
    if "--reverse" in sys.argv:
        todo = todo[::-1]
    t0 = time.time()
    print(f"{name}: {len(F)} flow states, {len(todo)} to run", flush=True)
    # in chunks, so that progress is visible in the log
    for i in range(0, len(todo), 40):
        try:
            run_flows(np.array(todo[i:i + 40]))
        except Exception as e:          # report and go on; rerun to retry
            print(f"  FAILED in chunk {i // 40}: {e!r}", flush=True)
        print(f"  {name}: {min(i + 40, len(todo))}/{len(todo)} done, "
              f"{time.time() - t0:.0f} s", flush=True)
print("ALL CFD RUNS DONE")
