"""
test_reproducibility.py — regenerate a sample of published records with the
current code and compare every stored field with the committed results.

Usage: python experiments/test_reproducibility.py [--quick] [--rtol R]
Exit status 1 on any mismatch. --quick skips the slow adaptive committee.

--rtol is the relative tolerance for floating-point fields (default 1e-9,
i.e. bitwise up to printing: what the reference environment of
docs/REPRODUCIBILITY.md achieves). On different hardware the same pinned
packages can give slightly different floating-point results (BLAS kernels
and PyTorch CPU paths depend on the processor, and the hyperparameter
optimisation amplifies the last bits), so cross-machine checks such as the
CI use a looser tolerance. The largest relative deviation of every case is
printed either way.
"""

import os, sys, json, subprocess, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTOL = float(sys.argv[sys.argv.index("--rtol") + 1]) if "--rtol" in sys.argv else 1e-9
CASES = [("physics", "gp_orig"), ("physics", "gp_amp"), ("borehole", "gp_ard"),
         ("borehole", "gp_m52"), ("plume", "gp_log"), ("plume", "al_gp"),
         ("borehole", "pas_ard"), ("physics", "pas_orig"), ("ishigami", "gp_m52")]
if "--quick" not in sys.argv:
    CASES.append(("physics", "al_var"))


def deviation(a, b):
    """Largest relative deviation between a stored and a regenerated field
    (inf for a non-numeric mismatch)."""
    if isinstance(a, dict):
        if not a.keys() <= b.keys():
            return np.inf
        return max((deviation(a[k], b[k]) for k in a), default=0.0)
    if isinstance(a, float):
        return float(abs(a - b) / max(abs(a), 1e-12))
    return 0.0 if a == b else np.inf


ok = True
with tempfile.TemporaryDirectory() as tmp:
    env = dict(os.environ, RESULTS_DIR=tmp, OMP_NUM_THREADS="1")
    for bench, arm in CASES:
        subprocess.run([sys.executable, os.path.join(ROOT, "experiments",
                        "campaign_generalisation.py"), bench, arm, "0", "1"],
                       env=env, check=True, capture_output=True)
        new = json.load(open(os.path.join(tmp, f"results_gen_{bench}_{arm}.json")))[0]
        old = [r for r in json.load(open(os.path.join(ROOT, "results",
               f"results_gen_{bench}_{arm}.json"))) if r["seed"] == 1000][0]
        keys = [k for k in old if k != "runtime_s"]
        dev = {k: deviation(old[k], new.get(k)) if k in new else np.inf for k in keys}
        bad = [k for k in keys if dev[k] > RTOL]
        worst = max(dev, key=dev.get)
        ok &= not bad
        print(f"[{'ok' if not bad else 'FAIL'}] {bench}/{arm}: {len(keys)} fields, "
              f"max rel. deviation {dev[worst]:.1e} ({worst})"
              + (f", above rtol {RTOL:g}: {bad}" if bad else ""))
sys.exit(0 if ok else 1)
