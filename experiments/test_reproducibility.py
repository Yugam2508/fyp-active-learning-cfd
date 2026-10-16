"""
test_reproducibility.py — regenerate a sample of published records with the
current code and compare every stored field with the committed results.

Usage: python experiments/test_reproducibility.py [--quick]
Exit status 1 on any mismatch. --quick skips the slow adaptive committee.
"""

import os, sys, json, subprocess, tempfile
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [("physics", "gp_orig"), ("physics", "gp_amp"), ("borehole", "gp_ard"),
         ("borehole", "gp_m52"), ("plume", "gp_log"), ("plume", "al_gp"),
         ("borehole", "pas_ard"), ("physics", "pas_orig"), ("ishigami", "gp_m52")]
if "--quick" not in sys.argv:
    CASES.append(("physics", "al_var"))


def same(a, b):
    if isinstance(a, dict):
        return a.keys() <= b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float):
        return np.isclose(a, b, rtol=1e-9, atol=1e-12)
    return a == b


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
        bad = [k for k in keys if not same(old[k], new.get(k))]
        ok &= not bad
        print(f"[{'ok' if not bad else 'FAIL'}] {bench}/{arm}: {len(keys)} fields"
              + (f", mismatched: {bad}" if bad else ""))
sys.exit(0 if ok else 1)
