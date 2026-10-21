"""
merge_parts.py — merge per-seed-block result files written with
RESULTS_DIR=results/parts/<block> into results/<file>.json (records sorted
by seed; an existing record for a seed is kept, a duplicate must agree).
Usage: python experiments/merge_parts.py
"""

import os, json, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
merged = {}
for f in sorted(glob.glob(os.path.join(RES, "parts", "*", "*.json"))):
    merged.setdefault(os.path.basename(f), []).extend(json.load(open(f)))
for name, recs in merged.items():
    path = os.path.join(RES, name)
    have = {r["seed"]: r for r in (json.load(open(path)) if os.path.exists(path) else [])}
    for r in recs:
        if r["seed"] in have:
            a, b = dict(have[r["seed"]]), dict(r)
            a.pop("runtime_s", None); b.pop("runtime_s", None)
            assert a == b, f"{name}: seed {r['seed']} differs between blocks"
        have.setdefault(r["seed"], r)
    json.dump([have[s] for s in sorted(have)], open(path, "w"))
    print(f"{name}: {len(have)} records")
