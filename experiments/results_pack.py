"""
results_pack.py — the bulky result files are kept in two ZIP files, so that
the archive of a release (GitHub source ZIP, archived at Zenodo) has fewer
than the 1000 entries that Zenodo's preview lists:
  results/cfd_runs.zip   the cached CFD runs, results/cfd_runs/*.json
  results/records.zip    the per-seed records, results/results_*.json,
                         results/parts/ and results/reference_sklearn/
The members are byte-identical to the files. Unpack once before running any
script; pack again after a campaign has added or changed files.
  python experiments/results_pack.py          # unpack (never overwrites a
                                              # file that differs)
  python experiments/results_pack.py pack     # files -> ZIP files
  python experiments/results_pack.py check    # ZIP files agree with the
                                              # files; release archive size
"""
import os, sys, glob, io, zipfile, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PACKS = {"cfd_runs.zip": ["cfd_runs/*.json"],
         "records.zip": ["results_*.json", "parts/*/*.json", "reference_sklearn/*.json"]}
MAX_ENTRIES = 1000            # PREVIEWER_ZIP_MAX_FILES of Zenodo (invenio-previewer)
MAX_BYTES = 100 * 1024 ** 2   # PREVIEWER_ZIP_MAX_BYTES


def files(patterns):
    return sorted(os.path.relpath(f, RES).replace(os.sep, "/")
                  for p in patterns for f in glob.glob(os.path.join(RES, p)))


def members(pack):
    with zipfile.ZipFile(os.path.join(RES, pack)) as z:
        out = {}
        for name in z.namelist():
            if name.startswith("/") or ".." in name.split("/"):
                sys.exit(f"{pack}: unsafe member name {name}")
            out[name] = z.read(name)
        return out


def pack():
    for pack, patterns in PACKS.items():
        names = files(patterns)
        tmp = os.path.join(RES, pack + ".tmp")
        with zipfile.ZipFile(tmp, "w") as z:      # fixed dates and order: reproducible
            for name in names:
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type, info.create_system = zipfile.ZIP_DEFLATED, 3
                info.external_attr = 0o644 << 16
                z.writestr(info, open(os.path.join(RES, name), "rb").read(), compresslevel=9)
        os.replace(tmp, os.path.join(RES, pack))
        print(f"{pack}: {len(names)} files, {os.path.getsize(os.path.join(RES, pack)) / 1e6:.1f} MB")


def unpack():
    for pack in PACKS:
        new = differ = 0
        for name, data in members(pack).items():
            path = os.path.join(RES, name)
            if os.path.exists(path):
                if open(path, "rb").read() != data:
                    differ += 1
                    print(f"kept, differs from {pack}: results/{name}")
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
            new += 1
        print(f"{pack}: {new} files written, {differ} differing files kept")


def check():
    bad = []
    for pack, patterns in PACKS.items():
        m, names = members(pack), files(patterns)
        if names:                                 # unpacked: the files must agree
            bad += [f"results/{n} not in {pack}" for n in sorted(set(names) - set(m))]
            bad += [f"{pack}: results/{n} missing" for n in sorted(set(m) - set(names))]
            bad += [f"results/{n} differs from {pack}" for n in sorted(set(m) & set(names))
                    if open(os.path.join(RES, n), "rb").read() != m[n]]
        print(f"{pack}: {len(m)} members, {len(names)} files")
    try:                                          # the source ZIP of a release at HEAD
        arc = subprocess.run(["git", "-C", ROOT, "archive", "--format=zip", "--prefix=repo/", "HEAD"],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        print("not a git checkout: release archive not checked")
    else:
        n = len(zipfile.ZipFile(io.BytesIO(arc)).infolist())
        print(f"release archive at HEAD: {n} entries (limit {MAX_ENTRIES}), {len(arc) / 1e6:.1f} MB")
        if n > MAX_ENTRIES or len(arc) > MAX_BYTES:
            bad.append("release archive too large for Zenodo's preview")
    for b in bad[:20]:
        print(b)
    sys.exit(f"{len(bad)} problems" if bad else 0)


if __name__ == "__main__":
    {"unpack": unpack, "pack": pack, "check": check}[sys.argv[1] if len(sys.argv) > 1 else "unpack"]()
