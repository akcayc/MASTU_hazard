"""Book-keeping: an annotation's n and threshold travel with its features.

  * make_annotation puts n in the file name and writes a .json sidecar;
  * a ChunkStore refuses to resume from a different annotation;
  * extract_parallel's meta carries the sidecar, consolidate carries the meta;
  * merge_pair refuses EPM and EPQ harvested from different annotations.
"""
import _paths  # noqa: F401
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from checkpoint import ChunkStore, consolidate
import extract_parallel as XP
import merge_pair as MP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ladder(path, n):
    rng = np.random.default_rng(0)
    shots = np.arange(100, 120)
    rm = np.where(rng.random(shots.size) < 0.4, 0.3 + 0.2 * rng.random(shots.size), np.nan)
    pd.DataFrame({"shot": shots, "isok_rm": 1, "whichn": n, "ta_rm": 0.1,
                  "tb_rm": 0.8, "rm0.01000": rm, "fa0.01000": 5e3,
                  "co0.01000": 0.9}).to_csv(path, index=False)
    return int(np.isfinite(rm).sum())


def raises(fn):
    try:
        fn()
    except SystemExit:
        return True
    return False


def main():
    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")

    d = tempfile.mkdtemp()
    try:
        # -- make_annotation: n in the name, sidecar beside it -------------
        lad = os.path.join(d, "rm_ladder_g0.8_n2.csv")
        E = ladder(lad, 2)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "make_annotation.py"),
                            lad, "--threshold", "0.01", "--out",
                            os.path.join(d, "joint.csv")],
                           capture_output=True, text=True)
        csv = os.path.join(d, "joint_n2_threshold0.01.csv")
        side = os.path.join(d, "joint_n2_threshold0.01.json")
        check("annotation named with n and threshold", r.returncode == 0
              and os.path.isfile(csv))
        check("last line still 'wrote <csv>' for run-study.sh",
              r.stdout.strip().splitlines()[-1].strip() == f"wrote {csv}")
        meta = json.load(open(side)) if os.path.isfile(side) else {}
        check("sidecar records n, threshold, events",
              meta.get("n") == 2 and meta.get("threshold") == 0.01
              and meta.get("events") == E and meta.get("ladder") == os.path.abspath(lad))

        # -- extract_parallel meta picks up the sidecar --------------------
        m1 = XP.source_meta(csv, "annotation", "EPM", False, True, 5)
        check("meta carries the sidecar and a checksum",
              m1["annotation"] == meta and len(m1["sha256"]) == 64)

        # -- a store refuses a different annotation ------------------------
        st = ChunkStore(os.path.join(d, "harvest_EPM"))
        st.append([{"shot": 100, "tag": "EPM"}], meta=m1)
        check("same annotation resumes",
              not raises(lambda: ChunkStore(st.path).check_meta(m1, XP.META_KEYS)))
        other = os.path.join(d, "joint_n1.csv")
        shutil.copy(csv, other)
        with open(other, "a") as fh:
            fh.write("121,0.1,0.8,nan,nan,0.8,0,nan,nan\n")
        m2 = XP.source_meta(other, "annotation", "EPM", False, True, 5)
        check("different annotation refused",
              raises(lambda: ChunkStore(st.path).check_meta(m2, XP.META_KEYS)))
        m3 = dict(m1, use_alts=True)
        check("different options refused",
              raises(lambda: ChunkStore(st.path).check_meta(m3, XP.META_KEYS)))

        # a chunk from before metas: counted, not refused
        legacy = ChunkStore(os.path.join(d, "old"))
        legacy.append([{"shot": 1, "tag": "EPM"}])
        check("legacy chunk counted as unverified",
              ChunkStore(legacy.path).check_meta(m1, XP.META_KEYS) == 1)

        # -- consolidate carries the meta ----------------------------------
        out = os.path.join(d, "feat_EPM.pickle")
        consolidate(ChunkStore(st.path), out)
        prov = MP.load_provenance(out)
        check("consolidated pickle has provenance",
              prov is not None and prov["sha256"] == m1["sha256"]
              and prov["unverified_chunks"] == 0 and prov["annotation"]["n"] == 2)

        # -- a finished harvest is recognised, a foreign pickle refused ----
        check("no pickle yet: harvest", XP.finished_pickle(os.path.join(d, "x.pickle"), m1) is False)
        check("pickle from the same annotation: done", XP.finished_pickle(out, m1))
        check("pickle from another annotation: refused",
              raises(lambda: XP.finished_pickle(out, m2)))

        # -- merge_pair compares the two trees -----------------------------
        check("same annotation passes", MP.check_same_annotation(m1, dict(m1, tag="EPQ")))
        check("different annotation refused",
              raises(lambda: MP.check_same_annotation(m1, m2)))
        check("missing provenance is reported, not refused",
              MP.check_same_annotation(m1, None) is False)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
