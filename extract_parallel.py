"""Threaded, resumable driver for extract_features.

Measured on Freya: 33 ms per node read, 29 reads per shot, so ~0.96 s/shot.
That is 5.3 h for 10,000 shots across EPM and EPQ, serially.  The time is
network latency, not computation -- the client is waiting, not working -- so
threads help almost linearly until the server pushes back.

Two pyuda notes:
  * Each worker gets its OWN client, via threading.local.  pyuda.Client is not
    documented as thread-safe and sharing one is not worth the risk.
  * --workers defaults to 8.  Raise it until the per-shot rate stops improving;
    past that you are only queueing on the server.

Results are written through checkpoint.ChunkStore, so a run that dies resumes
where it stopped.

    python extract_parallel.py --annotation joint.csv --tag EPM \
        --store harvest_EPM --workers 8
    python extract_parallel.py --store harvest_EPM --consolidate feat_EPM.pickle

Use --annotation.  It chops each shot at tevent, which is mandatory: the model
must never see a row from after the event.  Downstream, table_stack labels the
LAST row of each shot precisely because that row is the last pre-event slice --
so a table chopped anywhere else silently labels the wrong row.

--ladder exists for the case where you want one harvest to serve several
thresholds, since [ta_rm, tb_rm] does not depend on which rm* column is picked.
It is NOT the default path: it requires the stacking logic to do the tevent
chop instead, and that is the one place a bug would mislabel the whole dataset
without erroring.  At 33 ms/read with 8 workers a re-harvest is ~40 min per
tag, so the saving rarely justifies the risk.
"""

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import giopath  # noqa: F401
import checkpoint as CP
import extract_features as EF

try:
    import pyuda
except ImportError:
    giopath.report()
    raise

_local = threading.local()


def client():
    """One pyuda client per worker thread."""
    c = getattr(_local, "client", None)
    if c is None:
        c = pyuda.Client()
        _local.client = c
    return c


def windows_from_ladder(path):
    """{shot: (ta, tb)} from a ladder CSV -- the UNCHOPPED flat-top window.

    ta_rm and tb_rm do not depend on which rm* column is picked, so one harvest
    serves every threshold.  The tevent chop must then happen downstream; see
    the module docstring before using this.
    """
    df = pd.read_csv(path)
    for c in ("shot", "ta_rm", "tb_rm"):
        if c not in df.columns:
            raise SystemExit(f"{path}: missing column {c}")
    return {int(r["shot"]): (float(r["ta_rm"]), float(r["tb_rm"]))
            for _, r in df.iterrows()}


def windows_from_annotation(path):
    """{shot: (ta, tb)} from a joint annotation CSV -- chops at tevent."""
    ann = EF.read_annotation(path)
    return {s: (v["ta"], v["tb"]) for s, v in ann.items()}


def run(shots, windows, tag, store, workers, min_timestamps, use_alts,
        include_glitch, chunk_size, report_every):
    todo = [s for s in shots if s not in store.done()]
    skipped = len(shots) - len(todo)
    if skipped:
        print(f"  resuming: {skipped} shots already in {store.path}")
    if not todo:
        print("  nothing to do")
        return

    print(f"  {len(todo)} shots, {workers} workers, chunks of {chunk_size}")
    t0 = time.time()
    buf, dropped, n_done = [], {}, 0
    lock = threading.Lock()

    def work(shot):
        ta, tb = windows.get(shot, (None, None))
        return EF.extract_shot(shot, tag, client(), ta, tb, use_alts,
                               include_glitch, min_timestamps)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(work, s): s for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                with lock:
                    dropped[s] = f"{type(e).__name__}: {e}"
            else:
                with lock:
                    buf.append(rec)
            with lock:
                n_done += 1
                if len(buf) >= chunk_size:
                    store.append(buf, dropped)
                    buf, dropped = [], {}
                if n_done % report_every == 0:
                    el = time.time() - t0
                    rate = n_done / el
                    eta = (len(todo) - n_done) / rate if rate else float("nan")
                    print(f"    {n_done}/{len(todo)}  {rate:.1f} shot/s  "
                          f"ETA {eta/60:.1f} min")

    if buf or dropped:
        store.append(buf, dropped)
    el = time.time() - t0
    print(f"  done: {len(todo)} shots in {el/60:.1f} min "
          f"({len(todo)/max(el,1e-9):.1f} shot/s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotation", help="joint annotation CSV; chops each shot "
                                        "at tevent (the normal path)")
    p.add_argument("--ladder", help="ladder CSV instead; window is the full "
                                    "[ta_rm, tb_rm] flat-top and is NOT chopped "
                                    "at tevent -- the stacking step must then do "
                                    "it. Advanced; see the module docstring")
    p.add_argument("--store", required=True, help="checkpoint directory")
    p.add_argument("--tag", default="EPM")
    p.add_argument("--shots", type=int, nargs="+")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=CP.CHUNK_SIZE)
    p.add_argument("--report-every", type=int, default=100)
    p.add_argument("--min-timestamps", type=int, default=5)
    p.add_argument("--use-alts", action="store_true")
    p.add_argument("--no-glitch", action="store_true")
    p.add_argument("--consolidate", help="merge the store into one pickle and exit")
    p.add_argument("--status", action="store_true", help="report the store and exit")
    a = p.parse_args()

    store = CP.ChunkStore(a.store)

    if a.status:
        s = store.summary()
        print(f"  {s['path']}: {s['chunks']} chunks, {s['shots']} shots, "
              f"{s['bytes']/1e6:.1f} MB")
        return 0

    if a.consolidate:
        n, nd = CP.consolidate(store, a.consolidate)
        print(f"  {n} records, {nd} dropped -> {a.consolidate}")
        return 0

    if a.annotation:
        windows = windows_from_annotation(a.annotation)
    elif a.ladder:
        windows = windows_from_ladder(a.ladder)
        print("  WARNING --ladder: the table is NOT chopped at tevent, so it")
        print("          contains post-event rows.  The stacking step must chop")
        print("          them, or every shot will be labelled on the wrong row.")
    else:
        p.error("--annotation is required (or --ladder; see the docstring)")

    shots = a.shots or sorted(windows)
    run(shots, windows, a.tag.upper(), store, a.workers, a.min_timestamps,
        a.use_alts, not a.no_glitch, a.chunk_size, a.report_every)

    s = store.summary()
    print(f"  store: {s['chunks']} chunks, {s['shots']} shots, "
          f"{s['bytes']/1e6:.1f} MB")
    print(f"  consolidate with:  python extract_parallel.py --store {a.store} "
          f"--consolidate feat_{a.tag.upper()}.pickle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
