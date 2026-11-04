"""Resumable chunked store for long extraction runs.

A 10,000-shot harvest is hours of network-bound reads.  A plain loop that dies
at shot 8,000 loses everything, so results are written in chunks as they
arrive and a restart skips what is already on disk.

Writes are atomic -- each chunk goes to a temporary file and is renamed into
place -- so a kill during a write leaves the store readable, just one chunk
shorter.  Nothing here imports pyuda, so it is testable offline.

    store = ChunkStore("harvest_EPM")
    todo = [s for s in shots if s not in store.done()]
    ...
    store.append(records)          # every CHUNK_SIZE shots
    recs = store.load_all()        # when finished
    consolidate_and_remove(store, "feat.pickle")   # one file; store deleted

Each chunk can carry a `meta` dict saying what produced it (extract_parallel
puts the annotation's checksum there).  check_meta() refuses to resume a store
whose chunks were made from something else, so one store never mixes two
annotations; consolidate() copies the agreed meta into the pickle as
`provenance`.
"""

import os
import pickle
import time

CHUNK_SIZE = 200


class ChunkStore:
    """Append-only directory of pickled record chunks, keyed by shot."""

    def __init__(self, path, prefix="chunk"):
        self.path = path
        self.prefix = prefix
        os.makedirs(path, exist_ok=True)
        self._done = None
        self._metas = None

    # -- inspection ---------------------------------------------------------
    def chunk_files(self):
        return sorted(f for f in os.listdir(self.path)
                      if f.startswith(self.prefix + "-") and f.endswith(".pickle"))

    def _scan(self):
        self._done, self._metas = set(), {}
        for f in self.chunk_files():
            try:
                with open(os.path.join(self.path, f), "rb") as fh:
                    p = pickle.load(fh)
            except Exception as e:
                print(f"  warning: unreadable chunk {f} ({e}); ignoring")
                continue
            self._done |= set(p["shots"])
            self._metas[f] = p.get("meta")

    def done(self):
        """Shots already stored.  Cached; call refresh() after an append."""
        if self._done is None:
            self._scan()
        return self._done

    def metas(self):
        """{chunk file: meta}, None for a chunk written before metas existed."""
        if self._metas is None:
            self._scan()
        return self._metas

    def refresh(self):
        self._done = self._metas = None
        return self.done()

    def check_meta(self, meta, keys):
        """Hard error if any chunk's meta differs from `meta` on `keys`.

        Chunks without a meta predate the check and cannot be verified; they
        are counted and reported, not refused.
        """
        legacy, bad = 0, []
        for f, m in self.metas().items():
            if m is None:
                legacy += 1
            elif any(m.get(k) != meta.get(k) for k in keys):
                bad.append((f, {k: m.get(k) for k in keys if m.get(k) != meta.get(k)}))
        if bad:
            f, diff = bad[0]
            want = {k: meta.get(k) for k in diff}
            raise SystemExit(
                f"{self.path}: {len(bad)} chunks were made with different "
                f"settings,\n  e.g. {f}: {diff}\n  this run: {want}\n"
                f"  use a different --store, or delete {self.path}")
        return legacy

    # -- writing ------------------------------------------------------------
    def append(self, records, dropped=None, meta=None):
        """Write one chunk atomically.  `records` is a list of per-shot dicts."""
        if not records and not dropped:
            return None
        shots = [r["shot"] for r in records]
        payload = {"shots": shots, "records": records,
                   "dropped": dropped or {}, "written": time.time(),
                   "meta": meta}
        name = f"{self.prefix}-{len(self.chunk_files()):05d}-{int(time.time()*1e3)}.pickle"
        final = os.path.join(self.path, name)
        tmp = final + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)          # atomic: readers see old or new, never partial
        if self._done is not None:
            self._done |= set(shots)
            self._metas[name] = meta
        return final

    # -- reading ------------------------------------------------------------
    def load_all(self):
        """Every record, de-duplicated by shot, last write winning."""
        by_shot, dropped = {}, {}
        for f in self.chunk_files():
            with open(os.path.join(self.path, f), "rb") as fh:
                p = pickle.load(fh)
            for r in p["records"]:
                by_shot[r["shot"]] = r
            dropped.update(p.get("dropped", {}))
        for s in by_shot:
            dropped.pop(s, None)        # a later success clears an earlier failure
        return [by_shot[s] for s in sorted(by_shot)], dropped

    def remove(self):
        """Delete every chunk, then the directory if nothing else is in it."""
        for f in self.chunk_files():
            os.remove(os.path.join(self.path, f))
        self._done = self._metas = None
        try:
            os.rmdir(self.path)
        except OSError:
            pass                        # something else lives there; leave it

    def summary(self):
        files = self.chunk_files()
        nbytes = sum(os.path.getsize(os.path.join(self.path, f)) for f in files)
        return {"path": self.path, "chunks": len(files),
                "shots": len(self.done()), "bytes": nbytes}


def consolidate(store, out_path):
    """Merge a store into one pickle, matching extract_features' --out format.

    `provenance` is the chunks' common meta, with `unverified_chunks` counting
    those written before metas existed; None if no chunk carries one.
    check_meta() at write time is what keeps the metas in agreement.
    """
    records, dropped = store.load_all()
    metas = list(store.metas().values())
    known = [m for m in metas if m is not None]
    prov = dict(known[-1]) if known else None
    if prov is not None:
        prov["unverified_chunks"] = len(metas) - len(known)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"records": records, "dropped": dropped,
                     "tag": records[0]["tag"] if records else None,
                     "provenance": prov}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)           # atomic, as for chunks
    return len(records), len(dropped)


def consolidate_and_remove(store, out_path):
    """consolidate(), re-read the pickle, and delete the store only if every
    shot in the store is in it.  Returns (records, dropped, removed)."""
    n, nd = consolidate(store, out_path)
    with open(out_path, "rb") as fh:
        back = pickle.load(fh)
    shots = {r["shot"] for r in back["records"]}
    want = store.refresh()
    if shots != want:
        print(f"  store kept: {out_path} has {len(shots)} shots, the store "
              f"{len(want)}")
        return n, nd, False
    store.remove()
    return n, nd, True
