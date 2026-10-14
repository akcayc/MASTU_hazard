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

    # -- inspection ---------------------------------------------------------
    def chunk_files(self):
        return sorted(f for f in os.listdir(self.path)
                      if f.startswith(self.prefix + "-") and f.endswith(".pickle"))

    def done(self):
        """Shots already stored.  Cached; call refresh() after an append."""
        if self._done is None:
            self._done = set()
            for f in self.chunk_files():
                try:
                    with open(os.path.join(self.path, f), "rb") as fh:
                        self._done |= set(pickle.load(fh)["shots"])
                except Exception as e:
                    print(f"  warning: unreadable chunk {f} ({e}); ignoring")
        return self._done

    def refresh(self):
        self._done = None
        return self.done()

    # -- writing ------------------------------------------------------------
    def append(self, records, dropped=None):
        """Write one chunk atomically.  `records` is a list of per-shot dicts."""
        if not records and not dropped:
            return None
        shots = [r["shot"] for r in records]
        payload = {"shots": shots, "records": records,
                   "dropped": dropped or {}, "written": time.time()}
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

    def summary(self):
        files = self.chunk_files()
        nbytes = sum(os.path.getsize(os.path.join(self.path, f)) for f in files)
        return {"path": self.path, "chunks": len(files),
                "shots": len(self.done()), "bytes": nbytes}


def consolidate(store, out_path):
    """Merge a store into one pickle, matching extract_features' --out format."""
    records, dropped = store.load_all()
    with open(out_path, "wb") as fh:
        pickle.dump({"records": records, "dropped": dropped,
                     "tag": records[0]["tag"] if records else None}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return len(records), len(dropped)
