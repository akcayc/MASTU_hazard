"""checkpoint.ChunkStore: resume, atomicity, de-duplication."""
import _paths  # noqa: F401
import os
import pickle
import shutil
import tempfile

from checkpoint import ChunkStore, consolidate


def main():
    ok = True
    d = tempfile.mkdtemp()
    try:
        st = ChunkStore(os.path.join(d, "s"))

        def check(label, cond):
            nonlocal ok
            ok &= bool(cond)
            print(f"  {'PASS' if cond else 'FAIL'}  {label}")

        check("empty store has nothing done", st.done() == set())

        st.append([{"shot": 1, "tag": "EPM"}, {"shot": 2, "tag": "EPM"}])
        st.append([{"shot": 3, "tag": "EPM"}], dropped={4: "boom"})
        check("done() tracks appended shots", st.done() == {1, 2, 3})
        check("two chunks on disk", len(st.chunk_files()) == 2)

        # a fresh store over the same directory must see the same work
        st2 = ChunkStore(os.path.join(d, "s"))
        check("resume: new instance sees prior shots", st2.done() == {1, 2, 3})

        recs, dropped = st2.load_all()
        check("load_all returns all records", [r["shot"] for r in recs] == [1, 2, 3])
        check("dropped carried through", dropped == {4: "boom"})

        # a later success for a previously dropped shot clears the failure
        st2.append([{"shot": 4, "tag": "EPM"}])
        recs, dropped = st2.load_all()
        check("retry clears the earlier drop", 4 not in dropped and len(recs) == 4)

        # duplicate shot: last write wins, no duplicate rows
        st2.append([{"shot": 2, "tag": "EPM", "v": 99}])
        recs, _ = st2.load_all()
        check("duplicate de-duplicated", len([r for r in recs if r["shot"] == 2]) == 1)
        check("last write wins", [r for r in recs if r["shot"] == 2][0].get("v") == 99)

        # no .tmp files survive a clean write
        check("no temp files left", not any(f.endswith(".tmp")
                                            for f in os.listdir(st2.path)))

        # a truncated chunk must not take the store down
        bad = os.path.join(st2.path, "chunk-99999-0.pickle")
        with open(bad, "wb") as fh:
            fh.write(b"\x80\x04\x95truncated")
        st3 = ChunkStore(st2.path)
        got = st3.done()
        check("unreadable chunk is skipped, not fatal", got == {1, 2, 3, 4})
        os.remove(bad)

        out = os.path.join(d, "merged.pickle")
        n, nd = consolidate(ChunkStore(st2.path), out)
        with open(out, "rb") as fh:
            m = pickle.load(fh)
        check("consolidate writes the extract_features shape",
              n == 4 and set(m) == {"records", "dropped", "tag"})
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
