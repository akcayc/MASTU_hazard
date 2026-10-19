"""Does EPQ use more constraints than EPM?

The matched-pair design assumes EPM and EPQ stand to each other as DIII-D's
EFIT01 and EFIT02: one magnetics-only and widely available, the other better
constrained and present for fewer shots.  The evidence for that so far is one
line in saddle_data.load_betan, which tries /epq/ and falls back to /epm/.
That shows EPQ is preferred when present; it does not show what EPQ adds.

This answers it directly.  For each constraint group it reports whether the
node resolves in each tree and whether its WEIGHTS are actually non-zero -- a
constraint can be present and unused, which is the distinction that matters.
It also compares slice counts and reports EPQ availability over a shot range.

    python compare_trees.py --shot 47002
    python compare_trees.py --shot-min 47000 --shot-max 47100 --coverage-only
"""

import argparse

import numpy as np

import giopath  # noqa: F401

try:
    import pyuda
except ImportError:
    giopath.report()
    raise

#: every constraint group in the EPM tree dump
GROUPS = ["DIAMAGNETICFLUX", "FLUXLOOPS", "MAGNETICPROBES", "PFCIRCUITS",
          "PLASMACURRENT", "Q0"]
TREES = ["EPM", "EPQ"]


def get(client, tag, path, shot):
    try:
        v = client.get(f"{tag}/{path}", shot)
    except Exception:
        return None
    if not hasattr(v, "data"):
        return "container"
    return np.asarray(v.data, float)


def weights_live(w):
    """Does this constraint actually act?  Present but zero-weight means no."""
    if w is None or isinstance(w, str):
        return None
    w = w[np.isfinite(w)]
    if w.size == 0:
        return None
    nz = int(np.count_nonzero(w))
    return nz, w.size, float(np.nanmax(np.abs(w)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int)
    p.add_argument("--shot-min", type=int)
    p.add_argument("--shot-max", type=int)
    p.add_argument("--coverage-only", action="store_true")
    a = p.parse_args()
    client = pyuda.Client()

    if a.shot:
        print(f"shot {a.shot}\n")
        for tag in TREES:
            t = get(client, tag, "TIME", a.shot)
            if t is None:
                print(f"  {tag}: tree absent")
                continue
            print(f"  {tag}: {t.size} slices, [{t[0]:.4f}, {t[-1]:.4f}] s, "
                  f"dt = {1e3*np.median(np.diff(t)):.3f} ms")
        print()
        print(f"  {'constraint':>18} {'EPM':>22} {'EPQ':>22}")
        for g in GROUPS:
            cells = []
            for tag in TREES:
                tgt = get(client, tag, f"INPUT/CONSTRAINTS/{g}/TARGET", a.shot)
                w = get(client, tag, f"INPUT/CONSTRAINTS/{g}/WEIGHTS", a.shot)
                if tgt is None:
                    cells.append("absent")
                    continue
                lw = weights_live(w)
                if lw is None:
                    cells.append("present, w=?")
                else:
                    nz, n, mx = lw
                    cells.append(f"{nz}/{n} active, max {mx:g}"
                                 if nz else f"present, ALL ZERO")
            mark = "  <-- differs" if cells[0] != cells[1] else ""
            print(f"  {g:>18} {cells[0]:>22} {cells[1]:>22}{mark}")
        print("\n  'N/M active' = non-zero weights out of M.  A constraint that")
        print("  is present with all-zero weights is not being used.")
        print("  EPQ using strictly more -> the EFIT01/EFIT02 analogy holds and")
        print("  merge_pair.py is worth writing.")

    if a.shot_min is not None:
        shots = range(a.shot_min, a.shot_max)
        have = {t: 0 for t in TREES}
        both = 0
        for s in shots:
            got = {}
            for tag in TREES:
                tv = get(client, tag, "TIME", s)
                got[tag] = tv is not None and not isinstance(tv, str) and tv.size > 1
                have[tag] += int(got[tag])
            both += int(all(got.values()))
        n = len(shots)
        print(f"\n  coverage over {n} shots {a.shot_min}-{a.shot_max - 1}")
        for tag in TREES:
            print(f"    {tag}: {have[tag]:4d}  ({100*have[tag]/n:.0f}%)")
        print(f"    both: {both:4d}  ({100*both/n:.0f}%)  <- the matched pair")
        if have["EPQ"] < have["EPM"]:
            print("    EPQ is the scarcer tree, as EFIT02 is to EFIT01")
        elif have["EPQ"] == have["EPM"]:
            print("    equal coverage -- the EFIT01/EFIT02 analogy is weaker than")
            print("    assumed; check the constraint table above before merging")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
