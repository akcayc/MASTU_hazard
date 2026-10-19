"""Merge EPQ onto EPM's time base, and flatten to a survival table.

The MAST-U analogue of processEqDescriptorPair.py.  EPM is the primary, as
EFIT01 is for DIII-D: every EPM row is kept, and EPQ contributes NaN wherever
it has no slice within the matching window.  The choice between

    full EPM        every row, X1 only
    matched pair    only rows where EPQ also has a slice

is therefore made downstream, not here -- the merged table carries a `matched`
mask so both analyses run off one harvest.

Tolerances are derived from the measured 5.03 ms EPM cadence, not inherited
from DIII-D's 20 ms grid:

    merge delta   dt/4  = 1.26 ms   narrow enough that only genuinely
                                    co-timed slices pair, never neighbours
    event_delta   2 dt  = 10.05 ms  how far the last surviving equilibrium
                                    may sit before the event time before the
                                    label is untrustworthy

Because the window is +/- dt/4, at most one secondary slice can fall inside it,
so "earliest candidate" and "nearest candidate" are the same thing here -- the
ambiguity in the DIII-D original does not arise.

    python merge_pair.py --primary feat_EPM.pickle --secondary feat_EPQ.pickle \\
        --out merged.pickle
"""

import argparse
import pickle

import numpy as np

EPM_NOMINAL_DT = 5.025e-3


def derived_tolerances(dt):
    return {"merge_delta": 0.25 * dt, "event_delta": 2.0 * dt}


def load_records(path):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return d["records"] if isinstance(d, dict) and "records" in d else d


def create_index(records):
    return {int(r["shot"]): i for i, r in enumerate(records)}


def create_merge(D1, D2, delta):
    """Secondary features on the primary's time base; NaN where unmatched."""
    assert D1["shot"] == D2["shot"], "shot numbers must match"
    assert D1["feature-names"] == D2["feature-names"], "feature sets must match"
    t1, t2 = np.asarray(D1["time"]), np.asarray(D2["time"])
    X2 = np.full((t1.size, len(D1["feature-names"])), np.nan)
    for k, tk in enumerate(t1):
        cand = np.flatnonzero((t2 >= tk - delta) & (t2 <= tk + delta))
        if cand.size:
            X2[k, :] = D2["features"][cand[0], :]
    return X2


def create_merged_tables(T1, T2, delta, verbose=True):
    I2 = create_index(T2)
    tag2 = T2[0]["tag"] if T2 else None
    out, rows, missing, no_secondary = [], 0, 0, 0
    for D1 in T1:
        s = int(D1["shot"])
        if s in I2:
            X12 = create_merge(D1, T2[I2[s]], delta)
        else:
            X12 = np.full(D1["features"].shape, np.nan)
            no_secondary += 1
        D = dict(D1)
        D["features.2"] = X12
        D["tag.2"] = tag2
        out.append(D)
        rows += X12.shape[0]
        missing += int(np.sum(~np.isfinite(X12[:, 0])))
    if verbose:
        print(f"  {len(out)} shots, {rows} rows")
        print(f"    {no_secondary} shots have no secondary tree at all")
        print(f"    {missing} rows unmatched ({100*missing/max(rows,1):.1f}%)"
              f" -- these are NaN in X2, not dropped")
    return out


def table_stack(T12, features, event_delta, verbose=True):
    """Flatten to X1, X2, T, ID, Y, matched.

    The label goes on the LAST row of an event shot, because the table was
    chopped at tevent during extraction, so that row IS the last pre-event
    slice.  It is only applied if that row sits within event_delta of the
    event: if the equilibrium stopped earlier -- common, since a growing
    island degrades the fit -- attaching an onset label to a reconstruction
    from long before would be false, so the event is dropped and counted.
    """
    N = sum(len(e["time"]) for e in T12)
    cols = [T12[0]["feature-names"].index(f) for f in features]
    X1 = np.full((N, len(cols)), np.nan)
    X2 = np.full((N, len(cols)), np.nan)
    T = np.full(N, np.nan)
    ID = np.zeros(N, dtype=int)
    Y = np.zeros(N, dtype=int)
    r, lost, events = 0, 0, 0
    for D in T12:
        ni = D["features"].shape[0]
        sl = np.arange(r, r + ni)
        X1[sl, :] = D["features"][:, cols]
        X2[sl, :] = D["features.2"][:, cols]
        T[sl] = D["time"]
        ID[sl] = int(D["shot"])
        if D.get("event", 0) > 0:
            events += 1
            t_end = D["time"][-1]
            if D["tb"] - t_end <= event_delta:
                Y[r + ni - 1] = int(D["event"])
            else:
                lost += 1
        r += ni
    assert r == N
    matched = np.all(np.isfinite(X2), axis=1)
    if verbose:
        print(f"  stacked {N} rows, {events} event shots")
        if lost:
            print(f"    {lost} events dropped: last equilibrium more than "
                  f"{1e3*event_delta:.2f} ms before the event")
        print(f"    {int(matched.sum())} rows matched in both trees "
              f"({100*matched.mean():.1f}%)")
    return X1, X2, T, ID, Y, matched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--primary", required=True, help="EPM pickle")
    p.add_argument("--secondary", required=True, help="EPQ pickle")
    p.add_argument("--out", required=True)
    p.add_argument("--delta", type=float, default=None,
                   help="matching window [s]; default dt/4 from the data")
    p.add_argument("--event-delta", type=float, default=None,
                   help="label tolerance [s]; default 2*dt from the data")
    p.add_argument("--drop", nargs="*", default=["error", "chisq"],
                   help="columns to exclude from the model features; these are "
                        "reconstruction quality, kept for screening only")
    a = p.parse_args()

    T1, T2 = load_records(a.primary), load_records(a.secondary)
    if not T1:
        raise SystemExit("primary is empty")
    print(f"  primary   {a.primary}: {len(T1)} shots, tag {T1[0]['tag']}")
    print(f"  secondary {a.secondary}: {len(T2)} shots, "
          f"tag {T2[0]['tag'] if T2 else '--'}")

    dts = [np.median(np.diff(r["time"])) for r in T1 if len(r["time"]) > 1]
    dt = float(np.median(dts)) if dts else EPM_NOMINAL_DT
    tol = derived_tolerances(dt)
    delta = a.delta if a.delta is not None else tol["merge_delta"]
    ev = a.event_delta if a.event_delta is not None else tol["event_delta"]
    print(f"  cadence dt = {1e3*dt:.3f} ms -> merge delta {1e3*delta:.2f} ms,"
          f" event_delta {1e3*ev:.2f} ms")

    T12 = create_merged_tables(T1, T2, delta)
    names = [f for f in T1[0]["feature-names"] if f not in a.drop]
    X1, X2, T, ID, Y, matched = table_stack(T12, names, ev)

    data = {"X1": X1, "X2": X2, "T": T, "ID": ID, "Y": Y, "matched": matched,
            "Xnames": names, "tag1": T1[0]["tag"],
            "tag2": T2[0]["tag"] if T2 else None, "dt": dt,
            "merge_delta": delta, "event_delta": ev,
            "dropped_columns": list(a.drop)}
    with open(a.out, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    E, n = int((Y > 0).sum()), len(Y)
    print(f"\n  {n} rows, {len(names)} features, {E} events, "
          f"base rate E/N = {E/max(n,1):.3e}")
    print(f"  matched subset: {int(matched.sum())} rows, "
          f"{int((Y[matched] > 0).sum())} events")
    print(f"  wrote {a.out}")
    print("\n  Both analyses run off this one file:")
    print("    full EPM     X1, every row")
    print("    matched pair X1 and X2 where `matched` -- the EPM/EPQ comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
