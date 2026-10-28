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


def match_slices(t1, t2, delta):
    """For each primary time, the index of the secondary slice within delta, or -1.

    Window +/- delta with delta = dt/4 admits at most one slice, so the first
    candidate is also the nearest.
    """
    t1, t2 = np.asarray(t1), np.asarray(t2)
    idx = np.full(t1.size, -1, dtype=int)
    for k, tk in enumerate(t1):
        cand = np.flatnonzero((t2 >= tk - delta) & (t2 <= tk + delta))
        if cand.size:
            idx[k] = cand[0]
    return idx


def create_merge(D1, D2, delta):
    """Secondary features on the primary's time base; NaN where unmatched.

    NaN here means one of two different things -- no secondary slice in the
    window, or a slice whose own feature is NaN.  `match_slices` tells them
    apart, and create_merged_tables records it as `has_secondary`.
    """
    assert D1["shot"] == D2["shot"], "shot numbers must match"
    assert D1["feature-names"] == D2["feature-names"], "feature sets must match"
    idx = match_slices(D1["time"], D2["time"], delta)
    X2 = np.full((idx.size, len(D1["feature-names"])), np.nan)
    ok = idx >= 0
    X2[ok, :] = np.asarray(D2["features"])[idx[ok], :]
    return X2


def create_merged_tables(T1, T2, delta, verbose=True):
    I2 = create_index(T2)
    tag2 = T2[0]["tag"] if T2 else None
    out, rows, no_slice, no_secondary = [], 0, 0, 0
    for D1 in T1:
        s = int(D1["shot"])
        if s in I2:
            D2 = T2[I2[s]]
            has = match_slices(D1["time"], D2["time"], delta) >= 0
            X12 = create_merge(D1, D2, delta)
        else:
            has = np.zeros(len(D1["time"]), dtype=bool)
            X12 = np.full(D1["features"].shape, np.nan)
            no_secondary += 1
        D = dict(D1)
        D["features.2"] = X12
        D["has_secondary"] = has
        D["tag.2"] = tag2
        out.append(D)
        rows += X12.shape[0]
        no_slice += int((~has).sum())
    if verbose:
        print(f"  {len(out)} shots, {rows} rows")
        print(f"    {no_secondary} shots have no secondary tree at all")
        print(f"    {no_slice} rows have no secondary slice in the window "
              f"({100*no_slice/max(rows,1):.1f}%) -- NaN in X2, not dropped")
    return out


def table_stack(T12, features, event_delta, verbose=True):
    """Flatten to X1, X2, T, ID, Y, matched, has_secondary, usable1.

    has_secondary  a secondary slice exists in the window
    matched        the usable PAIR: a secondary slice AND every selected feature
                   finite in both trees
    usable1        every selected feature finite in the primary alone -- the
                   row the full-EPM analysis can use

    `matched` used to be "X2 has no NaN", which could not tell a missing slice
    from a slice with one NaN feature.

    The label goes on the LAST row of an event shot, because the table was
    chopped at tevent during extraction, so that row IS the last pre-event
    slice.  It is only applied if that row sits within event_delta of the
    event: if the equilibrium stopped earlier -- common, since a growing
    island degrades the fit -- attaching an onset label to a reconstruction
    from long before would be false, so the event is dropped and counted.
    """
    if not any("event" in D for D in T12):
        raise SystemExit(
            "no record carries an 'event' key, so every shot would be treated "
            "as censored and Y would be all zeros.\n"
            "  Harvests made with --ladder are unlabelled by design; one made "
            "with --annotation should carry the label.")
    N = sum(len(e["time"]) for e in T12)
    cols = [T12[0]["feature-names"].index(f) for f in features]
    X1 = np.full((N, len(cols)), np.nan)
    X2 = np.full((N, len(cols)), np.nan)
    HAS = np.zeros(N, dtype=bool)
    T = np.full(N, np.nan)
    ID = np.zeros(N, dtype=int)
    Y = np.zeros(N, dtype=int)
    r, lost, events = 0, 0, 0
    for D in T12:
        ni = D["features"].shape[0]
        sl = np.arange(r, r + ni)
        X1[sl, :] = D["features"][:, cols]
        X2[sl, :] = D["features.2"][:, cols]
        HAS[sl] = D.get("has_secondary", np.isfinite(D["features.2"][:, 0]))
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
    usable1 = np.all(np.isfinite(X1), axis=1)
    matched = HAS & usable1 & np.all(np.isfinite(X2), axis=1)
    if verbose:
        print(f"  stacked {N} rows, {events} event shots")
        if lost:
            print(f"    {lost} events dropped: last equilibrium more than "
                  f"{1e3*event_delta:.2f} ms before the event")
        print(f"    {int(HAS.sum())} rows have a secondary slice "
              f"({100*HAS.mean():.1f}%)")
        print(f"    {int(usable1.sum())} rows usable in the primary alone, "
              f"{int(matched.sum())} in the pair")
    return X1, X2, T, ID, Y, matched, HAS, usable1


def accounting(names, X1, X2, Y, has2, usable1, matched, verbose=True):
    """Where rows and events are lost to NaN, which is otherwise silent.

    A NaN feature on the label row removes the event the moment the model
    filters NaN rows, with no message.  The count is made here instead, for the
    full-EPM analysis and for the pair separately.
    """
    ev = Y > 0
    out = {
        "rows": int(len(Y)),
        "nan_frac_X1": {n: float(np.mean(~np.isfinite(X1[:, i])))
                        for i, n in enumerate(names)},
        # NaN in X2 among rows that DO have a slice: a feature problem
        "nan_frac_X2_given_slice": {
            n: float(np.mean(~np.isfinite(X2[has2, i]))) if has2.any() else np.nan
            for i, n in enumerate(names)},
        "rows_slice_but_nan": int((has2 & ~matched & usable1).sum()),
        "events": int(ev.sum()),
        "events_usable_X1": int((ev & usable1).sum()),
        "events_usable_pair": int((ev & matched).sum()),
        "events_lost_nan_X1": int((ev & ~usable1).sum()),
        "events_lost_to_pairing": int((ev & usable1 & ~matched).sum()),
    }
    if verbose:
        print("\n  NaN fraction per feature (primary | secondary, given a slice):")
        for n in names:
            a, b = out["nan_frac_X1"][n], out["nan_frac_X2_given_slice"][n]
            flag = "   <-- check" if (a > 0.05 or (np.isfinite(b) and b > 0.05)) else ""
            print(f"    {n:8s} {100*a:6.2f}% | {100*b:6.2f}%{flag}")
        print(f"\n  events: {out['events']} labelled")
        print(f"    usable in the primary alone  {out['events_usable_X1']}"
              f"   (lost to NaN in the label row: {out['events_lost_nan_X1']})")
        print(f"    usable in the matched pair   {out['events_usable_pair']}"
              f"   (lost to the pairing: {out['events_lost_to_pairing']})")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--primary", required=True, help="EPM pickle")
    p.add_argument("--secondary", default=None,
                   help="EPQ pickle; omit to work on the primary alone (X2 is all NaN)")
    p.add_argument("--out", required=True)
    p.add_argument("--delta", type=float, default=None,
                   help="matching window [s]; default dt/4 from the data")
    p.add_argument("--event-delta", type=float, default=None,
                   help="label tolerance [s]; default 2*dt from the data")
    p.add_argument("--drop", nargs="*", default=["error", "chisq"],
                   help="columns to exclude from the model features; these are "
                        "reconstruction quality, kept for screening only")
    a = p.parse_args()

    T1 = load_records(a.primary)
    T2 = load_records(a.secondary) if a.secondary else []
    if not T1:
        raise SystemExit("primary is empty")
    print(f"  primary   {a.primary}: {len(T1)} shots, tag {T1[0]['tag']}")
    if a.secondary:
        print(f"  secondary {a.secondary}: {len(T2)} shots, "
              f"tag {T2[0]['tag'] if T2 else '--'}")
    else:
        print("  no secondary: primary alone, X2 is all NaN, nothing is paired")

    dts = [np.median(np.diff(r["time"])) for r in T1 if len(r["time"]) > 1]
    dt = float(np.median(dts)) if dts else EPM_NOMINAL_DT
    tol = derived_tolerances(dt)
    delta = a.delta if a.delta is not None else tol["merge_delta"]
    ev = a.event_delta if a.event_delta is not None else tol["event_delta"]
    print(f"  cadence dt = {1e3*dt:.3f} ms -> merge delta {1e3*delta:.2f} ms,"
          f" event_delta {1e3*ev:.2f} ms")

    T12 = create_merged_tables(T1, T2, delta)
    names = [f for f in T1[0]["feature-names"] if f not in a.drop]
    X1, X2, T, ID, Y, matched, has2, usable1 = table_stack(T12, names, ev)
    acct = accounting(names, X1, X2, Y, has2, usable1, matched)

    data = {"X1": X1, "X2": X2, "T": T, "ID": ID, "Y": Y, "matched": matched,
            "has_secondary": has2, "usable1": usable1, "accounting": acct,
            "Xnames": names, "tag1": T1[0]["tag"],
            "tag2": T2[0]["tag"] if T2 else None, "dt": dt,
            "merge_delta": delta, "event_delta": ev,
            "dropped_columns": list(a.drop)}
    with open(a.out, "wb") as fh:
        pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    E, n = int((Y > 0).sum()), len(Y)
    print(f"\n  {n} rows, {len(names)} features, {E} events, "
          f"base rate E/N = {E/max(n,1):.3e}")
    print(f"  usable (primary alone): {int(usable1.sum())} rows, "
          f"{int((Y[usable1] > 0).sum())} events")
    print(f"  matched pair:           {int(matched.sum())} rows, "
          f"{int((Y[matched] > 0).sum())} events")
    print(f"  wrote {a.out}")
    print("\n  Both analyses run off this one file:")
    print("    full EPM     X1 where `usable1` (no NaN feature)")
    print("    matched pair X1 and X2 where `matched` -- the EPM/EPQ comparison")
    print("  Rows are never dropped here; the masks do the selecting.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
