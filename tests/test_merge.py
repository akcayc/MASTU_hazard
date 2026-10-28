"""merge_pair: alignment, unmatched handling, label placement, the matched mask."""
import _paths  # noqa: F401
import numpy as np

from merge_pair import (create_merge, create_merged_tables, table_stack,
                        derived_tolerances)

DT = 5.025e-3
NAMES = ["a", "b", "error"]


def rec(shot, t0, n, tag, event=0, tb=None, offset=0.0, dt=DT):
    t = t0 + dt * np.arange(n)
    X = np.column_stack([np.arange(n) + offset, 10 * np.arange(n) + offset,
                         np.zeros(n)])
    return {"shot": shot, "tag": tag, "time": t, "features": X,
            "feature-names": NAMES, "event": event,
            "tb": tb if tb is not None else t[-1]}


def main():
    ok = True
    tol = derived_tolerances(DT)

    def chk(label, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")

    print(f"  dt = {1e3*DT:.3f} ms -> merge delta {1e3*tol['merge_delta']:.2f} ms,"
          f" event_delta {1e3*tol['event_delta']:.2f} ms")

    # exact alignment: every row pairs
    A = rec(1, 0.10, 20, "EPM")
    B = rec(1, 0.10, 20, "EPQ", offset=100.0)
    X2 = create_merge(A, B, tol["merge_delta"])
    chk("aligned grids pair every row", np.isfinite(X2).all())
    chk("paired values come from the secondary", X2[3, 0] == 103.0)

    # offset by half a slice: outside a dt/4 window, so nothing pairs
    C = rec(1, 0.10 + DT / 2, 20, "EPQ", offset=100.0)
    X2b = create_merge(A, C, tol["merge_delta"])
    chk("half-slice offset pairs nothing (window is dt/4)",
        not np.isfinite(X2b).any())

    # at most one candidate can fall in a +/- dt/4 window, so "earliest" and
    # "nearest" cannot disagree -- the ambiguity in the DIII-D original is moot
    D = rec(1, 0.10 + DT / 8, 20, "EPQ", offset=100.0)
    cand = [np.sum((D["time"] >= tk - tol["merge_delta"]) &
                   (D["time"] <= tk + tol["merge_delta"])) for tk in A["time"]]
    chk(f"never more than one candidate in the window (max {max(cand)})",
        max(cand) <= 1)

    # a shot missing from the secondary keeps its primary rows, X2 all NaN
    T1 = [rec(1, 0.10, 20, "EPM", event=1),
          rec(2, 0.10, 20, "EPM", event=0)]
    T2 = [rec(1, 0.10, 20, "EPQ", offset=100.0)]
    T12 = create_merged_tables(T1, T2, tol["merge_delta"], verbose=False)
    chk("primary rows all kept", sum(len(d["time"]) for d in T12) == 40)
    chk("shot absent from secondary is all-NaN, not dropped",
        not np.isfinite(T12[1]["features.2"]).any())

    X1, X2c, T, ID, Y, matched = table_stack(
        T12, ["a", "b"], tol["event_delta"], verbose=False)
    chk("label on the LAST row of the event shot", Y[19] == 1 and Y[:19].sum() == 0)
    chk("censored shot carries no label", Y[20:].sum() == 0)
    chk("matched mask marks only the paired shot",
        matched[:20].all() and not matched[20:].any())
    chk("error column dropped from the model features", X1.shape[1] == 2)

    # an event whose equilibrium stopped too early must be dropped, not mislabelled
    late = rec(3, 0.10, 20, "EPM", event=1)
    late["tb"] = late["time"][-1] + 5 * DT          # 5 slices past the last row
    T12b = create_merged_tables([late], [], tol["merge_delta"], verbose=False)
    _, _, _, _, Yb, _ = table_stack(T12b, ["a", "b"], tol["event_delta"],
                                    verbose=False)
    chk("event dropped when the last equilibrium is too far before it",
        Yb.sum() == 0)

    # an unlabelled harvest must be refused, not silently all-censored
    noev = rec(5, 0.10, 20, "EPM")
    del noev["event"]
    T12d = create_merged_tables([noev], [], tol["merge_delta"], verbose=False)
    try:
        table_stack(T12d, ["a", "b"], tol["event_delta"], verbose=False)
        chk("unlabelled table refused", False)
    except SystemExit:
        chk("unlabelled table refused rather than all-censored", True)

    near = rec(4, 0.10, 20, "EPM", event=2)
    near["tb"] = near["time"][-1] + 1.5 * DT        # inside 2*dt
    T12c = create_merged_tables([near], [], tol["merge_delta"], verbose=False)
    _, _, _, _, Yc, _ = table_stack(T12c, ["a", "b"], tol["event_delta"],
                                    verbose=False)
    chk("event kept when inside the tolerance, type preserved", Yc[-1] == 2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
