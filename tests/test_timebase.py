"""probe_timebase.analyse on a time vector shaped like the OMAHA slow channel.

Coarse stamps before and after a dense 200 kHz section, with a hole punched
inside the dense section: the case where Gio's contiguous slice silently keeps
missing time.
"""
import _paths  # noqa: F401
import numpy as np
from probe_timebase import analyse

DT = 5.0e-6


def build(hole=True):
    pre = np.arange(-2.5, -0.1, 0.01)                    # coarse, before
    dense = np.arange(-0.1, 1.0, DT)
    post = np.arange(1.0 + 0.01, 6.0, 0.01)              # coarse, after
    if hole:
        dense = dense[(dense < 0.40) | (dense > 0.402)]  # 2 ms missing
    return np.concatenate([pre, dense, post])


def main():
    ok = True

    r = analyse(build(hole=False))
    g = r["gio"]
    checks = [
        ("nominal dt found", abs(r["dt_nom"] - DT) < 1e-9),
        ("slice starts at the dense section", abs(g["t0"] - (-0.1)) < 2 * DT),
        # time[idt[0]:idt[-1]] stops a few samples short of the last dense one
        ("slice ends at the dense section", abs(g["t1"] - 1.0) < 5 * DT),
        ("clean record: nothing hidden", g["hidden_n"] == 0 and g["backwards"] == 0),
        ("clean record: Fs error ~ 0", abs(g["fs_error"]) < 1e-6),
        ("coarse intervals counted, outside the dense section",
         r["counts"]["coarse"] > 400),
    ]
    for label, good in checks:
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label}")

    r = analyse(build(hole=True))
    g = r["gio"]
    checks = [
        ("hole detected inside the slice", g["hidden_n"] == 1),
        ("hole is ~2 ms", abs(g["hidden_time"] - 2.0e-3) < 2 * DT),
        ("hole is located at 0.40 s",
         g["hidden_where"] and abs(g["hidden_where"][0] - 0.40) < 1e-3),
        ("Fs as Gio computes it is biased low", g["fs_error"] < -5e-4),
        ("the bias is the missing time: ~ -dt_hole / T",
         abs(g["fs_error"] + g["hidden_time"] / (g["t1"] - g["t0"])) < 2e-4),
    ]
    for label, good in checks:
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {label}")

    # a backwards step must be reported, not averaged away
    t = build(hole=False)
    i = int(np.searchsorted(t, 0.5))
    t[i + 5] = t[i + 3]                                   # stamp goes back
    g = analyse(t)["gio"]
    good = g["backwards"] >= 1
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  non-monotonic stamp reported")

    # the histogram accounts for every interval
    r = analyse(build(hole=True))
    n = sum(h["n"] for h in r["hist"])
    good = n == r["n"] - 1
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  histogram holds every interval "
          f"({n} of {r['n'] - 1})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
