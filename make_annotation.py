"""Ladder CSV -> joint annotation CSV, the input to extract_features.py.

The ladder holds one first-crossing time per shot per threshold.  Choosing a
threshold selects one column, and that column IS the survival table: a finite
crossing is an event at that time, a NaN is a shot censored at end of window.
This script performs that selection and writes it in the 7-column format the
DIII-D feature extractor expects.

    shot, tsof, teof, trm, tlm, tevent, ievent

    tevent = min(teof, trm, tlm)          over the finite entries
    ievent = 0 censored | 1 rotating | 2 locked

With no locked-mode detector, tlm is always NaN and ievent is 0 or 1.  The
schema keeps the column so the LM branch can be added without a format change.

    python make_annotation.py rm_ladder.csv --threshold 0.0137 --out joint.csv
"""

import argparse
import numpy as np
import pandas as pd

#: EPM opens at 0.020 s while the OMAHA record opens at 0.001 s.  An event
#: before this has no equilibrium to attach to, so it is counted and reported.
EQ_START_DEFAULT = 0.020


def pick_column(df, threshold):
    """Nearest ladder column to `threshold` [gauss]."""
    cols = [c for c in df.columns if c.startswith("rm")]
    if not cols:
        raise SystemExit("no rm* columns -- is this a ladder CSV?")
    lv = np.array([float(c[2:]) for c in cols])
    i = int(np.argmin(np.abs(lv - threshold)))
    return cols[i], float(lv[i])


def build(df, col, eq_start):
    rows, skipped = [], {}
    n_before_eq = 0
    for _, r in df.iterrows():
        shot = int(r["shot"])
        tsof, teof = float(r["ta_rm"]), float(r["tb_rm"])
        trm = float(r[col])
        tlm = np.nan                      # no locked-mode detector yet

        if not (teof > tsof):
            skipped[shot] = f"window empty: ta={tsof:.4f} tb={teof:.4f}"
            continue
        # a crossing at or before the window start cannot be dated; the
        # detector only triggers inside [ta, tb], so this means the mode was
        # already present and the shot carries no pre-onset history.
        if np.isfinite(trm) and trm <= tsof:
            skipped[shot] = f"crossing at/before window start ({trm:.4f} <= {tsof:.4f})"
            continue
        if np.isfinite(trm) and trm < eq_start:
            n_before_eq += 1

        cand = [(teof, 0), (trm, 1), (tlm, 2)]
        cand = [(t, i) for t, i in cand if np.isfinite(t)]
        tevent, ievent = min(cand, key=lambda x: x[0])

        rows.append(dict(shot=shot, tsof=tsof, teof=teof, trm=trm, tlm=tlm,
                         tevent=tevent, ievent=ievent))
    return pd.DataFrame(rows), skipped, n_before_eq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ladder")
    p.add_argument("--threshold", type=float, required=True,
                   help="detection threshold in gauss; snapped to the nearest "
                        "ladder column")
    p.add_argument("--out", required=True)
    p.add_argument("--eq-start", type=float, default=EQ_START_DEFAULT,
                   help="when the equilibrium record opens [s]")
    a = p.parse_args()

    df = pd.read_csv(a.ladder)
    col, lv = pick_column(df, a.threshold)
    if abs(lv - a.threshold) > 1e-9:
        print(f"threshold {a.threshold:g} G -> nearest column {col} ({lv:.5f} G)")
    ann, skipped, n_before_eq = build(df, col, a.eq_start)

    if ann.empty:
        raise SystemExit("no usable shots")
    ann = ann.sort_values("shot")
    ann.to_csv(a.out, index=False,
               columns=["shot", "tsof", "teof", "trm", "tlm", "tevent", "ievent"])

    E = int((ann["ievent"] > 0).sum())
    N = len(ann)
    dwell = float((ann["tevent"] - ann["tsof"]).sum())
    print(f"\n{N} shots, {E} events, {N - E} censored")
    print(f"  threshold      {lv:.5f} G  ({col})")
    print(f"  total exposure {dwell:.2f} s")
    print(f"  event fraction {E / N:.3f}   crude rate {E / dwell:.4f} /s")
    if E:
        te = ann.loc[ann["ievent"] > 0, "tevent"]
        print(f"  event times    {te.min():.4f} .. {te.max():.4f} s "
              f"(median {te.median():.4f})")
    if n_before_eq:
        print(f"  WARNING {n_before_eq} events before the equilibrium record "
              f"opens at {a.eq_start:.3f} s -- no features to attach")
    if skipped:
        print(f"  {len(skipped)} shots skipped:")
        for s, why in list(skipped.items())[:8]:
            print(f"    {s}: {why}")

    # The row count the model will actually see, and hence the base rate, is
    # set by the equilibrium cadence rather than by anything here.
    print(f"\n  at a 5.03 ms equilibrium cadence this is roughly "
          f"{int(dwell / 5.025e-3)} rows, base rate E/N ~ "
          f"{E / max(dwell / 5.025e-3, 1):.2e}")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
