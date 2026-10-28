"""Ladder CSV -> joint annotation CSV, the input to extract_features.py.

The ladder holds one first-crossing time per shot per threshold.  Choosing a
threshold selects one column, and that column IS the survival table: a finite
crossing is an event at that time, a NaN is a shot censored at end of window.
This script performs that selection and writes it in the 7-column format the
DIII-D feature extractor expects.

    shot, tsof, teof, trm, tlm, tevent, ievent

    tevent = min(teof, trm, tlm)          over the finite entries
    ievent = 0 censored | 1 rotating | 2 locked

Two further columns follow when the ladder carries them: fa_event, the mode
frequency at the crossing, and coh_event, the coherence there.  They cost
nothing to carry and they are what tells a rotating event from a locked one.

With no locked-mode detector, tlm is always NaN and ievent is 0 or 1.  The
schema keeps the column so the LM branch can be added without a format change.

    python make_annotation.py rm_ladder.csv --threshold 0.0137 --out joint.csv
        -> writes joint_threshold0.0137.csv

The threshold actually applied (the ladder column it snaps to, 5 significant
digits like MATLAB's num2str) is appended to the --out name, since nothing
inside the CSV records it.
"""

import argparse
import os
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


def threshold_path(out, threshold):
    """joint.csv, 0.0137 -> joint_threshold0.0137.csv"""
    root, ext = os.path.splitext(out)
    return f"{root}_threshold{threshold:.5g}{ext or '.csv'}"


def build(df, col, eq_start, fa_col=None, co_col=None):
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

        row = dict(shot=shot, tsof=tsof, teof=teof, trm=trm, tlm=tlm,
                   tevent=tevent, ievent=ievent)
        # only meaningful for a rotating event: for a censored shot tevent is
        # just the end of the window, where no crossing happened
        row["fa_event"] = float(r[fa_col]) if (fa_col and ievent == 1) else np.nan
        row["coh_event"] = float(r[co_col]) if (co_col and ievent == 1) else np.nan
        rows.append(row)
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
    fa_col = col.replace("rm", "fa", 1) if col.replace("rm", "fa", 1) in df.columns else None
    co_col = col.replace("rm", "co", 1) if col.replace("rm", "co", 1) in df.columns else None
    if fa_col is None:
        print("  note: ladder has no fa*/co* columns (pre-dates them); "
              "fa_event and coh_event will be empty")
    ann, skipped, n_before_eq = build(df, col, a.eq_start, fa_col, co_col)

    if ann.empty:
        raise SystemExit("no usable shots")
    ann = ann.sort_values("shot")
    out = threshold_path(a.out, lv)
    # the first seven columns are the format extract_features reads; the two
    # that follow are extra and are ignored by anything that does not want them
    ann.to_csv(out, index=False,
               columns=["shot", "tsof", "teof", "trm", "tlm", "tevent", "ievent",
                        "fa_event", "coh_event"])

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
    fa = ann.loc[ann["ievent"] > 0, "fa_event"]
    fa = fa[np.isfinite(fa)]
    if fa.size:
        q1, q3 = np.percentile(fa / 1e3, [25, 75])
        print(f"  Fa at onset   {np.median(fa)/1e3:.2f} kHz  "
              f"IQR [{q1:.2f}, {q3:.2f}]  over {fa.size} events")
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
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
