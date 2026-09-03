"""Collapse a ladder CSV into events / dwell / base hazard per threshold.

The DIII-D analogue of brmEventsCounter.py.  This is how you read a ladder:
the ladder itself is one first-crossing time per shot per level; this reduces
it to the curve you actually judge a threshold on.

    python ladder_base_rate.py rm_ladder.csv
    python ladder_base_rate.py a.csv b.csv --labels ungated gated

Look for a KNEE -- a level where the event fraction falls off sharply.  Above
it the detector is finding modes; below it, noise.  No knee means the gate is
not separating anything, and the threshold cannot be chosen on evidence.
"""

import argparse
import numpy as np
import pandas as pd


def curve(df):
    cols = sorted([c for c in df.columns if c.startswith("rm")],
                  key=lambda c: float(c[2:]))
    dwell = float((df["tb_rm"] - df["ta_rm"]).sum())
    rows = []
    for c in cols:
        n = int(np.isfinite(df[c]).sum())
        rows.append((float(c[2:]), n, n / len(df), n / dwell if dwell else np.nan))
    return rows, dwell


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--labels", nargs="+")
    a = p.parse_args()
    labels = a.labels or [f.rsplit("/", 1)[-1] for f in a.csv]

    for path, label in zip(a.csv, labels):
        df = pd.read_csv(path)
        rows, dwell = curve(df)
        print(f"\n=== {label} ===")
        print(f"  {len(df)} shots, {dwell:.2f} s total dwell")
        if "truncated" in df:
            t = int(df["truncated"].sum())
            if t:
                print(f"  {t} shots truncated by the record, "
                      f"{df['lost_exposure'].sum():.2f} s exposure lost")
        print(f"\n  {'level':>8} {'events':>7} {'frac':>7} {'hazard 1/s':>11}   knee")
        prev = None
        for lvl, n, frac, haz in rows:
            drop = "" if prev is None or prev == 0 else \
                ("  <<<" if (prev - n) / prev > 0.30 else "")
            print(f"  {lvl:8.4f} {n:7d} {frac:7.3f} {haz:11.4f}{drop}")
            prev = n
        print("\n  <<< marks a >30% fall from the previous level.")

    if len(a.csv) == 2:
        (r0, _), (r1, _) = curve(pd.read_csv(a.csv[0])), curve(pd.read_csv(a.csv[1]))
        print(f"\n=== {labels[0]} -> {labels[1]} ===")
        print(f"  {'level':>8} {'before':>7} {'after':>7} {'removed':>8}")
        for (lvl, n0, _, _), (_, n1, _, _) in zip(r0, r1):
            print(f"  {lvl:8.4f} {n0:7d} {n1:7d} {n0-n1:8d}")
        print("\n  A gate should only ever REMOVE crossings.  Negative means a bug.")


if __name__ == "__main__":
    raise SystemExit(main())
