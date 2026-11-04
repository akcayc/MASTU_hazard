"""Distributions of every feature, to choose the screening limits from.

    python feature_diagnostics.py --features feat_EPM.pickle --outdir diag_EPM

Writes two figures and prints one table:

  <tag>_box.png   one box per feature, all rows (left) and the last pre-event
                  rows (right); whiskers at 1.5 IQR, fliers drawn
  <tag>_hist.png  histograms over each feature's full finite range with LOG
                  counts, so a handful of wild slices stays visible; event rows
                  overlaid; dotted grey at the 0.05% / 99.95% quantiles that
                  DIII-D's --limits auto uses, dashed red at DIII-D's fixed
                  limits (tabular_limits.STANDARD_FIXED_LIMITS) -- DIII-D
                  values, shown for reference only, not MAST-U limits
  stdout          per feature: rows, NaN, inf, min, quantiles, max, and the
                  rows outside each set of limits

Every row of the table is used, NaN rows included (counted, not plotted), and
the quality columns chisq and error are kept: they are what DIII-D screens on.
--table reads a merge_pair output instead; X1 (the primary tree) is shown.
"""

import argparse
import os
import pickle

import numpy as np

import merge_pair as MP

#: DIII-D's tabular_limits.STANDARD_FIXED_LIMITS, verbatim.  Rows pass when
#: x > min and x <= max.  Reference lines only.
DIIID_FIXED_LIMITS = {
    "chisq_max": 90.0, "error_max": 5.0e-3,
    "nhat_min": 0.0, "nhat_max": 2.0,
    "betap_min": 0.0, "betat_min": 0.0,
    "q0_min": 0.0, "q0_max": 10.0,
    "q95_min": 1.0, "q95_max": 20.0,
    "pres0_min": -20.0, "pres0_max": 50.0,
    "pres50_min": -20.0, "pres50_max": 50.0,
    "dpres50_min": -50.0, "dpres50_max": 20.0,
}
#: DIII-D's SCREEN_AUTO_QUANTILE_LOW / _HIGH
Q_LO, Q_HI = 0.0005, 0.9995


def load(path, table=False):
    if table:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
    else:
        d = MP.build_table(path, drop=())
    return (np.asarray(d["X1"], float), np.asarray(d["Y"]) > 0,
            list(d["Xnames"]), d.get("tag1") or "X1")


def stats(X, ev, names):
    rows = []
    for j, n in enumerate(names):
        x = X[:, j]
        fin = np.isfinite(x)
        r = {"name": n, "rows": x.size, "nan": int(np.isnan(x).sum()),
             "inf": int(np.isinf(x).sum()), "events_nan": int((ev & ~fin).sum())}
        if fin.any():
            v = x[fin]
            r.update(min=v.min(), max=v.max(),
                     q=np.quantile(v, [Q_LO, 0.01, 0.5, 0.99, Q_HI]))
            r["out_q"] = int(((v < r["q"][0]) | (v > r["q"][-1])).sum())
            lo = DIIID_FIXED_LIMITS.get(f"{n}_min", -np.inf)
            hi = DIIID_FIXED_LIMITS.get(f"{n}_max", np.inf)
            has = f"{n}_min" in DIIID_FIXED_LIMITS or f"{n}_max" in DIIID_FIXED_LIMITS
            r["out_d3d"] = int(((v <= lo) | (v > hi)).sum()) if has else None
            r["out_d3d_ev"] = (int((ev[fin] & ((v <= lo) | (v > hi))).sum())
                               if has else None)
        rows.append(r)
    return rows


def print_table(st, n_rows, n_ev):
    print(f"\n  {n_rows} rows, {n_ev} events\n")
    print(f"  {'feature':8s} {'NaN':>7s} {'inf':>4s} {'evNaN':>5s} "
          f"{'min':>10s} {'q0.05%':>10s} {'q1%':>10s} {'median':>10s} "
          f"{'q99%':>10s} {'q99.95%':>10s} {'max':>10s} {'>q':>6s} "
          f"{'D3D lim':>11s}")
    for r in st:
        head = f"  {r['name']:8s} {r['nan']:7d} {r['inf']:4d} {r['events_nan']:5d} "
        if "q" not in r:
            print(head + "  (no finite values)")
            continue
        q = r["q"]
        d3d = ("--" if r["out_d3d"] is None
               else f"{r['out_d3d']} ({r['out_d3d_ev']} ev)")
        print(head + f"{r['min']:10.3g} {q[0]:10.3g} {q[1]:10.3g} {q[2]:10.3g} "
              f"{q[3]:10.3g} {q[4]:10.3g} {r['max']:10.3g} {r['out_q']:6d} "
              f"{d3d:>11s}")
    print("\n  >q: finite rows outside [q0.05%, q99.95%]; D3D lim: rows a DIII-D "
          "fixed limit would remove (of them, event rows)")


def grid(n):
    c = int(np.ceil(np.sqrt(n * 1.5)))
    return int(np.ceil(n / c)), c


def plot_box(X, ev, names, path, title):
    import matplotlib.pyplot as plt
    nr, nc = grid(len(names))
    fig, ax = plt.subplots(nr, nc, figsize=(2.2 * nc, 2.6 * nr), squeeze=False)
    for j, n in enumerate(names):
        a = ax.flat[j]
        x = X[:, j]
        fin = np.isfinite(x)
        data = [x[fin], x[fin & ev]]
        if data[0].size:
            a.boxplot([d if d.size else [np.nan] for d in data],
                      tick_labels=["all", "event"], widths=0.6,
                      flierprops={"markersize": 2, "alpha": 0.3})
        a.set_title(n, fontsize=9)
        a.tick_params(labelsize=7)
    for a in ax.flat[len(names):]:
        a.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_hist(X, ev, names, path, title, bins=80):
    import matplotlib.pyplot as plt
    nr, nc = grid(len(names))
    fig, ax = plt.subplots(nr, nc, figsize=(2.8 * nc, 2.2 * nr), squeeze=False)
    for j, n in enumerate(names):
        a = ax.flat[j]
        x = X[:, j]
        fin = np.isfinite(x)
        a.set_title(n, fontsize=9)
        a.tick_params(labelsize=7)
        if not fin.any():
            a.text(0.5, 0.5, "no finite values", ha="center", transform=a.transAxes)
            continue
        v = x[fin]
        lo, hi = v.min(), v.max()
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, bins + 1)
        a.hist(v, edges, color="C0", alpha=0.8, label="all")
        if (fin & ev).any():
            a.hist(x[fin & ev], edges, color="C3", alpha=0.9, label="event")
        a.set_yscale("log")
        for q in np.quantile(v, [Q_LO, Q_HI]):
            a.axvline(q, color="0.4", ls=":", lw=1)
        for k in ("_min", "_max"):
            L = DIIID_FIXED_LIMITS.get(n + k)
            if L is not None and lo <= L <= hi:
                a.axvline(L, color="r", ls="--", lw=1)
    ax.flat[0].legend(fontsize=6)
    for a in ax.flat[len(names):]:
        a.axis("off")
    fig.suptitle(title + "   (log counts; dotted: 0.05/99.95% quantiles, "
                 "dashed red: DIII-D fixed limits)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", help="harvest pickle (feat_EPM.pickle)")
    src.add_argument("--table", help="merge_pair output; X1 is shown")
    p.add_argument("--outdir", default=".")
    p.add_argument("--bins", type=int, default=80)
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    path = a.features or a.table
    X, ev, names, tag = load(path, table=bool(a.table))
    print_table(stats(X, ev, names), X.shape[0], int(ev.sum()))

    os.makedirs(a.outdir, exist_ok=True)
    title = f"{tag}: {os.path.basename(path)}, {X.shape[0]} rows, {int(ev.sum())} events"
    for kind, fn in (("box", plot_box), ("hist", plot_hist)):
        out = os.path.join(a.outdir, f"{tag}_{kind}.png")
        if kind == "hist":
            fn(X, ev, names, out, title, a.bins)
        else:
            fn(X, ev, names, out, title)
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
