"""Figures for a ladder scan: events vs threshold, and where the knee is.

    python plot_ladder.py rm_ladder_g0.55.csv rm_ladder_g0.60.csv rm_ladder_g0.70.csv \
        --labels "gate 0.55" "gate 0.60" "gate 0.70" --out ladder

Two panels, deliberately:

  events vs threshold   the curve itself, log-log.
  local slope           d(log N)/d(log A).  A noise-dominated population falls
                        with a roughly constant log-log slope, because noise has
                        no characteristic amplitude.  A real population puts a
                        BREAK in that slope.  This panel is what turns "I think
                        I see a knee" into a located number.

Base hazard is not a third panel: it is events/dwell, a constant rescale of
panel one, so it would be the same curve in different units.  It is printed
instead.

Writes <out>.png, <out>.pdf and <out>_table.csv -- the table because the
figure alone is not an accessible artifact.
"""

import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # cluster: no display
import matplotlib.pyplot as plt

# Categorical slots of the validated reference palette, in fixed order, never
# cycled.  Line charts are scored on the ADJACENT pairlist, on which all eight
# slots pass in both modes (worst adjacent CVD dE 9.1 light / 8.4 dark, normal
# vision 19.6 / 19.3).  The tighter all-pairs cap of three applies to scatter,
# bubble and small multiples, not to this form.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

#: direct labels stay legible up to four curves; past that the legend carries it
MAX_DIRECT_LABELS = 4

#: type sizes in points; --font-scale multiplies all of them together
FS = dict(title=17.0, subtitle=12.0, axis=13.0, tick=11.5, minor=10.0,
          legend=12.5, direct=12.5, note=11.5)
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e4e3de"


def curve(df):
    """(levels, events, dwell) from a ladder CSV."""
    cols = sorted([c for c in df.columns if c.startswith("rm")],
                  key=lambda c: float(c[2:]))
    lv = np.array([float(c[2:]) for c in cols])
    ev = np.array([int(np.isfinite(df[c]).sum()) for c in cols], float)
    dwell = float((df["tb_rm"] - df["ta_rm"]).sum())
    return lv, ev, dwell


def local_slope(lv, ev, win=5):
    """d(log N)/d(log A), smoothed over `win` levels.  NaN where N == 0."""
    ok = ev > 0
    s = np.full(lv.size, np.nan)
    if ok.sum() < 3:
        return s
    x, y = np.log10(lv[ok]), np.log10(ev[ok])
    g = np.gradient(y, x)
    if win > 1 and g.size >= win:
        k = np.ones(win) / win
        g = np.convolve(g, k, mode="same")
        edge = win // 2
        g[:edge] = np.nan
        g[g.size - edge:] = np.nan
    s[ok] = g
    return s


def find_knee(lv, ev, slope, excess=1.0):
    """Where the curve departs from its low-threshold plateau.

    At low thresholds nearly every shot crosses, so the count saturates at the
    shot total and the log-log slope sits near zero.  The knee is where the
    threshold starts to discriminate -- the FIRST level whose slope is `excess`
    steeper than the plateau -- not the bottom of the subsequent dive, which is
    where an argmin of the slope difference would land.

    Returns (onset, steepest) in gauss, either may be None.
    """
    ok = np.isfinite(slope)
    if ok.sum() < 4:
        return None, None
    idx = np.flatnonzero(ok)
    plateau = float(np.median(slope[idx[:max(3, idx.size // 3)]]))
    below = idx[slope[idx] < plateau - excess]
    onset = float(lv[below[0]]) if below.size else None
    steepest = float(lv[idx[int(np.argmin(slope[idx]))]])
    return onset, steepest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+")
    p.add_argument("--labels", nargs="+")
    p.add_argument("--out", default="ladder")
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--font-scale", type=float, default=1.0,
                   help="multiply every type size (default 1.0)")
    p.add_argument("--excess", type=float, default=1.0,
                   help="how much steeper than the plateau counts "
                        "as the knee, in log-log slope units")
    a = p.parse_args()
    labels = a.labels or [f.rsplit("/", 1)[-1].replace(".csv", "") for f in a.csv]
    fs = {k: v * a.font_scale for k, v in FS.items()}
    if len(a.csv) > len(SERIES):
        raise SystemExit(
            f"{len(a.csv)} curves requested but the categorical order has "
            f"{len(SERIES)} slots, and a 9th series is never a generated hue. "
            f"Fold the rest into a representative subset, or run the script "
            f"twice and show the panels side by side.")
    direct = len(a.csv) <= MAX_DIRECT_LABELS

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10.0, 9.4), sharex=True,
        gridspec_kw=dict(height_ratios=[2.1, 1.0], hspace=0.14))
    fig.patch.set_facecolor(SURFACE)
    # right margin leaves room for the direct labels at the larger type size
    fig.subplots_adjust(left=0.125, right=0.760, top=0.878, bottom=0.088)

    rows, nshots, knees, ends = [], None, {}, []
    for i, (path, lab) in enumerate(zip(a.csv, labels)):
        df = pd.read_csv(path)
        nshots = len(df)
        lv, ev, dwell = curve(df)
        sl = local_slope(lv, ev, a.smooth)
        kn, steep = find_knee(lv, ev, sl, a.excess)
        knees[lab] = kn
        c = SERIES[i]

        m = ev > 0
        ax1.plot(lv[m], ev[m], color=c, lw=2.4, solid_capstyle="round", zorder=3)
        off = (i * 2) % 6
        ax1.plot(lv[m][off::6], ev[m][off::6], "o", color=c, ms=5.5, mec=SURFACE,
                 mew=1.3, zorder=4, label=lab)
        ax2.plot(lv, sl, color=c, lw=2.4, solid_capstyle="round", zorder=3)

        if m.any() and direct:            # deferred: placed after, destaggered
            ends.append([lv[m][-1], ev[m][-1], lab, c])
        if kn is not None:
            al = 0.55 if direct else 0.30
            ax1.axvline(kn, color=c, lw=1.0, ls=(0, (4, 3)), alpha=al, zorder=1)
            ax2.axvline(kn, color=c, lw=1.0, ls=(0, (4, 3)), alpha=al, zorder=1)

        base = ev / dwell if dwell else np.full_like(ev, np.nan)
        for j in range(lv.size):
            rows.append(dict(series=lab, threshold_G=lv[j], events=int(ev[j]),
                             shots=nshots, dwell_s=round(dwell, 3),
                             base_hazard_per_s=round(float(base[j]), 5),
                             log_slope=round(float(sl[j]), 3) if np.isfinite(sl[j]) else ""))
        n_at = int(ev[np.argmin(np.abs(lv - kn))]) if kn else 0
        print(f"{lab:>12}: {nshots} shots, {dwell:7.2f} s dwell,  "
              f"knee {('%.4f G' % kn) if kn else '  none  '} "
              f"({n_at:3d} events there), steepest at "
              f"{('%.4f G' % steep) if steep else 'n/a'}")

    # direct labels (relief rule): nudge apart when curves end together
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.autoscale_view()
    if ends:
        ends.sort(key=lambda e: e[1])
        inv, fwd = ax1.transData.inverted(), ax1.transData
        ypix = [fwd.transform((x, y))[1] for x, y, _, _ in ends]
        gap = fs["direct"] * 1.55              # separation scales with the type
        for i in range(1, len(ypix)):
            if ypix[i] - ypix[i - 1] < gap:
                ypix[i] = ypix[i - 1] + gap
        for (x, y, lab, c), yp in zip(ends, ypix):
            yd = inv.transform((0, yp))[1]
            # colored mark carries identity; the text stays in ink
            ax1.plot([x], [yd], "o", color=c, ms=6.5, mec=SURFACE, mew=1.3,
                     clip_on=False, zorder=5)
            ax1.annotate(lab, (x, yd), xytext=(11, 0), textcoords="offset points",
                         color=INK2, fontsize=fs["direct"], va="center",
                         annotation_clip=False)
    ax1.set_ylabel("events  (shots with a crossing)", color=INK2,
                   fontsize=fs["axis"])
    fig.text(0.125, 0.958, "Tearing-mode events vs threshold amplitude",
             color=INK, fontsize=fs["title"], fontweight="semibold", ha="left")
    fig.text(0.125, 0.922, f"n=1 OMAHA ladder, {nshots} MAST-U shots - "
             f"dashed lines mark the plateau departure",
             color=INK3, fontsize=fs["subtitle"], ha="left")
    leg = ax1.legend(frameon=False, loc="lower left", fontsize=fs["legend"],
                     labelcolor=INK2, handletextpad=0.7,
                     bbox_to_anchor=(0.015, 0.04),
                     ncol=1 if len(a.csv) <= 4 else 2,
                     columnspacing=1.6)
    leg.set_zorder(5)

    ax2.axhline(0, color=GRID, lw=1.0, zorder=1)
    ax2.set_ylabel("local slope\nd log N / d log A", color=INK2,
                   fontsize=fs["axis"])
    ax2.set_xlabel("threshold amplitude  [gauss]", color=INK2,
                   fontsize=fs["axis"], labelpad=8)
    ax2.text(0.0, 1.05, "flat = saturated; the break is where the threshold "
             "starts to discriminate",
             transform=ax2.transAxes, color=INK3, fontsize=fs["note"])

    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=fs["tick"], which="both")
    ax2.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    ax2.xaxis.set_minor_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}" if v in (0.002, 0.005, 0.02, 0.05, 0.2) else ""))
    ax2.tick_params(axis="x", which="minor", labelsize=fs["minor"], colors=INK3)

    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=170, facecolor=SURFACE,
                    bbox_inches="tight")
    pd.DataFrame(rows).to_csv(f"{a.out}_table.csv", index=False)
    print(f"\nwrote {a.out}.png  {a.out}.pdf  {a.out}_table.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
