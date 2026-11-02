"""Is the |n|=2 ladder seeing n=1 leaking through?

On the ~60 deg OMAHA array a pure n=1 mode projects onto n=2 with coherence
0.866 (and n=2 onto n=1 likewise).  Each bin goes to the n with the highest
coherence, so a clean mode is assigned correctly, but noise can flip a bin
and put n=1 power into the |n|=2 trace.  Raising the gate does not isolate
this: the argmax comes before the gate, so a flipped bin can still pass.

The ladders already carry what tells the two apart.  Each records Fa, the
centroid frequency at the crossing.  For a rigidly rotating plasma f ~ n f_rot:

    leaked n=1     the |n|=2 crossing sits at the SAME Fa as the |n|=1 one,
                   at about the same time                   ratio ~ 1
    genuine n=2    about TWICE the |n|=1 Fa                 ratio ~ 2

Two comparisons, because the |n|=1 threshold is several times the |n|=2 one:

    at the |n|=a threshold   |n|=a at its own knee against |n|=b at its own:
                             shots that are events in both ladders
    at the |n|=b threshold   |n|=a read at the SAME level as |n|=b.  Leaked
                             power is a fraction of the n=1 amplitude, so a
                             leaking n=1 also crosses this lower level; it
                             catches |n|=b events whose |n|=a is below its knee

Only first crossings are in a ladder: an |n|=a crossing long before the |n|=b
one says nothing about the later time, hence the --dt-window requirement.
Fa is a power-weighted centroid over the gated band, not one line, so the
ratio clusters are broad.  Confirm candidates with
plot_mode_track.py --shot <N> --n-list 1 2.

    python check_n_leakage.py 'GATE-080/rm_ladder_g0.8_n{n}.csv' --n-pair 1 2
    python check_n_leakage.py a_n1.csv a_n2.csv --thresholds 0.017 0.0043 \\
        --out leak --plot

Thresholds default to each ladder's knee, located as plot_ladder does.
"""

import argparse

import numpy as np
import pandas as pd

from make_annotation import pick_column
from plot_ladder import (curve, local_slope, find_knee, SERIES, INK, INK2,
                         INK3, SURFACE, GRID)


def knee_of(df, smooth, excess):
    lv, ev, _ = curve(df)
    kn, _ = find_knee(lv, ev, local_slope(lv, ev, smooth), excess)
    return kn


def crossings(df, threshold):
    """(shot-indexed frame of t, fa at the snapped level, the level)."""
    col, lv = pick_column(df, threshold)
    fa = col.replace("rm", "fa", 1)
    if fa not in df.columns:
        raise SystemExit(f"no {fa} column: this ladder pre-dates the fa*/co* "
                         f"columns, so there is no frequency to compare")
    out = pd.DataFrame({"t": df[col].to_numpy(float),
                        "fa": df[fa].to_numpy(float)},
                       index=df["shot"].astype(int))
    return out, lv


def ratio_class(r, tol, genuine=2.0):
    """'~1', '~<genuine>' or 'other' per frequency ratio; '' where undefined.

    genuine = nb/na, the ratio a real |n|=b mode has to |n|=a under rigid
    rotation: 2 for the 1-2 pair.
    """
    lr = np.log(np.asarray(r, float))
    w = np.log1p(tol)
    cls = np.where(np.abs(lr) <= w, "~1",
                   np.where(np.abs(lr - np.log(genuine)) <= w, f"~{genuine:g}",
                            "other"))
    return np.where(np.isfinite(lr), cls, "")


def compare(A, B, tol, dt_win, genuine=2.0):
    """Per-shot table for shots that cross both A and B."""
    j = A.join(B, lsuffix="_a", rsuffix="_b", how="inner")
    j = j[np.isfinite(j["t_a"]) & np.isfinite(j["t_b"])].copy()
    j["ratio"] = j["fa_b"] / j["fa_a"]
    j["dt"] = j["t_b"] - j["t_a"]
    j["class"] = ratio_class(j["ratio"], tol, genuine)
    j["leak"] = (j["class"] == "~1") & (np.abs(j["dt"]) <= dt_win)
    return j


def report(title, j, na, nb, n_b_events, dt_win):
    print(f"\n  {title}")
    k = len(j)
    print(f"    |n|={nb} events that also cross |n|={na}: {k} of {n_b_events}")
    if not k:
        return
    for c in ["~1"] + sorted(set(j["class"]) - {"~1", "other", ""}) + ["other", ""]:
        m = int((j["class"] == c).sum())
        if m:
            print(f"      Fa ratio {c or 'undefined':9s} {m:5d}  ({100*m/k:4.1f}%)")
    print(f"    crossing time |n|={nb} - |n|={na}: median {1e3*np.median(j['dt']):+.1f} ms, "
          f"{int((np.abs(j['dt']) <= dt_win).sum())} within "
          f"+/-{1e3*dt_win:.0f} ms")
    m = int(j["leak"].sum())
    print(f"    leak candidates (ratio ~1 AND within the window): {m}  "
          f"({100*m/max(n_b_events, 1):.1f}% of |n|={nb} events)")


def plot(jk, jl, labs, na, nb, out, dt_win):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # one panel per comparison: every shot thin, the shots whose crossings
    # fall within dt_win bold.  Only those compare Fa at about one time.
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    lo, hi = 1 / 3, 6.0
    bins = np.logspace(np.log10(lo), np.log10(hi), 37)
    for ax, j, lab, c in zip(axes, (jk, jl), labs, SERIES):
        ax.set_facecolor(SURFACE)
        r = j["ratio"].to_numpy(float)
        near = (np.abs(j["dt"].to_numpy(float)) <= dt_win)
        ok = np.isfinite(r) & (r > 0)
        r, near = r[ok], near[ok]
        # out-of-range ratios are counted at the edges, not piled into the
        # end bins where they would read as a peak
        for sel, lw, alpha, txt in (
                (np.ones_like(near), 1.2, 0.55, f"all  ({near.size})"),
                (near, 2.4, 1.0, f"|dt| <= {1e3*dt_win:g} ms  ({int(near.sum())})")):
            rs = r[sel]
            ax.hist(rs[(rs >= lo) & (rs <= hi)], bins=bins, histtype="step",
                    lw=lw, color=c, alpha=alpha, label=txt)
        for x, side, cnt in ((lo, "left", (r < lo, (r < lo) & near)),
                             (hi, "right", (r > hi, (r > hi) & near))):
            ax.text(0.01 if side == "left" else 0.99, 0.80,
                    f"{'<' if side == 'left' else '>'} {x:.2g}:\n"
                    f"{int(cnt[0].sum())} all\n{int(cnt[1].sum())} near",
                    transform=ax.transAxes, ha=side, va="top",
                    color=INK2, fontsize=9)
        for x, t in ((1.0, "leak  (same Fa)"),
                     (nb / na, f"genuine n={nb}  ({nb / na:g}x Fa)")):
            ax.axvline(x, color=INK3, lw=1.0, ls=(0, (4, 3)), zorder=0)
            ax.text(x * 1.03, 0.97, t, transform=ax.get_xaxis_transform(),
                    color=INK3, rotation=90, ha="left", va="top", fontsize=10)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 3, 4, 5]))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel(f"Fa(|n|={nb}) / Fa(|n|={na})", color=INK2)
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, which="both")
        # above the axes, so it never sits on a peak or a reference line
        ax.legend(frameon=False, labelcolor=INK2, loc="lower left",
                  bbox_to_anchor=(0.0, 1.01), ncol=2, borderaxespad=0)
        ax.set_title(lab, color=INK, fontsize=12, loc="left", pad=24)
    axes[0].set_ylabel("shots", color=INK2)
    fig.suptitle("Frequency ratio at the crossings", color=INK, fontsize=13,
                 fontweight="semibold", x=0.07, ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=170, facecolor=SURFACE, bbox_inches="tight")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("csv", nargs="+",
                   help="two ladder CSVs (|n|=a then |n|=b), or one path "
                        "containing {n} expanded over --n-pair")
    p.add_argument("--n-pair", type=int, nargs=2, default=[1, 2],
                   metavar=("A", "B"), help="the |n| pair (default 1 2)")
    p.add_argument("--thresholds", type=float, nargs=2, metavar=("TA", "TB"),
                   help="threshold for each ladder; default each one's knee")
    p.add_argument("--smooth", type=int, default=5)
    p.add_argument("--excess", type=float, default=1.0,
                   help="knee criterion, as in plot_ladder")
    p.add_argument("--ratio-tol", type=float, default=0.2,
                   help="a ratio within this fraction of 1 or 2 counts as ~1 "
                        "or ~2 (default 0.2)")
    p.add_argument("--dt-window", type=float, default=0.010,
                   help="|t_b - t_a| for a leak candidate [s] (default 0.010)")
    p.add_argument("--list", type=int, default=20,
                   help="how many leak candidates to print (default 20)")
    p.add_argument("--out", help="prefix: writes <out>.csv, and with --plot "
                                 "<out>.png/.pdf")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()

    na, nb = a.n_pair
    if len(a.csv) == 1:
        if "{n}" not in a.csv[0]:
            raise SystemExit("one path given: it must contain {n}, e.g. "
                             "'GATE-080/rm_ladder_g0.8_n{n}.csv'")
        paths = [a.csv[0].replace("{n}", str(n)) for n in (na, nb)]
    elif len(a.csv) == 2:
        paths = a.csv
    else:
        raise SystemExit("give two ladder CSVs, or one path containing {n}")
    da, db = (pd.read_csv(f) for f in paths)
    if a.plot and not a.out:
        raise SystemExit("--plot needs --out")

    if a.thresholds:
        ta, tb = a.thresholds
    else:
        ta, tb = knee_of(da, a.smooth, a.excess), knee_of(db, a.smooth, a.excess)
        if ta is None or tb is None:
            raise SystemExit("no knee found in one of the ladders; "
                             "pass --thresholds explicitly")
    A, la = crossings(da, ta)
    B, lb = crossings(db, tb)
    A_at_b, _ = crossings(da, lb)

    common = A.index.intersection(B.index)
    A, B, A_at_b = A.loc[common], B.loc[common], A_at_b.loc[common]
    ea, eb = np.isfinite(A["t"]), np.isfinite(B["t"])
    print(f"  |n|=a = {na}: {paths[0]}  at {la:.5f}"
          f"{'' if a.thresholds else '  (knee)'}")
    print(f"  |n|=b = {nb}: {paths[1]}  at {lb:.5f}"
          f"{'' if a.thresholds else '  (knee)'}")
    print(f"\n  {len(common)} shots in both ladders")
    print(f"    events  |n|={na}: {int(ea.sum())}   |n|={nb}: {int(eb.sum())}")
    print(f"    both {int((ea & eb).sum())}   |n|={na} only {int((ea & ~eb).sum())}"
          f"   |n|={nb} only {int((~ea & eb).sum())}"
          f"   neither {int((~ea & ~eb).sum())}")

    n_b_events = int(eb.sum())
    jk = compare(A, B, a.ratio_tol, a.dt_window, nb / na)
    jl = compare(A_at_b, B, a.ratio_tol, a.dt_window, nb / na)
    report(f"at the |n|={na} knee ({la:.5f})", jk, na, nb, n_b_events,
           a.dt_window)
    report(f"at the |n|={nb} threshold ({lb:.5f}) on both ladders", jl, na, nb,
           n_b_events, a.dt_window)

    leak = jk.index[jk["leak"]].union(jl.index[jl["leak"]])
    print(f"\n  leak candidates, either comparison: {len(leak)} of "
          f"{n_b_events} |n|={nb} events ({100*len(leak)/max(n_b_events, 1):.1f}%)")
    if len(leak) and a.list:
        # one row per shot: the same-level comparison where it flags the shot,
        # the knee comparison otherwise
        cand = pd.concat([jl[jl["leak"]], jk[jk["leak"] & ~jk.index.isin(
            jl.index[jl["leak"]])]])
        show = cand.loc[cand["dt"].abs().sort_values().index].head(a.list)
        print(f"  closest in time (check with plot_mode_track.py "
              f"--n-list {na} {nb}):")
        for s, r in show.iterrows():
            print(f"    {s}  t_a {r['t_a']:.4f}  t_b {r['t_b']:.4f}  "
                  f"Fa_a {r['fa_a']:7.0f}  Fa_b {r['fa_b']:7.0f}  "
                  f"ratio {r['ratio']:.2f}")

    if a.out:
        t = jl.add_suffix("_lvl_b").join(jk.add_suffix("_knee"), how="outer")
        t["leak_candidate"] = t.index.isin(leak)
        t.index.name = "shot"
        t.to_csv(f"{a.out}.csv")
        written = [f"{a.out}.csv"]
        if a.plot:
            plot(jk, jl, [f"at the |n|={na} knee", f"at the |n|={nb} threshold"],
                 na, nb, a.out, a.dt_window)
            written += [f"{a.out}.png", f"{a.out}.pdf"]
        print(f"\n  wrote {'  '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
