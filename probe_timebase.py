"""Is the OMAHA slow time base uniform, and where is it not?

saddle_data.load_omaha keeps  time[idt[0]:idt[-1]]  with
idt = flatnonzero(diff(time) < 1e-4): a contiguous slice from the first to the
last "dense" interval.  That is only the dense section if there are no coarse
intervals INSIDE the slice.  Anything inside it is silently kept, and since
SaddleFull.spectrum computes Fs from (t_end - t_start)/(N - 1) and has its
uniform-dt assertion commented out, a hole shifts the whole frequency axis.

This reports, for one channel's time vector:

  1. the intervals histogrammed by their value, with where in the shot they sit
  2. the record as runs: regular / irregular-dense / coarse / non-monotonic
  3. what Gio's slice actually keeps, what is hidden inside it, and the error
     that makes in the frequency axis

    python probe_timebase.py --shot 47002
    python probe_timebase.py --shots 47000 47001 47002 --summary
    python probe_timebase.py --shot 47002 --plot
"""

import argparse

import numpy as np

DENSE_DT = 1.0e-4          # Gio's rule: an interval below this is "dense"
REGULAR_TOL = 0.01         # within 1% of the nominal dt counts as regular
NOMINAL_FS = 200.0e3


def classify(dt, dt_nom):
    """Per-interval class: 0 regular, 1 dense but irregular, 2 coarse, 3 backwards."""
    cls = np.full(dt.size, 1, dtype=int)
    cls[np.abs(dt - dt_nom) <= REGULAR_TOL * dt_nom] = 0
    cls[dt >= DENSE_DT] = 2
    cls[dt <= 0] = 3
    return cls


NAMES = {0: "regular", 1: "dense-irregular", 2: "coarse", 3: "NON-MONOTONIC"}


def runs(time, cls):
    """Run-length encode the interval classes -> list of dicts."""
    dt = np.diff(time)
    out = []
    if cls.size == 0:
        return out
    edges = np.flatnonzero(np.diff(cls)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [cls.size]])
    for a, b in zip(starts, stops):
        seg = dt[a:b]
        out.append({"cls": int(cls[a]), "t0": float(time[a]), "t1": float(time[b]),
                    "n": int(b - a), "dt_min": float(seg.min()),
                    "dt_med": float(np.median(seg)), "dt_max": float(seg.max())})
    return out


def histogram(time, nb_per_decade=2):
    """Intervals binned by value (log), with where in the shot each bin lives."""
    dt = np.diff(time)
    pos = dt > 0
    lo, hi = np.floor(np.log10(dt[pos].min())), np.ceil(np.log10(dt[pos].max()))
    edges = 10 ** np.arange(lo, hi + 1e-9, 1.0 / nb_per_decade)
    tmid = 0.5 * (time[:-1] + time[1:])
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = pos & (dt >= a) & (dt < b)
        if not m.any():
            continue
        rows.append({"lo": float(a), "hi": float(b), "n": int(m.sum()),
                     "covers": float(dt[m].sum()),
                     "t_min": float(tmid[m].min()), "t_med": float(np.median(tmid[m])),
                     "t_max": float(tmid[m].max())})
    if (~pos).any():
        m = ~pos
        rows.append({"lo": float(dt[m].min()), "hi": 0.0, "n": int(m.sum()),
                     "covers": 0.0, "t_min": float(tmid[m].min()),
                     "t_med": float(np.median(tmid[m])), "t_max": float(tmid[m].max())})
    return rows


def gio_slice(time):
    """Exactly what saddle_data.load_omaha keeps, and what that implies."""
    dt = np.diff(time)
    idt = np.flatnonzero(dt < DENSE_DT)
    if idt.size < 2:
        return None
    i0, i1 = int(idt[0]), int(idt[-1])
    kept = time[i0:i1]
    seg = np.diff(kept)
    hidden = seg >= DENSE_DT
    Fs_gio = 1.0 / ((kept[-1] - kept[0]) / (kept.size - 1))     # SaddleFull.spectrum
    dense = dt[dt < DENSE_DT]
    Fs_true = 1.0 / np.median(dense)
    return {"i0": i0, "i1": i1, "t0": float(kept[0]), "t1": float(kept[-1]),
            "n": int(kept.size), "hidden_n": int(hidden.sum()),
            "hidden_time": float(seg[hidden].sum()),
            "hidden_where": ([float(kept[:-1][hidden].min()), float(kept[:-1][hidden].max())]
                             if hidden.any() else None),
            "backwards": int((seg <= 0).sum()),
            "Fs_gio": float(Fs_gio), "Fs_true": float(Fs_true),
            "fs_error": float(Fs_gio / Fs_true - 1.0)}


def analyse(time):
    time = np.asarray(time, float)
    dt = np.diff(time)
    dense = dt[(dt > 0) & (dt < DENSE_DT)]
    dt_nom = float(np.median(dense)) if dense.size else 1.0 / NOMINAL_FS
    cls = classify(dt, dt_nom)
    return {"n": int(time.size), "t0": float(time[0]), "t1": float(time[-1]),
            "dt_nom": dt_nom, "counts": {NAMES[k]: int((cls == k).sum()) for k in NAMES},
            "runs": runs(time, cls), "hist": histogram(time), "gio": gio_slice(time)}


def report(label, r, max_runs=25):
    print(f"\n{label}")
    print(f"  {r['n']} samples, t = {r['t0']:.4f} .. {r['t1']:.4f} s, "
          f"nominal dt = {r['dt_nom']*1e6:.3f} us (Fs {1/r['dt_nom']/1e3:.1f} kHz)")
    print("  intervals: " + ", ".join(f"{k} {v}" for k, v in r["counts"].items() if v))

    print(f"\n  intervals by value   {'dt range [s]':>24} {'n':>9} {'time covered':>13}"
          f"   where in the shot (min / median / max)")
    for h in r["hist"]:
        rng = f"{h['lo']:.2e} .. {h['hi']:.2e}" if h["hi"] else "<= 0"
        print(f"                       {rng:>24} {h['n']:9d} {h['covers']:12.4f}s"
              f"   {h['t_min']:8.4f} / {h['t_med']:8.4f} / {h['t_max']:8.4f}")

    rr = r["runs"]
    print(f"\n  runs ({len(rr)}):")
    show = rr
    if len(rr) > max_runs:
        keep = set(np.argsort([-x["t1"] + x["t0"] for x in rr])[:max_runs])
        show = [x for i, x in enumerate(rr) if i in keep]
        print(f"    {len(rr) - max_runs} short runs hidden; the {max_runs} longest:")
    print(f"    {'class':>16} {'t from':>9} {'t to':>9} {'span [s]':>9} {'n':>8} "
          f"{'dt median':>11} {'dt min..max':>24}")
    for x in show:
        print(f"    {NAMES[x['cls']]:>16} {x['t0']:9.4f} {x['t1']:9.4f} "
              f"{x['t1']-x['t0']:9.4f} {x['n']:8d} {x['dt_med']:11.3e} "
              f"{x['dt_min']:11.3e}..{x['dt_max']:.3e}")

    g = r["gio"]
    print("\n  what load_omaha keeps  (time[idt[0]:idt[-1]], idt = diff < 1e-4)")
    if g is None:
        print("    fewer than 2 dense intervals: nothing survives")
        return
    print(f"    samples {g['n']}   t = {g['t0']:.4f} .. {g['t1']:.4f} s "
          f"({g['t1']-g['t0']:.4f} s)")
    if g["hidden_n"] or g["backwards"]:
        print(f"    HIDDEN INSIDE THE SLICE: {g['hidden_n']} coarse intervals "
              f"({g['hidden_time']*1e3:.3f} ms of missing time), "
              f"{g['backwards']} backwards")
        if g["hidden_where"]:
            print(f"      located at t = {g['hidden_where'][0]:.4f} .. "
                  f"{g['hidden_where'][1]:.4f} s")
        print(f"    Fs as SaddleFull.spectrum computes it: {g['Fs_gio']/1e3:.3f} kHz "
              f"vs {g['Fs_true']/1e3:.3f} kHz true -> frequency axis off by "
              f"{100*g['fs_error']:+.3f}%")
        print("    and the spectrogram time stamps after the hole are shifted too")
    else:
        print("    clean: every kept interval is dense, so the slice is the dense "
              "section")
        print(f"    Fs {g['Fs_gio']/1e3:.3f} kHz  (frequency-axis error "
              f"{100*g['fs_error']:+.4f}%)")


def plot(shot, label, time, r, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dt = np.diff(time)
    tm = 0.5 * (time[:-1] + time[1:])
    fig, ax = plt.subplots(2, 1, figsize=(8, 7))
    k = max(1, dt.size // 200000)
    pos = dt[::k] > 0
    ax[0].semilogy(tm[::k][pos], dt[::k][pos], ".", ms=1.5)
    ax[0].axhline(DENSE_DT, ls=":", c="gray", label="dense threshold 1e-4 s")
    g = r["gio"]
    if g:
        ax[0].axvspan(g["t0"], g["t1"], color="C1", alpha=0.15, label="kept by load_omaha")
    ax[0].set_xlabel("t [s]")
    ax[0].set_ylabel("interval dt [s]")
    ax[0].legend(loc="best", fontsize=8)
    pos_all = dt[dt > 0]
    ax[1].hist(np.log10(pos_all), bins=80)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("log10(dt [s])")
    ax[1].set_ylabel("intervals")
    fig.suptitle(f"{shot}  {label}")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int)
    p.add_argument("--shots", type=int, nargs="+")
    p.add_argument("--all-coils", action="store_true",
                   help="every radial coil, not only the first")
    p.add_argument("--fast", action="store_true",
                   help="also the fast channel (15M samples: slow to read)")
    p.add_argument("--summary", action="store_true",
                   help="one line per shot and coil, no tables")
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    shots = a.shots or ([a.shot] if a.shot else [47002])

    import giopath  # noqa: F401
    try:
        import pyuda
        import omaha_coils as oc
    except ImportError:
        giopath.report()
        raise

    client = pyuda.Client()
    coils = oc.coils[oc.coils.orient == "R"]
    rows = coils if a.all_coils else coils.iloc[:1]
    if a.summary:
        print(f"{'shot':>7} {'channel':<24} {'n':>8} {'t0':>8} {'t1':>8} "
              f"{'kept t0':>8} {'kept t1':>8} {'hidden':>7} {'hole ms':>8} {'Fs err %':>9}")
    for shot in shots:
        for line in rows.itertuples():
            kinds = [("slow", line.name_slow)] + ([("fast", line.name_fast)] if a.fast else [])
            for kind, name in kinds:
                try:
                    t = np.asarray(client.get(name, shot).time.data, float)
                except Exception as e:
                    print(f"{shot} {name}: {type(e).__name__}: {e}")
                    continue
                r = analyse(t)
                if a.summary:
                    g = r["gio"]
                    if g is None:
                        print(f"{shot:7d} {name:<24} {r['n']:8d} {r['t0']:8.3f} "
                              f"{r['t1']:8.3f}   (nothing dense)")
                    else:
                        print(f"{shot:7d} {name:<24} {r['n']:8d} {r['t0']:8.3f} "
                              f"{r['t1']:8.3f} {g['t0']:8.4f} {g['t1']:8.4f} "
                              f"{g['hidden_n']:7d} {1e3*g['hidden_time']:8.3f} "
                              f"{100*g['fs_error']:+9.4f}")
                else:
                    report(f"shot {shot}  {kind}  {name}  (phi {line.phi})", r)
                if a.plot:
                    plot(shot, name, t, r, f"timebase_{shot}_{kind}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
