"""Diagnose a fitted hazard model from its out-of-fold predictions.

    python hazard_predict.py --pooled hazard_EPM.pickle --outdir pred_EPM
    python hazard_predict.py --pooled hazard_EPM.pickle --shots 48123 48200

--pooled reads hazard_model --out.  Its `oof` holds, for every row, the
prediction of the one outer-fold model that never saw that row's shot (one
column per --repeats), so everything here is a held-out test, no refit and no
xgboost needed.  p is the probability of onset within the next slice; the
hazard rate is h = -ln(1 - p)/dt (constant rate within the slice, so
p = 1 - exp(-h dt)), not p/dt, which runs low by about p/2.

  calibration   rows binned by predicted p (equal-count bins); per bin the
                mean p against the observed event fraction, with a 68%
                binomial interval.  On the diagonal = calibrated.  Also the
                totals (sum p vs events) and the Hosmer-Lemeshow statistic.
                docs/hazard_calibration.tex explains the construction.
  aligned       h against time before onset for event shots, and against time
                before the last slice for censored shots: median and IQR per
                time bin.  Does the hazard rise ahead of onset, and how early?
  separation    per shot, the peak h: event shots against censored shots, as
                histograms and the AUC (probability a random event shot peaks
                higher than a random censored one).
  alarm         a threshold at the --alarm-quantile of the censored peaks
                (dashed on the separation plot): event shots above it are
                CAUGHT, with their lead time (onset - first slice above it),
                the rest MISSED.  Both are listed; every shot, with its class
                (caught / missed / false alarm / quiet), goes to
                <tag>_shot_peaks.csv.
  shots         h(t) and survival S(t) = prod(1 - p) for --shots, or for
                --sample N random event and N random censored shots.

The model files (per-fold boosters) are not used here; they are for running on
new shots.
"""

import argparse
import os
import pickle

import numpy as np

EPS = 1e-12


def rate(p, dt):
    """Per-slice probability p -> hazard rate h [1/s], h = -ln(1 - p)/dt.
    Exact for a rate constant within the slice; log1p keeps precision at
    small p, and the float64 cast avoids xgboost's float32 output."""
    return -np.log1p(-np.asarray(p, np.float64)) / dt


def load(path):
    with open(path, "rb") as fh:
        r = pickle.load(fh)
    oof = np.asarray(r["oof"], float)
    if oof.ndim == 1:
        oof = oof[:, None]
    dt = r.get("dt")
    if not dt:
        dts = [np.median(np.diff(r["T"][r["S"] == s])) for s in np.unique(r["S"])
               if (r["S"] == s).sum() > 1]
        dt = float(np.median(dts))
    return oof, np.asarray(r["Y"], float), np.asarray(r["S"]), \
        np.asarray(r["T"], float), float(dt), r


# ---------------------------------------------------------------- calibration
def binom_interval(k, n, z=1.0):
    """Wilson score interval for k successes in n trials (z=1: 68%)."""
    n = np.maximum(n, 1)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    w = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return np.maximum(c - w, 0.0), c + w


def calibration(p, y, nbins=10):
    """Equal-count bins of predicted p.  Returns a list of per-bin dicts and
    the totals.  Ties at a bin edge go to one bin, so counts can differ."""
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, nbins)
    rows = []
    for g in groups:
        if g.size == 0:
            continue
        n, k = g.size, y[g].sum()
        lo, hi = binom_interval(k, n)
        rows.append({"rows": n, "p_lo": p[g].min(), "p_hi": p[g].max(),
                     "p_mean": p[g].mean(), "expected": p[g].sum(),
                     "events": k, "rate": k / n, "rate_lo": lo, "rate_hi": hi})
    E, Ehat = y.sum(), p.sum()
    hl = sum((b["events"] - b["expected"]) ** 2
             / max(b["expected"] * (1 - b["p_mean"]), EPS) for b in rows)
    return rows, {"events": E, "expected": Ehat, "hosmer_lemeshow": hl,
                  "bins": len(rows)}


def print_calibration(rows, tot, dt):
    print("\n  CALIBRATION (equal-count bins of predicted p; held-out rows)")
    print(f"  {'bin':>3s} {'rows':>8s} {'p range':>21s} {'mean p':>10s} "
          f"{'expected':>9s} {'events':>6s} {'observed':>10s} {'68% interval':>21s}")
    for i, b in enumerate(rows):
        print(f"  {i:3d} {b['rows']:8d} {b['p_lo']:10.2e}-{b['p_hi']:<10.2e} "
              f"{b['p_mean']:10.2e} {b['expected']:9.2f} {int(b['events']):6d} "
              f"{b['rate']:10.2e} {b['rate_lo']:10.2e}-{b['rate_hi']:<10.2e}")
    print(f"\n  in the large: {int(tot['events'])} events observed, "
          f"{tot['expected']:.1f} expected (sum of p)")
    print(f"  Hosmer-Lemeshow {tot['hosmer_lemeshow']:.1f} over {tot['bins']} bins "
          f"(~chi2 with {tot['bins'] - 2} dof if calibrated: "
          f"mean {tot['bins'] - 2}, sd {np.sqrt(2 * (tot['bins'] - 2)):.1f})")


def plot_calibration(rows, p, path, title):
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(2, 1, figsize=(5.5, 7), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1]})
    x = np.array([r["p_mean"] for r in rows])
    yv = np.array([r["rate"] for r in rows])
    lo = np.array([r["rate_lo"] for r in rows])
    hi = np.array([r["rate_hi"] for r in rows])
    floor = 0.3 * max(min(x.min(), 1.0), EPS)
    shown = np.maximum(yv, floor)          # zero-event bins drawn at the floor
    a.errorbar(x, shown, yerr=[shown - np.maximum(lo, floor), hi - shown],
               fmt="o", color="C0", capsize=2)
    z = yv == 0
    if z.any():
        a.plot(x[z], shown[z], "v", color="C3", label="no events in bin")
        a.legend(fontsize=8)
    lim = [min(floor, x.min()) * 0.8, max(hi.max(), x.max()) * 1.3]
    a.plot(lim, lim, "k--", lw=1)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(lim); a.set_ylim(lim)
    a.set_ylabel("observed event fraction in bin")
    a.set_title(title, fontsize=9)
    a.grid(True, which="both", alpha=0.3)
    b.hist(np.clip(p, lim[0], None), bins=np.geomspace(lim[0], lim[1], 60),
           color="0.5")
    for r in rows:
        b.axvline(r["p_hi"], color="C0", lw=0.5, alpha=0.5)
    b.set_yscale("log")
    b.set_xlabel("predicted p per slice (bin mean)")
    b.set_ylabel("rows")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ alignment
def shot_index(S):
    """{shot: row indices in time order}; the table is stacked shot by shot."""
    idx = {}
    for i, s in enumerate(S):
        idx.setdefault(s, []).append(i)
    return {s: np.asarray(v) for s, v in idx.items()}


def aligned(p, Y, S, T, dt, window, step):
    """h binned by time before the reference slice (onset row for event
    shots, last row for censored ones).  Returns bin centres and, for each
    class, median / q25 / q75 / rows per bin."""
    edges = np.arange(0, window + step, step)
    out = {"tau": 0.5 * (edges[1:] + edges[:-1])}
    taus = {True: [], False: []}
    hs = {True: [], False: []}
    for s, ix in shot_index(S).items():
        ev = Y[ix].any()
        ref = T[ix][np.flatnonzero(Y[ix])[-1]] if ev else T[ix][-1]
        taus[ev].append(ref - T[ix]); hs[ev].append(rate(p[ix], dt))
    for ev, key in ((True, "event"), (False, "censored")):
        if not taus[ev]:
            continue
        tau, h = np.concatenate(taus[ev]), np.concatenate(hs[ev])
        b = np.digitize(tau, edges) - 1
        med, q1, q3, n = (np.full(edges.size - 1, np.nan) for _ in range(4))
        for k in range(edges.size - 1):
            v = h[b == k]
            n[k] = v.size
            if v.size:
                q1[k], med[k], q3[k] = np.quantile(v, [0.25, 0.5, 0.75])
        out[key] = {"median": med, "q25": q1, "q75": q3, "rows": n}
    return out


def plot_aligned(al, base_h, path, title):
    import matplotlib.pyplot as plt
    fig, a = plt.subplots(figsize=(6.5, 4))
    for key, c in (("event", "C3"), ("censored", "C0")):
        if key not in al:
            continue
        d = al[key]
        a.plot(1e3 * al["tau"], d["median"], color=c, label=f"{key}: median")
        a.fill_between(1e3 * al["tau"], d["q25"], d["q75"], color=c, alpha=0.2,
                       label=f"{key}: 25-75%")
    a.axhline(base_h, color="k", ls=":", lw=1, label="base rate")
    a.set_yscale("log")
    a.invert_xaxis()
    a.set_xlabel("time before onset (event) / before last slice (censored)  [ms]")
    a.set_ylabel("h = -ln(1-p)/dt  [1/s]")
    a.set_title(title, fontsize=9)
    a.legend(fontsize=7)
    a.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- separation
def auc(pos, neg):
    """P(random pos > random neg), ties counted half (Mann-Whitney)."""
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(allv.size)
    sv = allv[order]
    i = 0
    while i < sv.size:                       # average ranks over ties
        j = i
        while j + 1 < sv.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    r = ranks[:len(pos)].sum()
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def shot_peaks(p, Y, S, T, dt):
    """One record per shot: peak h, when it peaks, and the reference time --
    onset (last event row + dt, as plot_shots draws it) for event shots, the
    last slice for censored ones."""
    out = []
    for s, ix in shot_index(S).items():
        h, t = rate(p[ix], dt), T[ix]
        ev = bool(Y[ix].any())
        k = int(np.argmax(h))
        ref = t[np.flatnonzero(Y[ix])[-1]] + dt if ev else t[-1]
        out.append({"shot": int(s), "event": ev, "rows": int(ix.size),
                    "peak_h": float(h[k]), "t_peak": float(t[k]),
                    "t_ref": float(ref), "h_last": float(h[-1]),
                    "h": h, "t": t})
    return out


def separation(peaks):
    pe = np.array([r["peak_h"] for r in peaks if r["event"]])
    pc = np.array([r["peak_h"] for r in peaks if not r["event"]])
    return pe, pc, auc(pe, pc)


def alarm_split(peaks, quantile):
    """Threshold = the `quantile` of the censored shots' peak h, so a fraction
    1 - quantile of censored shots would alarm.  Event shots peaking above it
    are caught, the rest missed.  For each shot, t_alarm is the first slice
    above the threshold and lead = t_ref - t_alarm."""
    pc = np.array([r["peak_h"] for r in peaks if not r["event"]])
    thr = float(np.quantile(pc, quantile)) if pc.size else np.nan
    for r in peaks:
        above = np.flatnonzero(r["h"] > thr)
        r["alarm"] = bool(above.size)
        r["t_alarm"] = float(r["t"][above[0]]) if above.size else np.nan
        r["lead"] = r["t_ref"] - r["t_alarm"]
        r["class"] = (("caught" if r["alarm"] else "missed") if r["event"]
                      else ("false alarm" if r["alarm"] else "quiet"))
    return thr


def print_alarm(peaks, thr, quantile, n_list):
    ev = [r for r in peaks if r["event"]]
    ce = [r for r in peaks if not r["event"]]
    caught = sorted([r for r in ev if r["alarm"]], key=lambda r: -r["peak_h"])
    missed = sorted([r for r in ev if not r["alarm"]], key=lambda r: r["peak_h"])
    fa = sum(r["alarm"] for r in ce)
    print(f"\n  ALARM at the censored {100 * quantile:g}% peak, h > {thr:.3g}/s:")
    print(f"    event shots    caught {len(caught)} / {len(ev)} "
          f"({100 * len(caught) / max(len(ev), 1):.0f}%), missed {len(missed)}")
    print(f"    censored shots alarming {fa} / {len(ce)} "
          f"({100 * fa / max(len(ce), 1):.0f}%)")
    if caught:
        L = 1e3 * np.array([r["lead"] for r in caught])
        q = np.percentile(L, [25, 50, 75])
        print(f"    lead time of caught events (onset - first alarm): median "
              f"{q[1]:.0f} ms, IQR {q[0]:.0f}-{q[2]:.0f} ms")

    def table(rows, head):
        print(f"\n  {head}")
        print(f"    {'shot':>7s} {'rows':>5s} {'peak h':>9s} {'t_peak':>7s} "
              f"{'onset':>7s} {'peak-onset':>10s} {'h last':>9s} {'lead':>7s}")
        for r in rows[:n_list]:
            lead = "--" if np.isnan(r["lead"]) else f"{1e3 * r['lead']:.0f} ms"
            print(f"    {r['shot']:7d} {r['rows']:5d} {r['peak_h']:9.3g} "
                  f"{r['t_peak']:7.3f} {r['t_ref']:7.3f} "
                  f"{1e3 * (r['t_peak'] - r['t_ref']):7.0f} ms {r['h_last']:9.3g} "
                  f"{lead:>7s}")
        if len(rows) > n_list:
            print(f"    ... {len(rows) - n_list} more in the csv")
        if rows:
            print("    trace: --shots " + " ".join(str(r["shot"]) for r in rows[:8]))

    table(missed, f"MISSED event shots, lowest peak first ({len(missed)})")
    table(caught, f"CAUGHT event shots, highest peak first ({len(caught)})")


def write_peaks(peaks, path):
    import csv
    keys = ["shot", "class", "event", "rows", "peak_h", "t_peak", "t_ref",
            "h_last", "t_alarm", "lead"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in sorted(peaks, key=lambda r: (not r["event"], r["peak_h"])):
            w.writerow([r[k] for k in keys])


def plot_separation(pe, pc, a_uc, path, title, thr=None):
    import matplotlib.pyplot as plt
    fig, a = plt.subplots(figsize=(6, 4))
    v = np.concatenate([pe, pc])
    edges = np.geomspace(max(v.min(), EPS), v.max() * 1.01, 40)
    a.hist(pc, edges, color="C0", alpha=0.6, density=True,
           label=f"censored ({pc.size})")
    a.hist(pe, edges, color="C3", alpha=0.6, density=True,
           label=f"event ({pe.size})")
    if thr is not None and np.isfinite(thr):
        a.axvline(thr, color="k", ls="--", lw=1, label=f"alarm {thr:.3g}/s")
    a.set_xscale("log")
    a.set_xlabel("peak h over the shot  [1/s]")
    a.set_ylabel("density")
    a.set_title(f"{title}\nAUC {a_uc:.3f}", fontsize=9)
    a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- shot traces
def plot_shots(shots, oof, Y, S, T, dt, path, title):
    import matplotlib.pyplot as plt
    idx = shot_index(S)
    shots = [s for s in shots if s in idx]
    if not shots:
        return False
    fig, axs = plt.subplots(len(shots), 1, figsize=(7, 2.2 * len(shots)),
                            squeeze=False)
    for a, s in zip(axs[:, 0], shots):
        ix = idx[s]
        H = rate(oof[ix], dt)
        t = T[ix]
        a.step(t, np.nanmean(H, 1), where="post", color="C0", label="h")
        if H.shape[1] > 1:
            a.fill_between(t, np.nanmin(H, 1), np.nanmax(H, 1), step="post",
                           color="C0", alpha=0.25, label="range over repeats")
        a.set_yscale("log")
        a.set_ylabel("h [1/s]", fontsize=8)
        surv = np.cumprod(1 - np.nanmean(oof[ix], 1))
        b = a.twinx()
        b.step(t, surv, where="post", color="0.4", lw=1)
        b.set_ylim(0, 1.05)
        b.set_ylabel("S(t)", fontsize=8)
        ev = np.flatnonzero(Y[ix])
        lab = "censored"
        if ev.size:
            a.axvline(t[ev[-1]] + dt, color="C3", ls="--", lw=1)
            lab = f"onset by {t[ev[-1]] + dt:.3f} s"
        a.set_title(f"#{s}  ({lab})", fontsize=8)
        a.tick_params(labelsize=7); b.tick_params(labelsize=7)
    axs[-1, 0].set_xlabel("time [s]")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pooled", required=True, help="hazard_model --out pickle")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--bins", type=int, default=10, help="calibration bins")
    ap.add_argument("--window", type=float, default=0.3,
                    help="aligned plot: seconds before the reference slice")
    ap.add_argument("--step", type=float, default=None,
                    help="aligned plot: time bin [s]; default 2*dt")
    ap.add_argument("--shots", type=int, nargs="*", default=[])
    ap.add_argument("--sample", type=int, default=0,
                    help="also trace N random event and N random censored shots")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alarm-quantile", type=float, default=0.95,
                    help="alarm threshold = this quantile of the censored "
                         "shots' peak h (default 0.95: 5%% of them alarm)")
    ap.add_argument("--list", type=int, default=15,
                    help="missed / caught event shots printed (all in the csv)")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")

    oof, Y, S, T, dt, r = load(a.pooled)
    p = np.nanmean(oof, 1)
    if np.isnan(p).any():
        raise SystemExit("rows without an out-of-fold prediction")
    tag = r.get("tag") or "model"
    ctl = " [SHUFFLED-TARGET CONTROL]" if (r.get("args") or {}).get("shuffle_targets") else ""
    base = Y.mean()
    title = (f"{tag}: {os.path.basename(a.pooled)}{ctl}, {Y.size} rows, "
             f"{int(Y.sum())} events, dt {1e3 * dt:.2f} ms")
    print(f"  {title}")
    print(f"  {oof.shape[1]} repeat(s); base rate {base:.3e}/slice = "
          f"{rate(base, dt):.3f}/s")
    os.makedirs(a.outdir, exist_ok=True)
    out = lambda k: os.path.join(a.outdir, f"{tag}_{k}.png")  # noqa: E731

    rows, tot = calibration(p, Y, a.bins)
    print_calibration(rows, tot, dt)
    plot_calibration(rows, p, out("calibration"), title)

    al = aligned(p, Y, S, T, dt, a.window, a.step or 2 * dt)
    plot_aligned(al, rate(base, dt), out("aligned"), title)
    print("\n  ALIGNED median h [1/s]  (event | censored), by time before reference")
    for k in range(0, al["tau"].size, max(1, al["tau"].size // 8)):
        e = al.get("event", {}).get("median", [np.nan] * al["tau"].size)[k]
        c = al.get("censored", {}).get("median", [np.nan] * al["tau"].size)[k]
        print(f"    {1e3 * al['tau'][k]:7.1f} ms   {e:9.3g} | {c:9.3g}")

    peaks = shot_peaks(p, Y, S, T, dt)
    pe, pc, a_uc = separation(peaks)
    thr = alarm_split(peaks, a.alarm_quantile)
    plot_separation(pe, pc, a_uc, out("separation"), title, thr)
    print(f"\n  SEPARATION  peak h per shot: AUC {a_uc:.3f}  "
          f"({pe.size} event / {pc.size} censored shots; 0.5 = no separation)")
    print("    note: event shots are cut at onset, censored ones run to the end"
          " of the window, so the shot lengths differ")
    print_alarm(peaks, thr, a.alarm_quantile, a.list)
    csv_path = os.path.join(a.outdir, f"{tag}_shot_peaks.csv")
    write_peaks(peaks, csv_path)
    print(f"\n  wrote {csv_path}")

    shots = list(a.shots)
    if a.sample:
        rng = np.random.default_rng(a.seed)
        idx = shot_index(S)
        evs = [s for s, ix in idx.items() if Y[ix].any()]
        cen = [s for s, ix in idx.items() if not Y[ix].any()]
        shots += list(rng.choice(evs, min(a.sample, len(evs)), replace=False))
        shots += list(rng.choice(cen, min(a.sample, len(cen)), replace=False))
    if shots:
        if not plot_shots(shots, oof, Y, S, T, dt, out("shots"), title):
            print("  none of --shots is in the table")
    for k in ("calibration", "aligned", "separation") + (("shots",) if shots else ()):
        if os.path.exists(out(k)):
            print(f"  wrote {out(k)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
