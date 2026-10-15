"""How is the OMAHA slow (LF) output related to the fast (HF) one?

Both channels come off the same coil.  Whatever the coil produces is volts
proportional to dB/dt; the question is what the LF path did to it.  The
transfer function H(f) = S_xy / S_xx (slow per fast), estimated from the two
signals themselves, answers that without anyone's help:

    |H| flat,  phase ~ 0     same quantity (a pure gain, maybe a filter)
    |H| ~ 1/f, phase ~ -90   slow is the TIME INTEGRAL of fast, i.e. B if
                             fast is dB/dt  -- dividing by omega again is wrong
    |H| ~ f,   phase ~ +90   slow is the derivative of fast
    anything else            some filter; read the table

The flat-band |H| is also the relative gain between the two paths.

    python compare_lf_hf.py --shot 47002
    python compare_lf_hf.py --shot 47002 --all-coils --t0 0.2 --t1 0.8 --plot

Use a window with MHD activity.  Pure noise has no common signal, the
coherence comes out low, and the verdict says so rather than guessing.
"""

import argparse

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline

FS_SLOW = 200.0e3
BAND = (1.0e3, 30.0e3)            # where the slope and phase are judged
COH_OK = 0.8                      # bins below this carry no common signal


def to_slow_grid(t_fast, x_fast, t_slow, t0, t1):
    """Fast signal, low-passed and resampled onto the slow channel's time stamps.

    The fast channel is decimated to 200 kHz with a zero-phase FIR first, so
    nothing above 100 kHz aliases into the band, then interpolated (cubic) to
    the slow channel's exact sample times.  Zero-phase keeps any real latency
    in the LF path visible as a phase slope instead of hiding it.
    """
    t_fast, x_fast = np.asarray(t_fast, float), np.asarray(x_fast, float)
    m = (t_fast >= t0 - 1e-3) & (t_fast <= t1 + 1e-3)
    t_f, x_f = t_fast[m], x_fast[m]
    fs_fast = 1.0 / np.median(np.diff(t_f))
    q = int(round(fs_fast / FS_SLOW))
    if q < 1 or abs(fs_fast / q - FS_SLOW) > 0.01 * FS_SLOW:
        raise ValueError(f"fast rate {fs_fast/1e3:.1f} kHz is not an integer "
                         f"multiple of {FS_SLOW/1e3:.0f} kHz")
    x_f = x_f - np.mean(x_f)
    t_d = t_f
    # decimate in stages: scipy advises q <= 13 per call
    stages = [f for f in (10, 5, 4, 3, 2) if q % f == 0]
    rest = q
    for f in stages:
        while rest % f == 0 and rest > 1:
            x_f = signal.decimate(x_f, f, ftype="fir", zero_phase=True)
            t_d = t_d[::f][: x_f.size]
            rest //= f
    if rest != 1:                     # prime factor > 13: do it in one go
        x_f = signal.decimate(x_f, rest, ftype="fir", zero_phase=True)
        t_d = t_d[::rest][: x_f.size]
    ts = np.asarray(t_slow, float)
    ms = (ts >= t0) & (ts <= t1) & (ts >= t_d[0]) & (ts <= t_d[-1])
    return ts[ms], CubicSpline(t_d, x_f)(ts[ms]), ms


def transfer(t_fast, x_fast, t_slow, x_slow, t0, t1, nperseg=2048):
    """H(f) = S_xy / S_xx, coherence, on the slow grid.  Returns a dict."""
    ts, xf, ms = to_slow_grid(t_fast, x_fast, t_slow, t0, t1)
    ys = np.asarray(x_slow, float)[ms]
    ys = ys - np.mean(ys)
    if ts.size < 4 * nperseg:
        raise ValueError(f"only {ts.size} common samples; need >= {4*nperseg}")
    f, Pxx = signal.welch(xf, FS_SLOW, nperseg=nperseg)
    _, Pyy = signal.welch(ys, FS_SLOW, nperseg=nperseg)
    _, Pxy = signal.csd(xf, ys, FS_SLOW, nperseg=nperseg)
    H = Pxy / Pxx
    coh = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    return {"f": f, "H": H, "coh": coh, "n": ts.size}


def verdict(res, band=BAND, coh_ok=COH_OK):
    """Slope of |H| and phase over `band` on bins with real common signal."""
    f, H, coh = res["f"], res["H"], res["coh"]
    sel = (f >= band[0]) & (f <= band[1]) & (coh >= coh_ok)
    n_sel = int(sel.sum())
    out = {"n_bins": n_sel, "n_band": int(((f >= band[0]) & (f <= band[1])).sum())}
    if n_sel < 8:
        out.update(kind="no common signal",
                   note=f"only {n_sel} bins in {band[0]/1e3:.0f}-{band[1]/1e3:.0f} kHz "
                        f"reach coherence {coh_ok}; pick a window with MHD activity")
        return out
    lf, lh = np.log10(f[sel]), np.log10(np.abs(H[sel]))
    slope = float(np.polyfit(lf, lh, 1)[0])
    # phase of the complex mean, so +-180 wrapping cannot fool a plain average
    phase = float(np.degrees(np.angle(np.mean(H[sel] / np.abs(H[sel])))))
    gain = float(10 ** np.median(lh))
    out.update(slope=slope, phase_deg=phase, gain=gain)
    if abs(slope) < 0.25 and abs(phase) < 25:
        kind = "same quantity (flat gain)"
    elif abs(slope + 1) < 0.3 and abs(phase + 90) < 30:
        kind = "slow = time integral of fast"
    elif abs(slope - 1) < 0.3 and abs(phase - 90) < 30:
        kind = "slow = derivative of fast"
    elif abs(slope) < 0.25 and abs(abs(phase) - 180) < 25:
        kind = "same quantity, opposite sign"
    else:
        kind = "neither: a filter, or mixed -- read the table"
    out["kind"] = kind
    return out


def table(res, nb=14, lo=300.0, hi=60.0e3):
    f, H, coh = res["f"], res["H"], res["coh"]
    edges = np.geomspace(lo, hi, nb + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (f >= a) & (f < b)
        if not m.any():
            continue
        c = float(np.mean(coh[m]))
        h = np.mean(H[m] / np.abs(H[m]))
        rows.append((float(np.sqrt(a * b)), c, float(np.median(np.abs(H[m]))),
                     float(np.degrees(np.angle(h)))))
    return rows


def report(label, res):
    print(f"\n{label}   ({res['n']} common samples)")
    print(f"  {'f [kHz]':>8} {'coh':>6} {'|H| slow/fast':>14} {'phase':>8}")
    for fc, c, g, ph in table(res):
        flag = "" if c >= COH_OK else "   (low coherence)"
        print(f"  {fc/1e3:8.2f} {c:6.2f} {g:14.4g} {ph:8.1f}{flag}")
    v = verdict(res)
    print(f"\n  verdict: {v['kind']}")
    if "note" in v:
        print(f"           {v['note']}")
    else:
        print(f"           {BAND[0]/1e3:.0f}-{BAND[1]/1e3:.0f} kHz, {v['n_bins']}/"
              f"{v['n_band']} bins: log-log slope of |H| = {v['slope']:+.2f}, "
              f"phase = {v['phase_deg']:+.0f} deg, gain = {v['gain']:.4g}")
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int, default=47002)
    p.add_argument("--t0", type=float, default=0.2)
    p.add_argument("--t1", type=float, default=0.8)
    p.add_argument("--all-coils", action="store_true")
    p.add_argument("--plot", action="store_true", help="write compare_lf_hf_<shot>.png")
    a = p.parse_args()

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
    results = {}
    for line in rows.itertuples():
        try:
            vs = client.get(line.name_slow, a.shot)
            vf = client.get(line.name_fast, a.shot)
            res = transfer(vf.time.data, vf.data, vs.time.data, vs.data,
                           a.t0, a.t1)
        except Exception as e:
            print(f"\n{line.name_slow}: {type(e).__name__}: {e}")
            continue
        results[line.name_slow] = report(
            f"{line.name_slow}  vs  {line.name_fast}   phi {line.phi}", res)
        if a.plot:
            _plot(a.shot, line.name_slow, res)

    if len(results) > 1:
        gains = [v["gain"] for v in results.values() if "gain" in v]
        if len(gains) > 1:
            print(f"\nflat-band gain across coils: min {min(gains):.4g} "
                  f"max {max(gains):.4g}  (spread {max(gains)/min(gains):.2f}x)")
            print("  a spread here is a relative-sensitivity difference between "
                  "the LF paths --\n  exactly what the toroidal fit assumes is "
                  "absent")
    return 0


def _plot(shot, name, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, H, coh = res["f"], res["H"], res["coh"]
    ok = (f > 200) & (f < 60e3)
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
    ax[0].loglog(f[ok], np.abs(H[ok]), lw=0.8)
    ax[0].set_ylabel("|H| slow/fast")
    ax[1].semilogx(f[ok], np.degrees(np.angle(H[ok])), lw=0.8)
    ax[1].set_ylabel("phase [deg]")
    ax[2].semilogx(f[ok], coh[ok], lw=0.8)
    ax[2].axhline(COH_OK, ls=":", c="gray")
    ax[2].set_ylabel("coherence")
    ax[2].set_xlabel("f [Hz]")
    fig.suptitle(f"{shot}  {name}")
    out = f"compare_lf_hf_{shot}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"  wrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
