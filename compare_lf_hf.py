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


def decimate_fast(t_fast, x_fast, t0, t1):
    """Fast signal low-passed and decimated to 200 kHz over [t0, t1].

    Zero-phase FIR, so nothing above 100 kHz aliases into the band and any
    real latency in the LF path stays visible as a phase slope.  Returns the
    decimated time stamps and values (a uniform 200 kHz grid).
    """
    t_fast, x_fast = np.asarray(t_fast, float), np.asarray(x_fast, float)
    m = (t_fast >= t0) & (t_fast <= t1)
    t_d, x_d = t_fast[m], x_fast[m]
    if t_d.size < 100:
        raise ValueError(f"fast channel has {t_d.size} samples in "
                         f"[{t0:.3f}, {t1:.3f}] s")
    fs_fast = 1.0 / np.median(np.diff(t_d))
    q = int(round(fs_fast / FS_SLOW))
    if q < 1 or abs(fs_fast / q - FS_SLOW) > 0.01 * FS_SLOW:
        raise ValueError(f"fast rate {fs_fast/1e3:.1f} kHz is not an integer "
                         f"multiple of {FS_SLOW/1e3:.0f} kHz")
    x_d = x_d - np.mean(x_d)
    rest = q
    for f in (10, 5, 4, 3, 2):        # scipy advises q <= 13 per call
        while rest % f == 0 and rest > 1:
            x_d = signal.decimate(x_d, f, ftype="fir", zero_phase=True)
            t_d = t_d[::f][: x_d.size]
            rest //= f
    if rest != 1:                     # a prime factor > 13: one go
        x_d = signal.decimate(x_d, rest, ftype="fir", zero_phase=True)
        t_d = t_d[::rest][: x_d.size]
    return t_d, x_d


def estimate_delay(t_d, x_d, ts, ys, max_lag, band=BAND):
    """Delay d of the slow channel behind the fast one, by cross-correlation.

    Two digitisers do not share a time zero, and an offset of more than a few
    ms is enough to flatten the coherence completely: the analysis segments
    (10 ms) then see different stretches of signal.  Both signals are
    band-passed first so the correlation is about the mode band, not drift.

    slow(t) ~ fast(t - d).  Returns (d [s], peak normalised correlation).
    """
    dt = 1.0 / FS_SLOW
    sos = signal.butter(4, [band[0], band[1]], btype="band", fs=FS_SLOW,
                        output="sos")
    xb = signal.sosfiltfilt(sos, x_d)
    yb = signal.sosfiltfilt(sos, ys - np.mean(ys))
    if xb.size < yb.size:
        return 0.0, np.nan
    c = signal.correlate(xb, yb, mode="valid", method="fft")      # [k]
    e = np.concatenate([[0.0], np.cumsum(xb * xb)])
    local = np.sqrt(np.maximum(e[yb.size:yb.size + c.size] - e[:c.size], 1e-300))
    rho = c / (local * np.sqrt(np.sum(yb * yb)))
    k = np.arange(c.size)
    d = ts[0] - t_d[0] - k * dt
    ok = np.abs(d) <= max_lag
    if not ok.any():
        return 0.0, np.nan
    i = int(np.argmax(np.where(ok, np.abs(rho), -1.0)))
    return float(d[i]), float(rho[i])


def transfer(t_fast, x_fast, t_slow, x_slow, t0, t1, nperseg=2048,
             max_lag=0.05):
    """H(f) = S_xy / S_xx, coherence, delay and diagnostics.  Returns a dict."""
    ts_all = np.asarray(t_slow, float)
    ms = (ts_all >= t0) & (ts_all <= t1)
    ts, ys = ts_all[ms], np.asarray(x_slow, float)[ms]
    if ts.size < 4 * nperseg:
        raise ValueError(f"only {ts.size} slow samples in [{t0}, {t1}]; "
                         f"need >= {4*nperseg}")
    dts = np.diff(ts)
    irregular = float(np.max(np.abs(dts - np.median(dts))) / np.median(dts))

    t_d, x_d = decimate_fast(t_fast, x_fast, t0 - max_lag, t1 + max_lag)
    delay, rho = estimate_delay(t_d, x_d, ts, ys, max_lag)

    spl = CubicSpline(t_d, x_d)
    keep = (ts - delay >= t_d[0]) & (ts - delay <= t_d[-1])
    ts, ys = ts[keep], ys[keep]
    if ts.size < 4 * nperseg:
        raise ValueError("fast and slow records overlap too little after "
                         f"a {1e3*delay:.2f} ms alignment")
    xf = spl(ts - delay)
    ys = ys - np.mean(ys)

    f, Pxx = signal.welch(xf, FS_SLOW, nperseg=nperseg, detrend="linear")
    _, Pyy = signal.welch(ys, FS_SLOW, nperseg=nperseg, detrend="linear")
    _, Pxy = signal.csd(xf, ys, FS_SLOW, nperseg=nperseg, detrend="linear")
    return {"f": f, "H": Pxy / Pxx, "coh": np.abs(Pxy) ** 2 / (Pxx * Pyy),
            "n": ts.size, "delay": delay, "xcorr": rho,
            "rms_fast": float(np.std(xf)), "rms_slow": float(np.std(ys)),
            "irregular": irregular}


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
    # Phase = phi0 - 2 pi f dt: a constant phase (integral, derivative, sign)
    # plus a residual delay left over from the coarse alignment.  Fitting both
    # at once keeps a few microseconds of leftover delay from masquerading as a
    # phase, and avoids unwrapping.  For each trial dt the best phi0 is the
    # argument of z(dt) = sum_k u_k exp(+i 2 pi f_k dt), u = H/|H|.
    u = H[sel] / np.abs(H[sel])
    dts = np.arange(-200e-6, 200e-6, 0.5e-6)
    z = (u[:, None] * np.exp(2j * np.pi * f[sel][:, None] * dts[None, :])).sum(axis=0)
    j = int(np.argmax(np.abs(z)))
    phase = float(np.degrees(np.angle(z[j])))
    out["delay_resid"] = float(dts[j])
    out["phase_fit"] = float(np.abs(z[j]) / u.size)    # 1 = phase fully explained
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
    print(f"  delay of slow behind fast: {1e3*res['delay']:+.3f} ms by correlation"
          f"   (peak band-passed correlation {res['xcorr']:.3f})")
    print(f"  {'f [kHz]':>8} {'coh':>6} {'|H| slow/fast':>14} {'phase':>8}")
    for fc, c, g, ph in table(res):
        flag = "" if c >= COH_OK else "   (low coherence)"
        print(f"  {fc/1e3:8.2f} {c:6.2f} {g:14.4g} {ph:8.1f}{flag}")
    v = verdict(res)
    print(f"\n  verdict: {v['kind']}")
    if "note" in v:
        print(f"           {v['note']}")
        c = res["coh"]
        band = (res["f"] >= BAND[0]) & (res["f"] <= BAND[1])
        i = int(np.argmax(np.where(band, c, -1)))
        print(f"\n  diagnostics:")
        print(f"    best coherence in band   {c[i]:.2f} at {res['f'][i]/1e3:.2f} kHz,"
              f"  median {np.median(c[band]):.2f}  (pure noise ~ 0.05)")
        print(f"    rms in window            fast {res['rms_fast']:.4g}"
              f"   slow {res['rms_slow']:.4g}   [V]")
        print(f"    slow time stamps         worst dt deviation "
              f"{100*res['irregular']:.1f}% of the median")
        print(f"    band-passed correlation  {res['xcorr']:.3f} at the best delay")
        print("    -> correlation near 0 even after the delay search: the two\n"
              "       channels see no common signal.  Check the coil pairing, a\n"
              "       window with real MHD activity, and the time stamps.")
        print("    -> correlation clearly above 0 but coherence still low: the\n"
              "       delay is drifting (clock mismatch); try a shorter window.")
    else:
        print(f"           {BAND[0]/1e3:.0f}-{BAND[1]/1e3:.0f} kHz, {v['n_bins']}/"
              f"{v['n_band']} bins: log-log slope of |H| = {v['slope']:+.2f}, "
              f"phase = {v['phase_deg']:+.0f} deg, gain = {v['gain']:.4g}")
        print(f"           residual delay {1e6*v['delay_resid']:+.1f} us -> total "
              f"{1e3*(res['delay'] + v['delay_resid']):+.4f} ms;  phase fit quality "
              f"{v['phase_fit']:.2f} (1 = clean constant phase + delay)")
    return v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int, default=47002)
    p.add_argument("--t0", type=float, default=0.2)
    p.add_argument("--t1", type=float, default=0.8)
    p.add_argument("--all-coils", action="store_true")
    p.add_argument("--max-lag", type=float, default=0.05,
                   help="largest slow-behind-fast delay searched [s]")
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
                           a.t0, a.t1, max_lag=a.max_lag)
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
