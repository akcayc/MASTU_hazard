"""Is the coherent power plasma, or is it instrumental?

A tearing mode cannot exist before breakdown.  The OMAHA record opens at
~1 ms and breakdown is later, so every shot carries a free no-plasma control.
Comparing the GATED amplitude there against the flat-top answers the one
question the ladder's knee cannot:

    ratio ~ 1 at every gate     the surviving coherent power is present with
                                no plasma -- it is instrumental (a supply
                                harmonic, PF switching, an NBI chopper), and
                                the coherence gate is isolating a line, not MHD
    ratio grows with the gate   the gate is removing incoherent noise and
                                leaving plasma signal.  The detector is real

Ungated the ratio is 1.31 (1.92e-07 vs 2.52e-07 over shots 47000-47099), which
is why the gate was introduced.  The question is whether gating separates them.

The frequency is the second, more diagnostic test and is free here: Fa is
already stored per slice.  A narrow peak at a FIXED frequency appearing in both
windows and repeating shot to shot names an instrumental line.  Broad,
shot-varying and drifting is MHD.

    python compare_pre_plasma.py --shot-min 47000 --shot-max 47020
    python compare_pre_plasma.py --shots 47000 47001 --gates 0 0.4 0.8 --amp-form centroid
"""

import argparse
import time

import numpy as np

import giopath  # noqa: F401

try:
    import pyuda
    from saddle_data import load_omaha_slow, AmplitudeSelector
    from saddle_extras import amplitude_with_coherence, amplitude_diiid
    from saddle_analysis import (NTOR_DEFAULT, TIME_INTERVAL_DEFAULT,
                                 FMIN_DEFAULT, FMAX_DEFAULT)
    from flattop import find_flattop, FlatTopConfig
except ImportError:
    giopath.report()
    raise

GAUSS_PER_TESLA = 1.0e4
GATES = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def flattop_window(shot, client, t0, t1):
    """[ta, tb] from Ip, or the fixed fallback if that fails."""
    if t0 is not None and t1 is not None:
        return float(t0), float(t1), "fixed"
    try:
        v = client.get("/AMC/PLASMA_CURRENT", shot)
        ft = find_flattop(np.asarray(v.time.data), 1e-3 * np.asarray(v.data),
                          FlatTopConfig())
        if ft.ok:
            return ft.t_a, ft.t_b, "flattop"
    except Exception:
        pass
    return 0.2, 0.8, "fallback"


def stats(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(n=0, med=np.nan, p90=np.nan, mx=np.nan)
    return dict(n=a.size, med=float(np.median(a)),
                p90=float(np.percentile(a, 90)), mx=float(a.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shots", type=int, nargs="+")
    p.add_argument("--shot-min", type=int)
    p.add_argument("--shot-max", type=int)
    p.add_argument("--gates", type=float, nargs="+", default=GATES)
    p.add_argument("--n-list", type=int, nargs="+", default=[1],
                   help="|n| values to analyse; one block of output each. The "
                        "load and the spectrogram are shared, so extra |n| "
                        "cost almost nothing")
    p.add_argument("--n-detect", type=int, help="deprecated alias for --n-list")
    p.add_argument("--out", help="also write everything printed to this .txt")
    p.add_argument("--nfft", type=int, default=512)
    p.add_argument("--t-pre", type=float, default=0.020,
                   help="end of the no-plasma window [s]")
    p.add_argument("--t0", type=float, help="fixed flat-top start, else from Ip")
    p.add_argument("--t1", type=float, help="fixed flat-top end")
    p.add_argument("--amp-form", choices=["centroid", "perbin"],
                   default="centroid")
    p.add_argument("--report-every", type=int, default=100,
                   help="progress line every N shots; 0 to silence")
    a = p.parse_args()
    if a.n_detect is not None:
        a.n_list = [a.n_detect]
    shots = a.shots or list(range(a.shot_min, a.shot_max))
    nlist = sorted({abs(int(n)) for n in a.n_list if int(n) != 0})
    if not nlist:
        raise SystemExit("--n-list needs at least one non-zero |n|")

    client = pyuda.Client()
    # acc[(|n|, gate)] = (pre amplitudes, flat amplitudes, pre freqs,
    #                     flat freqs, pre slices surviving, pre slices total)
    acc = {(n, g): ([], [], [], [], 0, 0) for n in nlist for g in a.gates}
    nshot, win_kind = 0, None

    # The load, the spectrogram and n_detection do NOT depend on the gate --
    # only the masking inside the amplitude does.  Doing them once per shot
    # rather than once per shot per gate is an 8x saving at eight gates, which
    # is the difference between an hour and most of a day on 2800 shots.
    # Extra |n| share all of that too -- only the masking differs.
    ti = TIME_INTERVAL_DEFAULT
    mags = sorted({abs(int(n)) for n in NTOR_DEFAULT if int(n) != 0})
    search = np.asarray([0] + [sg * m for m in mags for sg in (-1, 1)])
    sels = {n: AmplitudeSelector(ntor=n, time=ti,
                                 f_min=np.array([FMIN_DEFAULT] * len(ti)),
                                 f_max=np.array([FMAX_DEFAULT] * len(ti)),
                                 v_min=0.0) for n in nlist}
    t0_run = time.time()

    for i, shot in enumerate(shots):
        try:
            ta, tb, win_kind = flattop_window(shot, client, a.t0, a.t1)
            sd = load_omaha_slow(shot)
            if sd.time.size < 2:
                raise RuntimeError("no OMAHA data")
            rm = sd.spectrum(a.nfft).n_detection(search)
            for nabs in nlist:
              sel = sels[nabs]
              for g in a.gates:
                if a.amp_form == "perbin":
                    t, amp_r = amplitude_diiid(rm, sel, g, True)
                    _, frq, _, _, _ = amplitude_with_coherence(rm, sel, g, True)
                else:
                    t, frq, _, amp_r, _ = amplitude_with_coherence(rm, sel, g, True)
                t = np.asarray(t, float)
                amp = np.asarray(amp_r, float) * GAUSS_PER_TESLA
                frq = np.asarray(frq, float)
                pre = t < a.t_pre
                flat = (t >= ta) & (t <= tb)
                P, F, PF, FF, surv, tot = acc[(nabs, g)]
                P.extend(amp[pre]); F.extend(amp[flat])
                PF.extend(frq[pre]); FF.extend(frq[flat])
                acc[(nabs, g)] = (P, F, PF, FF,
                                  surv + int(np.isfinite(amp[pre]).sum()),
                                  tot + int(pre.sum()))
            nshot += 1
            if a.report_every and (i + 1) % a.report_every == 0:
                el = time.time() - t0_run
                rate = (i + 1) / el
                print(f"    {i+1}/{len(shots)}  {rate:.1f} shot/s  "
                      f"ETA {(len(shots)-i-1)/max(rate,1e-9)/60:.0f} min")
        except Exception as e:
            print(f"  skip {shot}: {type(e).__name__}: {e}")

    if not nshot:
        raise SystemExit("no shots analysed")

    lines = []

    def emit(txt=""):
        print(txt)
        lines.append(txt)

    emit(f"\n{nshot} shots, amp form '{a.amp_form}', window from {win_kind}")
    emit(f"pre-plasma is t < {a.t_pre*1e3:.0f} ms -- no plasma, so any "
         f"amplitude there is instrumental")

    for nabs in nlist:
        emit(f"\n{'='*70}\n|n| = {nabs}\n")
        emit(f"  {'gate':>5} {'pre med':>10} {'flat med':>10} {'ratio':>7}"
             f" {'pre p90':>10} {'flat p90':>10} {'pre kept':>9}")
        for g in a.gates:
            P, F, PF, FF, surv, tot = acc[(nabs, g)]
            sp, sf = stats(P), stats(F)
            r = sf["med"] / sp["med"] if sp["med"] and np.isfinite(sp["med"]) else np.nan
            kept = 100.0 * surv / tot if tot else np.nan
            emit(f"  {g:5.2f} {sp['med']:10.3e} {sf['med']:10.3e} {r:7.2f}"
                 f" {sp['p90']:10.3e} {sf['p90']:10.3e} {kept:8.1f}%")

        emit(f"\n  {'gate':>5} {'Fa pre kHz (med, IQR)':>28} "
             f"{'Fa flat kHz (med, IQR)':>28}")
        for g in a.gates:
            _, _, PF, FF, _, _ = acc[(nabs, g)]

            def fs(x):
                x = np.asarray(x, float); x = x[np.isfinite(x)] / 1e3
                if x.size == 0:
                    return "         --"
                q1, q3 = np.percentile(x, [25, 75])
                return f"{np.median(x):8.2f}  [{q1:6.2f},{q3:6.2f}]"
            emit(f"  {g:5.2f} {fs(PF):>28} {fs(FF):>28}")

        g0, g1 = a.gates[0], a.gates[-1]
        P0, F0, PF0, FF0, s0, t0_ = acc[(nabs, g0)]
        P1, F1, PF1, FF1, s1, t1_ = acc[(nabs, g1)]
        sp0, sp1, sf0, sf1 = stats(P0), stats(P1), stats(F0), stats(F1)

        def iqr(x):
            x = np.asarray(x, float); x = x[np.isfinite(x)] / 1e3
            if x.size < 2:
                return np.nan
            q1, q3 = np.percentile(x, [25, 75])
            return q3 - q1

        drop = sp0["med"] / sp1["med"] if sp1["med"] else np.inf
        kept0 = 100.0 * s0 / t0_ if t0_ else np.nan
        kept1 = 100.0 * s1 / t1_ if t1_ else np.nan
        p90_change = sf1["p90"] / sf0["p90"] if sf0["p90"] else np.nan
        iqr0, iqr1 = iqr(FF0), iqr(FF1)
        narrow = iqr0 / iqr1 if iqr1 else np.nan
        ratio0 = sf0["med"] / sp0["med"] if sp0["med"] else np.nan
        ratio1 = sf1["med"] / sp1["med"] if sp1["med"] else np.nan

        c1 = (drop > 3.0) or (np.isfinite(kept1) and kept1 < 20.0)
        c2 = np.isfinite(p90_change) and 0.5 < p90_change < 2.0
        c3 = np.isfinite(narrow) and narrow > 2.0

        emit(f"\n  gate {g0:.2f} -> {g1:.2f}   ratio {ratio0:.2f} -> {ratio1:.2f}")
        emit(f"  [{'YES' if c1 else 'NO '}] vacuum collapses      "
             f"pre med /{drop:6.1f},  pre kept {kept0:.0f}% -> {kept1:.0f}%")
        emit(f"  [{'YES' if c2 else 'NO '}] strong slices hold    "
             f"flat p90 x{p90_change:5.2f}")
        emit(f"  [{'YES' if c3 else 'NO '}] Fa concentrates       "
             f"flat IQR {iqr0:.2f} -> {iqr1:.2f} kHz (x{narrow:.1f} narrower), "
             f"median -> {np.nanmedian(np.asarray(FF1,float))/1e3:.2f} kHz")

        n_ok = sum((c1, c2, c3))
        if n_ok == 3:
            emit("  VERDICT all three -- plasma signal.")
        elif n_ok == 2:
            emit("  VERDICT two of three -- probably signal; the pass marks are")
            emit("          arbitrary (drop > 3x, kept < 20%, IQR > 2x narrower),")
            emit("          so read the numbers, not the flags.")
        elif n_ok == 1:
            emit("  VERDICT one of three -- not convincing.")
        else:
            emit("  VERDICT none -- no evidence of a mode in this window.")
        if c1 and np.isfinite(kept1) and 0 < kept1 < 20 and s1 <= 2:
            emit(f"  NOTE only {s1} pre-plasma slice(s) survive at gate {g1:.2f};")
            emit(f"       the Fa pre row there is one bin, not a distribution.")

    emit("\n  'pre kept' is the fraction of no-plasma slices still carrying a")
    emit("  finite amplitude.  A gate that works should drive this toward zero.")

    if a.out:
        out = a.out if a.out.endswith(".txt") else a.out + ".txt"
        with open(out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
