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

import numpy as np

import giopath  # noqa: F401

try:
    import pyuda
    from saddle_extras import analyze_shot
    from saddle_analysis import NTOR_DEFAULT
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
    p.add_argument("--n-detect", type=int, default=1, help="|n| to report")
    p.add_argument("--nfft", type=int, default=512)
    p.add_argument("--t-pre", type=float, default=0.020,
                   help="end of the no-plasma window [s]")
    p.add_argument("--t0", type=float, help="fixed flat-top start, else from Ip")
    p.add_argument("--t1", type=float, help="fixed flat-top end")
    p.add_argument("--amp-form", choices=["centroid", "perbin"],
                   default="centroid")
    a = p.parse_args()
    shots = a.shots or list(range(a.shot_min, a.shot_max))
    key = "amp_perbin" if a.amp_form == "perbin" else "amp_centroid"

    client = pyuda.Client()
    # rows[gate] = (pre amplitudes, flat amplitudes, pre freqs, flat freqs,
    #               pre slices surviving, pre slices total)
    acc = {g: ([], [], [], [], 0, 0) for g in a.gates}
    nshot, win_kind = 0, None

    for shot in shots:
        try:
            ta, tb, win_kind = flattop_window(shot, client, a.t0, a.t1)
            for g in a.gates:
                per_n, _ = analyze_shot(shot, NFFT=a.nfft, ntor=NTOR_DEFAULT,
                                        coh_min=g)
                m = per_n.get(abs(a.n_detect))
                if m is None:
                    continue
                t = np.asarray(m["time"], float)
                amp = np.asarray(m[key], float) * GAUSS_PER_TESLA
                frq = np.asarray(m["freq"], float)
                pre = t < a.t_pre
                flat = (t >= ta) & (t <= tb)
                P, F, PF, FF, surv, tot = acc[g]
                P.extend(amp[pre]); F.extend(amp[flat])
                PF.extend(frq[pre]); FF.extend(frq[flat])
                acc[g] = (P, F, PF, FF,
                          surv + int(np.isfinite(amp[pre]).sum()),
                          tot + int(pre.sum()))
            nshot += 1
        except Exception as e:
            print(f"  skip {shot}: {type(e).__name__}: {e}")

    if not nshot:
        raise SystemExit("no shots analysed")

    print(f"\n{nshot} shots, |n|={abs(a.n_detect)}, amp form '{a.amp_form}', "
          f"window from {win_kind}")
    print(f"pre-plasma is t < {a.t_pre*1e3:.0f} ms -- no plasma, so any "
          f"amplitude there is instrumental\n")
    print(f"  {'gate':>5} {'pre med':>10} {'flat med':>10} {'ratio':>7}"
          f" {'pre p90':>10} {'flat p90':>10} {'pre kept':>9}")
    ratios = []
    for g in a.gates:
        P, F, PF, FF, surv, tot = acc[g]
        sp, sf = stats(P), stats(F)
        r = sf["med"] / sp["med"] if sp["med"] and np.isfinite(sp["med"]) else np.nan
        ratios.append((g, r))
        kept = 100.0 * surv / tot if tot else np.nan
        print(f"  {g:5.2f} {sp['med']:10.3e} {sf['med']:10.3e} {r:7.2f}"
              f" {sp['p90']:10.3e} {sf['p90']:10.3e} {kept:8.1f}%")

    print(f"\n  {'gate':>5} {'Fa pre kHz (med, IQR)':>28} "
          f"{'Fa flat kHz (med, IQR)':>28}")
    for g in a.gates:
        _, _, PF, FF, _, _ = acc[g]
        def fs(x):
            x = np.asarray(x, float); x = x[np.isfinite(x)] / 1e3
            if x.size == 0:
                return "         --"
            q1, q3 = np.percentile(x, [25, 75])
            return f"{np.median(x):8.2f}  [{q1:6.2f},{q3:6.2f}]"
        print(f"  {g:5.2f} {fs(PF):>28} {fs(FF):>28}")

    # Three independent signatures, not one ratio.  A real mode shows all
    # three; any one alone can be produced by an artefact.
    g0, g1 = a.gates[0], a.gates[-1]
    P0, F0, PF0, FF0, s0, t0_ = acc[g0]
    P1, F1, PF1, FF1, s1, t1_ = acc[g1]
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

    c1 = (drop > 3.0) or (np.isfinite(kept1) and kept1 < 20.0)
    c2 = np.isfinite(p90_change) and 0.5 < p90_change < 2.0
    c3 = np.isfinite(narrow) and narrow > 2.0

    print(f"\n  gate {g0:.2f} -> {g1:.2f}")
    print(f"  [{'YES' if c1 else 'NO '}] vacuum collapses      "
          f"pre med /{drop:6.1f},  pre kept {kept0:.0f}% -> {kept1:.0f}%")
    print(f"      nothing coherent survives where there is no plasma")
    print(f"  [{'YES' if c2 else 'NO '}] strong slices hold    "
          f"flat p90 x{p90_change:5.2f}")
    print(f"      gating leaves the big excursions alone, so their power is")
    print(f"      coherent; a rise in flat med is then slices dropping out,")
    print(f"      not the amplitude inflating")
    print(f"  [{'YES' if c3 else 'NO '}] Fa concentrates       "
          f"flat IQR {iqr0:.2f} -> {iqr1:.2f} kHz (x{narrow:.1f} narrower), "
          f"median -> {np.nanmedian(np.asarray(FF1,float))/1e3:.2f} kHz")
    print(f"      the surviving power sits at one frequency, not spread")

    n = sum((c1, c2, c3))
    print()
    if n == 3:
        print("  VERDICT all three -- plasma signal.  The mode frequency is the")
        print("          converged flat-top Fa above.")
    elif n == 2:
        print("  VERDICT two of three -- probably signal, but check which failed.")
    elif n == 1:
        print("  VERDICT one of three -- not convincing.  One signature alone is")
        print("          reachable by an artefact.")
    else:
        print("  VERDICT none -- no evidence of a mode in this window.")
    if c1 and np.isfinite(kept1) and 0 < kept1 < 20 and s1 <= 2:
        print(f"  NOTE only {s1} pre-plasma slice(s) survive at gate {g1:.2f};")
        print(f"       the Fa pre row there is one bin, not a distribution.")

    print("\n  'pre kept' is the fraction of no-plasma slices still carrying a")
    print("  finite amplitude.  A gate that works should drive this toward zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
