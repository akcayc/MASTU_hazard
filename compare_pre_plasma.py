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
    p.add_argument("--amp-form", choices=["perbin", "centroid"],
                   default="perbin")
    a = p.parse_args()
    shots = a.shots or list(range(a.shot_min, a.shot_max))
    key = "amp_centroid" if a.amp_form == "centroid" else "amp_perbin"

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

    fin = [(g, r) for g, r in ratios if np.isfinite(r)]
    print()
    if not fin:
        print("  no verdict: the gate removed everything in one window")
    else:
        lo, hi = fin[0][1], fin[-1][1]
        print(f"  ratio at gate {fin[0][0]:.2f}: {lo:.2f}   "
              f"at gate {fin[-1][0]:.2f}: {hi:.2f}")
        if hi > 3.0 * lo and hi > 3.0:
            print("  VERDICT the gate separates plasma from vacuum -- the")
            print("          surviving coherent power is plasma signal.")
        elif hi < 1.5 * lo:
            print("  VERDICT gating does NOT separate plasma from vacuum.")
            print("          Coherent power of the same size is present with no")
            print("          plasma, so it is instrumental, not MHD.  Check the")
            print("          Fa table above: a fixed frequency in both windows")
            print("          names the line.  The coherence gate is isolating")
            print("          that line, and the ladder is detecting it.")
        else:
            print("  VERDICT partial separation -- some plasma signal, but the")
            print("          vacuum window is not clean.  Look at the Fa table.")
    print("\n  'pre kept' is the fraction of no-plasma slices still carrying a")
    print("  finite amplitude.  A gate that works should drive this toward zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
