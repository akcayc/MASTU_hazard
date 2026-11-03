"""One shot, frequency against time: where the surviving power actually sits.

compare_pre_plasma pools Fa over every slice and reports a median and an IQR.
That summary is wrong for a mode whose frequency CHIRPS: the pooled spread
covers the whole range the mode swept, so the "Fa concentrates" criterion
fails even though each individual slice is perfectly concentrated.  Fa is a
weighted mean over FREQUENCY within one slice -- there is no averaging across
time, so it tracks a chirp faithfully; only the summary loses it.

This draws the thing the summary hides:

  1. the gated spectrogram -- power against frequency and time, with every bin
     that fails the coherence gate or the |n| mask blanked, and Fa(t) on top.
     A real mode is a continuous track.  Noise is scatter.
  2. the amplitude the ladder actually triggers on
  3. the per-slice coherence

With --phase-lock, two more panels test each |n|=b >= 2 as an overtone of n=1:

  4. psi_b(t) = arg( z_b[b k1] conj(z_1[k1])^b ), per slice, darker where the
     slice weighs more in hp.  Flat = phase-locked overtone; scatter = no lock.
  5. hp in a sliding +/- --harmonic-window, the statistic run_master --harmonics
     writes to the ladder, with the noise level 1/sqrt(N_eff) dashed.

    python plot_mode_track.py --shot 49149 --coherence-min 0.6 --out track_49149
    python plot_mode_track.py --shot 49149 --coherence-min 0.6 --t0 0.6 --t1 0.8
    python plot_mode_track.py --shot 51574 --coherence-min 0.8 --n-list 1 2 \
        --phase-lock --t0 0.1854
"""

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, to_rgb

import giopath  # noqa: F401

try:
    from saddle_data import load_omaha_slow, AmplitudeSelector
    from saddle_extras import (_select_bins, amplitude_with_coherence,
                               amplitude_diiid, harmonic_inputs, phase_lock)
    from saddle_analysis import (NTOR_DEFAULT, TIME_INTERVAL_DEFAULT,
                                 FMIN_DEFAULT, FMAX_DEFAULT)
except ImportError:
    giopath.report()
    raise

GAUSS_PER_TESLA = 1.0e4
# Categorical hues, one per |n|, in fixed order.  Slots 1-3 of the reference
# palette validate all-pairs in both modes; the 4th skips yellow (slot 4) for
# violet (slot 7), because yellow against orange fails the all-pairs floors and
# these fields are overlaid, not adjacent.  Each becomes a transparent-to-hue
# ramp so several spectrograms can sit on top of one another.
NCOLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
MARK = "#8a8984"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e4e3de"
FS = dict(title=17.0, subtitle=12.0, axis=13.0, tick=11.5, note=11.5, legend=12.0)
#: correlation of neighbouring Hann-windowed segments at 50% overlap
RHO_HANN50 = 1.0 / 6.0


def running_lock(t, z1, zh, b, half):
    """phase_lock over |t' - t| <= half, at every slice."""
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    return np.array([phase_lock(z1[i:j], zh[i:j], b) for i, j in zip(lo, hi)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int, required=True)
    p.add_argument("--n-list", type=int, nargs="+", default=[1, 2, 3],
                   help="|n| values to overlay; up to 4")
    p.add_argument("--coherence-min", type=float, default=0.6)
    p.add_argument("--nfft", type=int, default=512)
    p.add_argument("--f-min", type=float, default=FMIN_DEFAULT)
    p.add_argument("--f-max", type=float, default=FMAX_DEFAULT)
    p.add_argument("--t0", type=float, help="mark a time, or with --t1 a "
                                             "window start [s]")
    p.add_argument("--t1", type=float, help="mark a window end [s]")
    p.add_argument("--amp-form", choices=["centroid", "perbin"], default="centroid")
    p.add_argument("--track-snr", type=float, default=3.0,
                   help="draw the Fa track only where the amplitude exceeds "
                        "this multiple of that |n|'s own pre-plasma floor; Fa "
                        "is meaningless when no power is concentrated")
    p.add_argument("--phase-lock", action="store_true",
                   help="add the overtone panels: psi_b per slice and the "
                        "running hp, for each |n|>=2 against n=1")
    p.add_argument("--harmonic-window", type=float, default=0.020,
                   help="half-width [s] of the running hp (default 0.020, as "
                        "run_master --harmonics)")
    p.add_argument("--out", default=None)
    p.add_argument("--font-size", type=float, default=FS["axis"],
                   help="axis-label size in points; every other type size "
                        f"scales with it (default {FS['axis']:g})")
    a = p.parse_args()
    fs = {k: v * a.font_size / FS["axis"] for k, v in FS.items()}
    nlist = sorted({abs(int(n)) for n in a.n_list if int(n) != 0})
    if len(nlist) > len(NCOLOURS):
        raise SystemExit(f"at most {len(NCOLOURS)} values of |n|; a 5th series "
                         f"is never a generated hue -- drop one or run twice")
    out = a.out or f"track_{a.shot}"

    sd = load_omaha_slow(a.shot)
    if sd.time.size < 2:
        raise SystemExit(f"no OMAHA data for shot {a.shot}")
    # n=0 is searched as the common-mode sink, exactly as analyze_shot does
    mags = sorted({abs(int(n)) for n in NTOR_DEFAULT if int(n) != 0} | set(nlist))
    search = [0] + [s * m for m in mags for s in (-1, 1)]
    sp = sd.spectrum(a.nfft)
    rm = sp.n_detection(np.asarray(search))

    ti = TIME_INTERVAL_DEFAULT
    series = {}
    sels = {}
    for nabs in nlist:
        sel = AmplitudeSelector(ntor=nabs, time=ti,
                                f_min=np.array([a.f_min] * len(ti)),
                                f_max=np.array([a.f_max] * len(ti)), v_min=0.0)
        sels[nabs] = sel
        got = _select_bins(rm, sel, a.coherence_min, True)
        if got is None:
            raise SystemExit("record too short")
        tsel, Fsel, power, id_mask, _ = got
        if a.amp_form == "perbin":
            t_a, amp = amplitude_diiid(rm, sel, a.coherence_min, True)
            _, Fa, _, _, coh = amplitude_with_coherence(rm, sel, a.coherence_min, True)
        else:
            t_a, Fa, _, amp, coh = amplitude_with_coherence(rm, sel, a.coherence_min,
                                                            True)
        amp = np.asarray(amp, float) * GAUSS_PER_TESLA
        # Fa is a centroid: on noise it wanders over the whole band and tells
        # you nothing, so the track is drawn only where there IS concentrated
        # power.  The floor is this |n|'s own pre-plasma median, the same
        # definition used everywhere else.
        pre = t_a < 0.020
        fl = np.nanmedian(amp[pre & np.isfinite(amp)]) if (pre & np.isfinite(amp)).any() \
            else np.nan
        Fa_t = np.asarray(Fa, float).copy()
        if np.isfinite(fl) and fl > 0:
            Fa_t[~(amp > a.track_snr * fl)] = np.nan
        series[nabs] = dict(t=tsel, F=Fsel, P=np.ma.array(power, mask=id_mask),
                            t_a=t_a, Fa=np.asarray(Fa, float), Fa_track=Fa_t,
                            floor=fl, amp=amp, coh=np.asarray(coh, float))

    orders = [n for n in nlist if n >= 2]
    lock = {}
    if a.phase_lock:
        if not orders:
            raise SystemExit("--phase-lock needs an |n| >= 2 in --n-list")
        if 1 not in sels:
            sels[1] = AmplitudeSelector(ntor=1, time=ti,
                                        f_min=np.array([a.f_min] * len(ti)),
                                        f_max=np.array([a.f_max] * len(ti)),
                                        v_min=0.0)
        h = harmonic_inputs(rm, sp, sels, a.coherence_min, orders)
        z1 = h[1]["z_pk"]
        for b in orders:
            zh = h[b]["z_h"]
            t = series[b]["t_a"]
            lock[b] = dict(psi=np.degrees(np.angle(zh * np.conj(z1) ** b)),
                           w=np.abs(zh) * np.abs(z1) ** b,
                           hp=running_lock(t, z1, zh, b, a.harmonic_window))

    nax = 5 if a.phase_lock else 3
    fig, axes = plt.subplots(
        nax, 1, figsize=(11.0, 10.4 + 4.4 * (nax == 5)), sharex=True,
        gridspec_kw=dict(height_ratios=[2.4, 1.0, 1.0, 1.0, 1.0][:nax],
                         hspace=0.13))
    ax1, ax2, ax3 = axes[:3]
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.105, right=0.975,
                        top=0.858 if nax == 3 else 0.89, bottom=0.075 if nax == 3 else 0.055)

    # --- 1. gated spectrogram per |n|, overlaid, Fa on top ----------------
    allpos = np.concatenate([d["P"].compressed() for d in series.values()])
    allpos = allpos[allpos > 0]
    if allpos.size:
        vmin, vmax = np.percentile(allpos, [40, 99.5])
        norm = LogNorm(vmin=max(vmin, vmax * 1e-4), vmax=vmax)
    else:
        norm = None

    handles = []
    for i, nabs in enumerate(nlist):
        c = NCOLOURS[i]
        r, g, b = to_rgb(c)
        cmap = LinearSegmentedColormap.from_list(
            f"n{nabs}", [(r, g, b, 0.0), (r, g, b, 1.0)])
        d = series[nabs]
        ax1.pcolormesh(d["t"], d["F"] / 1e3, d["P"].T, cmap=cmap, norm=norm,
                       shading="nearest", rasterized=True, zorder=2 + i)
        ax1.plot(d["t_a"], d["Fa_track"] / 1e3, color=c, lw=2.2, alpha=0.95,
                 solid_capstyle="round", zorder=10 + i)
        handles.append(plt.Line2D([], [], color=c, lw=2.4, label=f"|n| = {nabs}"))
    ax1.set_ylabel("frequency  [kHz]", color=INK2, fontsize=fs["axis"])
    ax1.set_ylim(a.f_min / 1e3, a.f_max / 1e3)
    leg = ax1.legend(handles=handles, frameon=False, loc="lower left",
                     bbox_to_anchor=(0.0, 1.015), fontsize=fs["legend"],
                     labelcolor=INK2, ncol=len(nlist), columnspacing=2.0)
    leg.set_zorder(20)


    # --- 2 & 3. amplitude and coherence, one line per |n| -----------------
    for i, nabs in enumerate(nlist):
        c = NCOLOURS[i]
        d = series[nabs]
        ax2.plot(d["t_a"], d["amp"], color=c, lw=2.0, solid_capstyle="round")
        ax3.plot(d["t_a"], d["coh"], color=c, lw=2.0, solid_capstyle="round")
    ax2.set_ylabel(f"amplitude\n[gauss, {a.amp_form}]", color=INK2,
                   fontsize=fs["axis"] * 0.92)
    finite = np.concatenate([series[n]["amp"] for n in nlist])
    if np.isfinite(finite).any() and np.nanmax(finite) > 0:
        ax2.set_yscale("log")
    ax3.axhline(a.coherence_min, color=MARK, lw=1.4, ls=(0, (4, 3)))
    ax3.text(0.002, a.coherence_min, f" gate {a.coherence_min:g}", color=MARK,
             fontsize=fs["note"], va="top", ha="left",
             transform=ax3.get_yaxis_transform())
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("coherence", color=INK2, fontsize=fs["axis"] * 0.92)

    # --- 4 & 5. overtone test against n=1 ---------------------------------
    if a.phase_lock:
        ax4, ax5 = axes[3:]
        for b in orders:
            c = NCOLOURS[nlist.index(b)]
            L = lock[b]
            ok = np.isfinite(L["psi"]) & np.isfinite(L["w"])
            wmax = np.percentile(L["w"][ok], 99) if ok.any() else 1.0
            rgba = np.zeros((ok.sum(), 4))
            rgba[:, :3] = to_rgb(c)
            rgba[:, 3] = np.clip(L["w"][ok] / wmax, 0.04, 1.0)
            ax4.scatter(series[b]["t_a"][ok], L["psi"][ok], s=7, c=rgba,
                        linewidths=0, rasterized=True)
            ax5.plot(series[b]["t_a"], L["hp"], color=c, lw=2.0,
                     solid_capstyle="round", label=f"|n| = {b}")
        ax4.set_ylim(-180, 180)
        ax4.set_yticks([-180, -90, 0, 90, 180])
        ax4.set_ylabel("$\\psi_b$ = arg $z_b\\,\\bar z_1^{\\,b}$\n[deg]",
                       color=INK2, fontsize=fs["axis"] * 0.92)
        stride = float(np.median(np.diff(series[orders[0]]["t_a"])))
        n_eff = (2 * a.harmonic_window / stride) / (1 + 2 * RHO_HANN50)
        ax5.axhline(1 / np.sqrt(n_eff), color=MARK, lw=1.4, ls=(0, (4, 3)))
        ax5.text(0.002, 1 / np.sqrt(n_eff), f" noise 1/$\\sqrt{{N_{{eff}}}}$ = "
                 f"{1 / np.sqrt(n_eff):.2f}", color=MARK, fontsize=fs["note"],
                 va="bottom", ha="left", transform=ax5.get_yaxis_transform())
        ax5.set_ylim(0, 1)
        ax5.set_ylabel(f"hp, $\\pm${1e3 * a.harmonic_window:g} ms",
                       color=INK2, fontsize=fs["axis"] * 0.92)
        ax4.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
        ax5.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    axes[-1].set_xlabel("time  [s]", color=INK2, fontsize=fs["axis"], labelpad=7)

    for ax in axes:
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=fs["tick"])
        if a.t0 is not None and a.t1 is not None:
            ax.axvspan(a.t0, a.t1, color=MARK, alpha=0.08, zorder=0)
            for tv in (a.t0, a.t1):
                ax.axvline(tv, color=MARK, lw=1.2, ls=(0, (4, 3)), alpha=0.7)
        elif a.t0 is not None:
            ax.axvline(a.t0, color=MARK, lw=1.2, ls=(0, (4, 3)), alpha=0.7)
    ax2.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax3.grid(True, which="major", color=GRID, lw=0.8, zorder=0)

    fig.text(0.105, 0.966 if nax == 3 else 0.975, f"Shot {a.shot}", color=INK,
             fontsize=fs["title"], fontweight="semibold", ha="left")
    fig.text(0.105, 0.934 if nax == 3 else 0.95,
             f"colour = which |n| claimed the bin, intensity = power;  bins "
             f"failing the mask or the {a.coherence_min:g} gate are blank;  "
             f"$F_a$ drawn above {a.track_snr:g}x floor",
             color=INK3, fontsize=fs["subtitle"] * 0.92, ha="left")

    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=170, facecolor=SURFACE, bbox_inches="tight")

    print(f"  shot {a.shot}, gate {a.coherence_min:g}, "
          f"|n| = {', '.join(str(n) for n in nlist)}")
    for nabs in nlist:
        Fa = series[nabs]["Fa"]
        f = np.isfinite(Fa)
        if not f.any():
            print(f"    |n|={nabs}: nothing survives the gate")
            continue
        tr = np.isfinite(series[nabs]["Fa_track"])
        if tr.any():
            q1, q3 = np.percentile(series[nabs]["Fa_track"][tr] / 1e3, [25, 75])
            print(f"    |n|={nabs}: track on {100*tr.mean():5.1f}% of slices,"
                  f"  Fa median {np.nanmedian(series[nabs]['Fa_track'][tr])/1e3:6.2f}"
                  f" kHz  IQR [{q1:5.2f}, {q3:5.2f}]")
        else:
            print(f"    |n|={nabs}: no slice above {a.track_snr:g}x floor "
                  f"-- nothing to track")
    print("    a wide IQR with a continuous track in panel 1 is a CHIRP, not noise")
    if a.phase_lock and a.t0 is not None:
        for b in orders:
            k = int(np.argmin(np.abs(series[b]["t_a"] - a.t0)))
            print(f"    |n|={b}: hp at t0 = {lock[b]['hp'][k]:.2f}  "
                  f"(+/-{1e3 * a.harmonic_window:g} ms)")
    print(f"  wrote {out}.png  {out}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
