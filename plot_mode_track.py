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

    python plot_mode_track.py --shot 49149 --coherence-min 0.6 --out track_49149
    python plot_mode_track.py --shot 49149 --coherence-min 0.6 --t0 0.6 --t1 0.8
"""

import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import giopath  # noqa: F401

try:
    from saddle_data import load_omaha_slow, AmplitudeSelector
    from saddle_extras import _select_bins, amplitude_with_coherence, amplitude_diiid
    from saddle_analysis import (NTOR_DEFAULT, TIME_INTERVAL_DEFAULT,
                                 FMIN_DEFAULT, FMAX_DEFAULT)
except ImportError:
    giopath.report()
    raise

GAUSS_PER_TESLA = 1.0e4
# sequential = one hue light->dark for magnitude; categorical slot 2 for the
# overlay so the track reads against it.  Never a rainbow for power.
SEQ, TRACK, MARK = "Blues", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, GRID = "#fcfcfb", "#e4e3de"
FS = dict(title=17.0, subtitle=12.0, axis=13.0, tick=11.5, note=11.5, legend=12.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int, required=True)
    p.add_argument("--n-detect", type=int, default=1, help="|n| to track")
    p.add_argument("--coherence-min", type=float, default=0.6)
    p.add_argument("--nfft", type=int, default=512)
    p.add_argument("--f-min", type=float, default=FMIN_DEFAULT)
    p.add_argument("--f-max", type=float, default=FMAX_DEFAULT)
    p.add_argument("--t0", type=float, help="mark a window start [s]")
    p.add_argument("--t1", type=float, help="mark a window end [s]")
    p.add_argument("--amp-form", choices=["centroid", "perbin"], default="centroid")
    p.add_argument("--out", default=None)
    p.add_argument("--font-scale", type=float, default=1.0)
    a = p.parse_args()
    fs = {k: v * a.font_scale for k, v in FS.items()}
    out = a.out or f"track_{a.shot}"
    nabs = abs(a.n_detect)

    sd = load_omaha_slow(a.shot)
    if sd.time.size < 2:
        raise SystemExit(f"no OMAHA data for shot {a.shot}")
    # n=0 is searched as the common-mode sink, exactly as analyze_shot does
    mags = sorted({abs(int(n)) for n in NTOR_DEFAULT if int(n) != 0})
    search = [0] + [s * m for m in mags for s in (-1, 1)]
    rm = sd.spectrum(a.nfft).n_detection(np.asarray(search))

    ti = TIME_INTERVAL_DEFAULT
    sel = AmplitudeSelector(ntor=nabs, time=ti,
                            f_min=np.array([a.f_min] * len(ti)),
                            f_max=np.array([a.f_max] * len(ti)), v_min=0.0)

    got = _select_bins(rm, sel, a.coherence_min, True)
    if got is None:
        raise SystemExit("record too short")
    tsel, Fsel, power, id_mask, _ = got
    P = np.ma.array(power, mask=id_mask)             # (time, freq), gated

    if a.amp_form == "perbin":
        t_a, amp = amplitude_diiid(rm, sel, a.coherence_min, True)
        _, Fa, _, _, coh = amplitude_with_coherence(rm, sel, a.coherence_min, True)
    else:
        t_a, Fa, _, amp, coh = amplitude_with_coherence(rm, sel, a.coherence_min, True)
    amp = np.asarray(amp, float) * GAUSS_PER_TESLA

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11.0, 10.4), sharex=True,
        gridspec_kw=dict(height_ratios=[2.4, 1.0, 1.0], hspace=0.13))
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.105, right=0.90, top=0.885, bottom=0.075)

    # --- 1. gated spectrogram, Fa on top ----------------------------------
    pos = P.compressed()
    pos = pos[pos > 0]
    if pos.size:
        vmin, vmax = np.percentile(pos, [5, 99.5])
        norm = LogNorm(vmin=max(vmin, vmax * 1e-5), vmax=vmax)
    else:
        norm = None
    mesh = ax1.pcolormesh(tsel, Fsel / 1e3, P.T, cmap=SEQ, norm=norm,
                          shading="nearest", rasterized=True)
    # The colourbar must take its width from a divider, not from `ax=ax1`:
    # the latter shrinks ax1 alone and the panels stop lining up vertically,
    # which defeats sharex.  ax2 and ax3 get a matching invisible slot.
    cax = make_axes_locatable(ax1).append_axes("right", size="2%", pad=0.10)
    for ax in (ax2, ax3):
        spacer = make_axes_locatable(ax).append_axes("right", size="2%", pad=0.10)
        spacer.set_axis_off()
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("power of surviving bins", color=INK2, fontsize=fs["note"])
    cb.ax.tick_params(colors=INK2, labelsize=fs["tick"] * 0.85)
    ax1.plot(t_a, np.asarray(Fa, float) / 1e3, color=TRACK, lw=1.6, alpha=0.9,
             solid_capstyle="round", zorder=5, label="$F_a$")
    ax1.set_ylabel("frequency  [kHz]", color=INK2, fontsize=fs["axis"])
    ax1.set_ylim(a.f_min / 1e3, a.f_max / 1e3)
    ax1.legend(frameon=False, loc="upper right", fontsize=fs["legend"],
               labelcolor=INK2)

    # --- 2. the amplitude the ladder triggers on --------------------------
    ax2.plot(t_a, amp, color=INK2, lw=2.0, solid_capstyle="round")
    ax2.set_ylabel(f"amplitude\n[gauss, {a.amp_form}]", color=INK2,
                   fontsize=fs["axis"] * 0.92)
    if np.isfinite(amp).any() and np.nanmax(amp) > 0:
        ax2.set_yscale("log")

    # --- 3. per-slice coherence -------------------------------------------
    ax3.plot(t_a, np.asarray(coh, float), color=INK2, lw=2.0,
             solid_capstyle="round")
    ax3.axhline(a.coherence_min, color=MARK, lw=1.4, ls=(0, (4, 3)))
    ax3.text(0.995, a.coherence_min, f"gate {a.coherence_min:g} ", color=MARK,
             fontsize=fs["note"], va="bottom", ha="right",
             transform=ax3.get_yaxis_transform())
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("coherence", color=INK2, fontsize=fs["axis"] * 0.92)
    ax3.set_xlabel("time  [s]", color=INK2, fontsize=fs["axis"], labelpad=7)

    for ax in (ax1, ax2, ax3):
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
    ax2.grid(True, which="major", color=GRID, lw=0.8, zorder=0)
    ax3.grid(True, which="major", color=GRID, lw=0.8, zorder=0)

    fig.text(0.105, 0.957, f"Shot {a.shot}  |n| = {nabs}", color=INK,
             fontsize=fs["title"], fontweight="semibold", ha="left")
    fig.text(0.105, 0.923,
             f"bins failing the |n| mask or the {a.coherence_min:g} coherence "
             f"gate are blank - a real mode is a continuous track",
             color=INK3, fontsize=fs["subtitle"], ha="left")

    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=170, facecolor=SURFACE, bbox_inches="tight")

    f = np.isfinite(Fa)
    print(f"  shot {a.shot}, |n|={nabs}, gate {a.coherence_min:g}")
    print(f"    slices with surviving bins: {100*f.mean():.1f}%")
    if f.any():
        q1, q3 = np.percentile(np.asarray(Fa)[f] / 1e3, [25, 75])
        print(f"    Fa median {np.nanmedian(np.asarray(Fa)[f])/1e3:.2f} kHz,"
              f" IQR [{q1:.2f}, {q3:.2f}] -- wide IQR with a continuous track"
              f" in panel 1 means a CHIRP, not noise")
    print(f"  wrote {out}.png  {out}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
