"""
MAST-U tearing-mode hazard study -- main driver.

Stage A of the DIII-D framework, ported.  The organising principle there is
that a detector never emits a label: it emits, per shot, the *first crossing
time as a function of a threshold ladder*.  The threshold then becomes a
post-processing parameter and the sensitivity study costs no recomputation
over the database.  That is what this driver produces.

Pipeline per shot:
    collect_signals()   fetch Ip, OMAHA modes, Dalpha, density, Te
      -> flat-top window from Ip                       (flattop.find_flattop)
      -> OMAHA n-resolved amplitude + coherence        (saddle_extras)
      -> auxiliary traces, restricted to the window
    detect_ladder()     first crossing time per threshold, inside the window

Output: one row per shot, columns rm<level> holding the first crossing time
(NaN if never crossed), plus tsof/teof/isok -- the direct analogue of DIII-D's
fdp-regen-brm-plain-*.csv, and the input to the joint annotation stage.

STATUS.  Runs only on Freya (needs pyuda and Gio's saddle_data/omaha_coils/
mode_functions).  Two things are known-missing before the ladder means
anything physically:
  * absolute amplitude calibration -- see saddle_extras.analyze_shot
  * a locked-mode detector -- the spectral path cannot see a DC island, and
    the freq<2kHz test is not a substitute.  Until then the pathway model
    degenerates to rotating-only / uneventful.
"""

import argparse
import json

import numpy as np
import pandas as pd

import giopath  # noqa: F401  -- puts Giovannozzi's modules on sys.path

try:
    import pyuda

    from saddle_analysis import NTOR_DEFAULT
    from saddle_extras import analyze_shot, noise_floor
    from flattop import find_flattop, FlatTopConfig
except ImportError:
    giopath.report()
    raise


# ---------------------------------------------------------------------------
# Threshold ladder, in the amplitude's own units.
#
# An earlier version normalised by each shot's pre-plasma floor, on the
# grounds that the per-shot median amplitude varied by ~10x.  That was
# measured over the whole record, which contains the plasma.  Measured where
# there IS no plasma -- t < 20 ms, so instrumental by construction -- the
# floor varies by only 1.88x across the 78 shots of 47000-47099 (12% rel.
# std dev, no shot above 3x the median).  The factor of ten was physics.
#
# So the threshold is absolute.  A fixed absolute level is then a consistent
# criterion to ~12%, whereas a fixed signal-to-floor level would vary by
# 1.88x in absolute terms -- inconsistent in the direction that matters, and
# it would divide out the cross-shot variation the hazard model learns from.
#
# Units: GAUSS.  ModeAmplitude.amplitude is nominally tesla, so it is scaled
# by GAUSS_PER_TESLA here -- the same convention as the DIII-D detector, which
# applies its own 1.0e4 for exactly this reason.  The reading is settled by
# plausibility: as stored, the noise floor is 1.8e-07 and the largest
# excursion 3.0e-05.  Taken as gauss that floor would be 0.018 nT, far below
# any real magnetic noise.  Taken as tesla it is 1.8 mG with a 0.30 G peak,
# which is what saddle coils should see.
#
# Still not an absolute calibration.  Giovannozzi: "I'm saying T and T/s but,
# really, I'm not sure that the data I'm reading are in absolute units, so use
# it for comparing shots."  So these gauss are consistent between shots --
# which is all a ladder needs -- but carry an unverified O(1) factor from the
# missing specgram window-power normalisation, and should not yet be compared
# against published DIII-D numbers.  Settling that factor relabels every
# column by one constant; no re-detection.
#
# Log spacing because the usable range is wide: 1.8 mG to 0.30 G is 160x.
# ---------------------------------------------------------------------------
GAUSS_PER_TESLA = 1.0e4

LADDER_MIN = 2.0e-3      # gauss
LADDER_MAX = 3.0e-1      # gauss
LADDER_NUM = 61
DEFAULT_LADDER = np.logspace(np.log10(LADDER_MIN), np.log10(LADDER_MAX), LADDER_NUM)


def schmitt_first_crossing(time, signal, level, debounce=0.008):
    """First upward crossing of `level`, confirmed by `debounce` seconds.

    Reproduces the DIII-D trigger: no hysteresis (both Schmitt levels equal),
    and the reported time is back-dated to the crossing, not the confirmation.
    Returns NaN if never crossed.
    """
    time = np.asarray(time, float)
    signal = np.asarray(signal, float)
    above = np.isfinite(signal) & (signal >= level)
    if not above.any():
        return np.nan

    t_cross, held = None, 0.0
    for i in range(1, time.size):
        dt = time[i] - time[i - 1]
        if above[i]:
            if t_cross is None:
                t_cross = time[i]
                held = 0.0
            else:
                held += dt
            if held >= debounce:
                return float(t_cross)
        else:
            t_cross, held = None, 0.0
    return np.nan


def collect_signals(shot, client, t_start=0.2, ntor_list=None, NFFT=512,
                    ft_cfg=None, coherence_min=0.0, want_profiles=False,
                    normalize_floor=False):
    """Fetch everything one shot contributes, on that shot's flat-top window.

    Returns a dict; `ok` False means the shot is to be dropped, with `reason`
    saying why.  Nothing here thresholds or labels -- that is detect_ladder's
    job.
    """
    out = {"shot": shot, "ok": False, "reason": ""}

    # --- Ip and the analysis window -------------------------------------
    # measured current, not the EPM reconstruction: the window must exist even
    # where the equilibrium did not converge.
    try:
        v = client.get("/AMC/PLASMA_CURRENT", shot)
        t_ip, ip = np.asarray(v.time.data), 1e-3 * np.asarray(v.data)   # kA -> MA
    except Exception as e:
        out["reason"] = f"no Ip: {e}"
        return out

    ft = find_flattop(t_ip, ip, ft_cfg or FlatTopConfig())
    if not ft.ok:
        out["reason"] = f"flat-top: {ft.reason}"
        return out
    t_a, t_b = ft.t_a, ft.t_b
    out.update(ip_mean=ft.ip_mean, ft_slope=ft.slope, ft_nrmse=ft.nrmse,
               t_flattop=(ft.t_a, ft.t_b))

    # --- OMAHA mode analysis --------------------------------------------
    # both signs: n_detection projects onto exp(+i n phi), so a positive-only
    # list sees one rotation direction only.  n=0 is axisymmetric, so excluded.
    ntor = np.array([n for n in (ntor_list or NTOR_DEFAULT) if n != 0])
    try:
        modes, meta = analyze_shot(shot, NFFT=NFFT, ntor=ntor,
                                   coh_min=coherence_min)
    except Exception as e:
        out["reason"] = f"OMAHA: {e}"
        return out
    out["modes"] = modes
    out["meta"] = meta

    # The magnetics record is finite and can be SHORTER than the flat-top: the
    # OMAHA slow channels cover a fixed ~1.0 s window, while MAST-U flat-tops
    # can run to ~1.5 s.  The observed window is the intersection, and it is
    # the intersection that must be recorded -- writing the Ip-derived t_b
    # would overstate the dwell time, biasing the base hazard downward at
    # exactly the late times where accumulated-stress events live.  An event
    # after the record ends is a censored shot, not an eventless one.
    t_omaha = modes[int(ntor[0])]["time"]
    if t_omaha.size < 2:
        out["reason"] = "empty OMAHA time base"
        return out
    t_a = max(t_a, float(t_omaha[0]))
    t_b = min(t_b, float(t_omaha[-1]))
    if not (t_b > t_a):
        out["reason"] = (f"flat-top [{ft.t_a:.4f}, {ft.t_b:.4f}] does not overlap "
                         f"OMAHA record [{t_omaha[0]:.4f}, {t_omaha[-1]:.4f}]")
        return out
    out.update(tsof=t_a, teof=t_b,
               t_record=(float(t_omaha[0]), float(t_omaha[-1])),
               truncated=bool(ft.t_b > t_omaha[-1] + 1e-9),
               lost_exposure=float(max(0.0, ft.t_b - t_omaha[-1])))

    # normalised detector trace, per n, restricted to the flat-top window and
    # gated on the toroidal-fit coherence.
    traces = {}
    for n in ntor:
        m = modes[n]
        t, amp, coh = m["time"], np.asarray(m["amp"], float), np.asarray(m["coherence"], float)
        floor = m["noise_floor"]
        if not np.isfinite(floor) or floor <= 0:
            floor = noise_floor(t, amp, t_pre=min(0.020, max(t_a * 0.5, 1e-3)))
        if not np.isfinite(floor) or floor <= 0:
            continue
        amp = amp * GAUSS_PER_TESLA          # -> gauss
        floor = floor * GAUSS_PER_TESLA
        # No slice-level filter here: coherence_min was applied per frequency
        # bin inside the amplitude sum (saddle_extras.amplitude_with_coherence),
        # which is where it belongs.  Gating slices as well would double-count.
        sig = amp / floor if normalize_floor else amp
        win = (t >= t_a) & (t <= t_b)
        traces[int(n)] = {"time": t[win], "amp": sig[win], "coherence": coh[win],
                          "floor": floor}
    if not traces:
        out["reason"] = "no usable OMAHA trace"
        return out
    out["traces"] = traces

    # --- auxiliary signals, for the feature table later -----------------
    aux = {}

    def _grab(key, channel, time_channel=None, factor=1.0):
        try:
            v = client.get(channel, shot)
            tt = np.asarray(client.get(time_channel, shot).data) if time_channel \
                else np.asarray(v.time.data)
            aux[key] = {"time": tt, "data": factor * np.asarray(v.data)}
        except Exception as e:
            aux[key] = {"error": str(e)}

    _grab("dalpha_t", "XIM/DA/HM10/T", "XIM/TIME1")
    _grab("ne_dl", "ANE/DENSITY", "ANE/TIME1", factor=1e-20)
    if "data" in aux.get("ne_dl", {}):
        a = aux["ne_dl"]
        if a["time"].size == a["data"].size and a["time"].size > 2:
            a["dne_dt"] = np.gradient(a["data"], a["time"], edge_order=2)

    if want_profiles:
        # Thomson: 2-D (Nt, Nr).  Radial gradients are expensive over a whole
        # shot; take them near the event once the labels exist, not here.
        try:
            aux["Te"] = {"data": np.asarray(client.get("AYC/T_E", shot).data),
                         "R": np.asarray(client.get("AYC/R", shot).data)}
        except Exception as e:
            aux["Te"] = {"error": str(e)}

    out["aux"] = aux
    out["ok"] = True
    out["reason"] = "ok"
    return out


def detect_ladder(sig, ladder, n_detect=1, debounce=0.008):
    """First crossing time of the n=`n_detect` trace, for every ladder level."""
    tr = sig["traces"].get(int(n_detect))
    if tr is None:
        # Returning NaN here would write a CSV full of nan with no explanation.
        raise ValueError(
            f"no trace for n={n_detect} on shot {sig.get('shot')}; "
            f"available: {sorted(sig['traces'])}")
    return {f"rm{lv:.5f}": schmitt_first_crossing(tr["time"], tr["amp"], lv, debounce)
            for lv in ladder}


def build_rm_detector(shots, client, ladder=None, ntor_list=None, NFFT=512,
                      t_start=0.2, n_detect=1, debounce=0.008,
                      coherence_min=0.0, ft_cfg=None, normalize_floor=False):
    """Run the detector over `shots` and return the ladder table."""
    ladder = DEFAULT_LADDER if ladder is None else np.asarray(ladder)

    # The ladder triggers on one n, which must be among those analysed --
    # otherwise every row comes back NaN and the cause is invisible.  Check
    # before touching the database, not per shot.
    ntor = [n for n in (ntor_list or NTOR_DEFAULT) if n != 0]
    if int(n_detect) not in [int(n) for n in ntor]:
        raise ValueError(
            f"--n-detect {n_detect} is not in --n-tor {sorted(int(n) for n in ntor)}. "
            f"The ladder triggers on one toroidal mode number, which must be one "
            f"of those analysed. Add {n_detect} to --n-tor, or pick a different "
            f"--n-detect.")

    rows, dropped = [], {}

    for shot in shots:
        sig = collect_signals(shot, client, t_start=t_start, ntor_list=ntor_list,
                              NFFT=NFFT, ft_cfg=ft_cfg, coherence_min=coherence_min,
                              normalize_floor=normalize_floor)
        if not sig["ok"]:
            dropped[shot] = sig["reason"]
            print(f"{shot}: dropped -- {sig['reason']}")
            continue
        row = {"shot": shot, "isok_rm": 1, "whichn": n_detect,
               "ta_rm": sig["tsof"], "tb_rm": sig["teof"],
               "ip_mean": sig["ip_mean"], "ft_nrmse": sig["ft_nrmse"],
               "floor": sig["traces"][int(n_detect)]["floor"],
               "truncated": int(sig["truncated"]),
               "lost_exposure": sig["lost_exposure"]}
        row.update(detect_ladder(sig, ladder, n_detect, debounce))
        rows.append(row)
        print(f"{shot}: ok  window [{sig['tsof']:.4f}, {sig['teof']:.4f}] s")

    df = pd.DataFrame(rows)
    return {"table": df, "dropped": dropped, "ladder": ladder}


def run_master(shots, build_detector=False, thresholds=None, ntor_list=None,
               NFFT=512, t_start=0.2, out_csv=None, coherence_min=0.0,
               debounce=0.008, n_detect=1, normalize_floor=False,
               ladder_spec=None):
    client = pyuda.Client()
    outputs = {}

    if build_detector:
        ladder = None
        if thresholds and "ladder" in thresholds:
            ladder = np.asarray(thresholds["ladder"], float)
        elif ladder_spec is not None:
            lo, hi, num = ladder_spec
            ladder = np.logspace(np.log10(lo), np.log10(hi), num)
        det = build_rm_detector(shots=shots, client=client, ladder=ladder,
                                ntor_list=ntor_list, NFFT=NFFT, t_start=t_start,
                                n_detect=n_detect, debounce=debounce,
                                coherence_min=coherence_min,
                                normalize_floor=normalize_floor)
        outputs["detector"] = det
        if out_csv:
            det["table"].to_csv(out_csv, index=False)
            print(f"\nwrote {len(det['table'])} rows -> {out_csv}"
                  f"  ({len(det['dropped'])} shots dropped)")

    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])

    parser.add_argument("--shots", type=int, nargs="+", required=True)
    parser.add_argument("--build-detector", action="store_true")

    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--n-tor", type=int, nargs="+", default=NTOR_DEFAULT,
                        help="toroidal mode numbers to search; both signs")
    parser.add_argument("--n-detect", type=int, default=1,
                        help="toroidal mode number the ladder triggers on")
    parser.add_argument("--t-start", type=float, default=0.2)
    parser.add_argument("--debounce", type=float, default=0.008,
                        help="trigger confirmation time [s]")
    parser.add_argument("--coherence-min", type=float, default=0.0,
                        help="zero frequency bins whose toroidal-fit coherence is "
                             "below this, before the amplitude sum")
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument("--normalize-floor", action="store_true",
                        help="divide the amplitude by each shot's pre-plasma "
                             "floor and use a signal-to-floor ladder; off by "
                             "default, see the note on DEFAULT_LADDER")
    parser.add_argument("--ladder-min", type=float, default=LADDER_MIN)
    parser.add_argument("--ladder-max", type=float, default=LADDER_MAX)
    parser.add_argument("--ladder-num", type=int, default=LADDER_NUM)

    parser.add_argument("--thresholds", type=str)
    parser.add_argument("--thresholds-file", type=str)

    args = parser.parse_args()

    thresholds = {}
    if args.thresholds:
        thresholds = json.loads(args.thresholds)
    elif args.thresholds_file:
        with open(args.thresholds_file) as f:
            thresholds = json.load(f)

    run_master(
        shots=args.shots,
        build_detector=args.build_detector,
        thresholds=thresholds,
        ntor_list=args.n_tor,
        NFFT=args.nfft,
        t_start=args.t_start,
        out_csv=args.out_csv,
        coherence_min=args.coherence_min,
        debounce=args.debounce,
        n_detect=args.n_detect,
        normalize_floor=args.normalize_floor,
        ladder_spec=(args.ladder_min, args.ladder_max, args.ladder_num),
    )


if __name__ == "__main__":
    main()
