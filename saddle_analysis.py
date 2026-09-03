"""
OMAHA saddle-coil mode analysis for MAST-U.

This is the original routine with four defects repaired and nothing else
changed.  The signature, the output keys, and the pickle layout are exactly as
they were, so existing readers and existing .pkl files keep working.  Anything
genuinely new lives in saddle_extras.py.

Repairs:
  1. The batch driver ran at module level, so `from saddle_analysis import
     saddle_analysis_one_shot` triggered a full database sweep and a pickle
     write on import.  It now sits under `if __name__ == "__main__"`.
  2. `time_interval`, `freq_min` and `freq_max` were read as module globals
     from inside the function, so calling it from any other module raised
     NameError.  They are optional keyword arguments now, appended *after* the
     original four positional parameters, defaulting to the same values.
  3. The batch loop caught every exception and `pass`ed, so dropped shots left
     no record of why.  Reasons are now written to a JSON sidecar; the pickle
     itself is unchanged.
  4. A dead `amp = modes[itor].amplitude` at the end of the loop body.
"""

import argparse
import json
import pickle
import sys

import numpy as np

if sys.version_info[:2] < (3, 9):
    raise RuntimeError(
        "This code (saddle_data) requires Python 3.9 or newer.\n"
        "You are running Python {}.\n"
        "Please load a newer Python module (e.g., module load python/3.9+)."
        .format(sys.version.split()[0])
    )

import giopath  # noqa: F401  -- puts Giovannozzi's modules on sys.path

from saddle_data import load_omaha_slow, AmplitudeSelector
import pyuda


# defaults, formerly module-level globals read from inside the function
NFFT_DEFAULT = 512
NTOR_DEFAULT = [1, 2, 3, 4]
TIME_INTERVAL_DEFAULT = np.array([0.0, 2.0])
FMIN_DEFAULT = 1.0e2
FMAX_DEFAULT = 50.0e3
VMIN_DEFAULT = 0.0
LM_FREQ_HZ = 2000.0      # PPCF 2025 convention: below this, call it locked


def saddle_analysis_one_shot(shot, NFFT, ntor, SaddleDict,
                             time_interval=None, freq_min=None, freq_max=None,
                             v_min=VMIN_DEFAULT):
    """Populate SaddleDict[n] for one shot.  Original behaviour.

    The first four arguments are positional and unchanged:
        saddle_analysis_one_shot(shot, NFFT, ntor, SaddleDict)

    Note that `LMlabels` is a per-slice frequency test only, with no amplitude
    and no coherence condition, and that the search band starts at freq_min --
    a fully locked mode (f -> 0) falls below the band rather than being
    labelled.  It is not an event label.  See saddle_extras.py.
    """
    if time_interval is None:
        time_interval = TIME_INTERVAL_DEFAULT
    nt = len(time_interval)
    if freq_min is None:
        freq_min = np.array([FMIN_DEFAULT] * nt)
    if freq_max is None:
        freq_max = np.array([FMAX_DEFAULT] * nt)

    sd = load_omaha_slow(shot)
    sp = sd.spectrum(NFFT)
    rm = sp.n_detection(ntor)

    asd = {
        n: AmplitudeSelector(
            n,
            time_interval,
            freq_min,
            freq_max,
            v_min
        )
        for n in ntor
    }
    modes = rm.all_amplitudes(asd)
    for itor in ntor:
        freq = modes[itor].frequency

        # per PPCF 2025, label as an LM event anything with freq < 2kHz
        LM_labels = (freq < LM_FREQ_HZ).astype(int)

        SaddleDict[itor] = {
           'amp': modes[itor].amplitude,
           'damp_dt': modes[itor].damplitude_dt,
           'freq': modes[itor].frequency,
           'time': modes[itor].time,
           'LMlabels': LM_labels
        }

    return SaddleDict


# ---------------------------------------------------------------------------
def run_batch(shots, out_pik, min_ip_ma=0.05, NFFT=NFFT_DEFAULT,
              ntor=None, **kw):
    """Original batch sweep.  Pickle layout unchanged: {shot: {n: {...}}}."""
    ntor = NTOR_DEFAULT if ntor is None else ntor
    client = pyuda.Client()

    SaddleDict = {}
    good_shots = []
    dropped = {}

    for shot in shots:
        SaddleFeats = {}
        try:
            print(shot)
            IP = 1e-6 * client.get(
                "EPM/OUTPUT/GLOBALPARAMETERS/PLASMACURRENT", shot).data

            # Discard any shot with IP < 50kA
            if np.nanmax(IP) < min_ip_ma:
                dropped[shot] = f"max Ip {np.nanmax(IP):.4f} MA < {min_ip_ma}"
                continue

            saddle_analysis_one_shot(shot, NFFT, ntor, SaddleFeats, **kw)
            print(f"Max Ip for shot {shot} = ", np.nanmax(IP), " MA")
            good_shots.append(shot)
            SaddleDict[shot] = SaddleFeats
        except Exception as e:
            dropped[shot] = f"{type(e).__name__}: {e}"
            print(f"Skipping shot {shot}: {e}")

    with open(out_pik, "wb") as f:
        pickle.dump(SaddleDict, f)

    # drop reasons go beside the pickle, so the pickle format is untouched
    side = out_pik.rsplit(".", 1)[0] + ".dropped.json"
    with open(side, "w") as f:
        json.dump({"good_shots": good_shots,
                   "dropped": {str(k): v for k, v in dropped.items()}}, f, indent=2)

    print(f"\n{len(good_shots)} kept, {len(dropped)} dropped")
    print(f"  {out_pik}\n  {side}")
    return SaddleDict


def main():
    p = argparse.ArgumentParser(description="OMAHA saddle mode analysis (batch)")
    p.add_argument("--shot-min", type=int, default=47000)
    p.add_argument("--shot-max", type=int, default=47100)
    p.add_argument("--nfft", type=int, default=NFFT_DEFAULT)
    p.add_argument("--n-tor", type=int, nargs="+", default=NTOR_DEFAULT)
    p.add_argument("--f-min", type=float, default=FMIN_DEFAULT)
    p.add_argument("--f-max", type=float, default=FMAX_DEFAULT)
    p.add_argument("--min-ip", type=float, default=0.05)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    shots = range(a.shot_min, a.shot_max)
    out = a.out or f"mag_perturbations_shots{shots[0]}_{shots[-1]}.pkl"
    ti = TIME_INTERVAL_DEFAULT
    run_batch(shots, out, min_ip_ma=a.min_ip, NFFT=a.nfft, ntor=a.n_tor,
              time_interval=ti,
              freq_min=np.array([a.f_min] * len(ti)),
              freq_max=np.array([a.f_max] * len(ti)))


if __name__ == "__main__":
    main()
