"""
Additions to the OMAHA saddle-coil analysis, kept separate from
saddle_analysis.py so that the original routine stays untouched and testable.

Nothing here modifies upstream behaviour.  Three things are provided:

  amplitude_with_coherence()
      A copy of saddle_data.RecognizedModes.amplitude() that additionally
      returns the toroidal-fit coherence for each amplitude sample.  The
      amplitude and frequency it returns must be identical to the upstream
      method -- check_equivalence.py asserts exactly that.

  noise_floor()
      Per-shot pre-plasma baseline.  The OMAHA record opens ~1 ms, before
      breakdown, so every shot carries its own no-plasma reference.

  analyze_shot()
      Enriched per-shot entry point: same output keys as
      saddle_analysis_one_shot plus 'coherence' and 'noise_floor'.  It returns
      (per_n, meta) as a 2-tuple rather than mixing a string key into the
      integer-keyed dict, so `for n in per_n:` stays safe.

WHY THE COHERENCE MATTERS.  saddle_data.Spectra.n_detection computes

    coherence = |sum_c exp(-i n phi_c) S_c|^2 / N_c^2 / <|S_c|^2>

which is 1 for a perfect n-mode and ~1/N_c for uncorrelated noise, and
RecognizedModes.threshold() exists to gate on it -- but amplitude() never
calls either, and the driver passes v_min=0.0 so the one remaining rejection
line is inert.  The mode number is then assigned by an argmax, which always
returns something.  Every noise bin is therefore admitted to the amplitude
sum, which is why the shipped amplitude tracks the noise floor: measured over
78 shots, the n=1 amplitude before breakdown (1.92e-07) is within 31% of its
flat-top value (2.52e-07).

VALIDATED OFFLINE against Gio's real classes on synthetic data (8 coils,
200 kHz, an n=2 mode at 10 kHz switched on at t=0.15 s):
  * rm.coherence has shape (n_searched, nfreq, ntime) -- the indexing below is
    correct;
  * amplitude and frequency are bit-identical to RecognizedModes.amplitude();
  * for the injected mode the coherence reaches 0.9994 against a measured
    noise floor of ~0.25, while every other n stays at the floor.

HANDEDNESS -- READ THIS BEFORE CHOOSING ntor.  Spectra.n_detection builds

    ap = np.exp(1j * np.radians(phi) * ntor[:, None])

i.e. it projects onto exp(+i n phi), so it detects modes whose spectral phase
runs as exp(-i n phi).  A mode of the opposite handedness is INVISIBLE to a
positive-only search list: in the test above, injecting sin(wt + n phi) instead
of sin(wt - n phi) drove the coherence for every n BELOW the noise floor
(0.005-0.010) -- while the ungated amplitude still rose by a factor of 10-17
for every n indiscriminately.  That is precisely the false-positive mode the
gate exists to remove.

Which handedness a real mode has depends on the toroidal rotation direction,
and therefore on the sign of Ip and Bt.  DIII-D handles this explicitly: its
locked-mode detector branches on sign(<Bt><Ip>) to pick MmatrixRH or
MmatrixLH.  The MAST-U path has no such branch and ntor=[1,2,3,4] is one-sided.
Until the convention has been checked against a shot with a known mode, search
both signs -- see NTOR_BOTH_SIGNS -- and confirm which half carries the
coherence.
"""

import numpy as np

from saddle_data import AmplitudeSelector, load_omaha_slow
from mode_functions import zeros_spectrum

from saddle_analysis import (
    NFFT_DEFAULT, NTOR_DEFAULT, TIME_INTERVAL_DEFAULT,
    FMIN_DEFAULT, FMAX_DEFAULT, VMIN_DEFAULT, LM_FREQ_HZ,
)


# Search list covering both toroidal handednesses.  See the module docstring.
NTOR_BOTH_SIGNS = [-4, -3, -2, -1, 1, 2, 3, 4]

# Coherence gate.  The theoretical pure-noise floor is 1/N_c, but the measured
# floor in the offline test sat near 2/N_c, so set the gate from the measured
# distribution rather than from theory.  With a floor of ~0.25 and a genuine
# mode at ~1.0, anything in 0.5-0.7 separates cleanly.
COHERENCE_GATE_SUGGESTED = 0.6


def amplitude_with_coherence(rm, sel: AmplitudeSelector):
    """RecognizedModes.amplitude() plus the power-weighted coherence.

    Mirrors the upstream method exactly for amplitude and frequency, then
    reduces rm.coherence over the same (time, freq) selection and n-mask:

        coh(t) = sum_f coherence(f,t) P(f,t) / sum_f P(f,t)

    Returns (time, freq, Bp, Ba, coh), coh in [0, 1], NaN where no bin
    survived the mask.
    """
    F, time = rm.freq, rm.time
    if time.size < 2:
        z = np.zeros(0)
        return z, z, z, z, z

    f_min, f_max = min(sel.f_min), max(sel.f_max)
    t_min, t_max = min(sel.time), max(sel.time)

    idf = (F >= f_min) & (F <= f_max)
    idf[0] = False
    idf[-1] = False
    idt = (time >= t_min) & (time <= t_max)

    power_spectrum = rm.power.T[:, idf][idt, :]
    Fsel = F[idf]
    Ntor = rm.ntor.astype(int).T[:, idf][idt, :]
    tsel = time[idt]
    power_spectrum = zeros_spectrum(sel.time, sel.f_min, sel.f_max,
                                    tsel, Fsel, power_spectrum)

    id_mask = Ntor != sel.ntor
    power_masked = np.ma.array(power_spectrum, mask=id_mask)
    power_sum = power_masked.sum(axis=1)

    Bp = np.sqrt(power_sum).filled(fill_value=np.nan)
    Fa = ((power_masked * Fsel).sum(axis=1) / power_sum).filled(fill_value=np.nan)
    Fa[Bp < sel.v_min] = np.nan
    Ba = Bp / Fa / 2 / np.pi

    # coherence for this n, same selection, power-weighted across frequency
    match = np.flatnonzero(np.asarray(rm.n_tor_searched) == sel.ntor)
    if match.size == 0:
        coh_w = np.full(tsel.shape, np.nan)
    else:
        coh = rm.coherence[int(match[0])].T[:, idf][idt, :]
        coh_masked = np.ma.array(coh, mask=id_mask)
        coh_w = ((coh_masked * power_masked).sum(axis=1) / power_sum).filled(np.nan)

    return tsel, Fa, Bp, Ba, coh_w


def noise_floor(time, values, t_pre=0.020):
    """Median of `values` before `t_pre` -- the pre-breakdown baseline.

    Dividing an amplitude by this converts an absolute ladder (which the
    current pipeline cannot support, the per-shot floor varying by ~10x across
    shots) into a signal-to-floor ladder comparable across shots.
    """
    pre = np.asarray(time) < t_pre
    if pre.sum() < 5:
        return np.nan
    v = np.asarray(values)[pre]
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def analyze_shot(shot, NFFT=NFFT_DEFAULT, ntor=None, time_interval=None,
                 freq_min=None, freq_max=None, v_min=VMIN_DEFAULT,
                 t_pre=0.020, out=None):
    """Per-shot analysis with coherence and noise floor.

    Returns (per_n, meta):
        per_n[n] = {amp, damp_dt, freq, time, coherence, noise_floor, LMlabels}
        meta     = {shot, NFFT, ntor, n_coils, phi, fs_hz, freq_band, v_min}

    On naming, inherited from saddle_data.ModeAmplitude:
        damp_dt = sqrt(sum of masked power)  -- raw dB/dt spectral amplitude
        amp     = damp_dt / (2*pi*freq)      -- time-integrated, i.e. B
    Neither is in physical units: matplotlib.mlab.specgram(mode='complex')
    applies no window-power or sample-rate normalisation, and no coil area or
    amplifier gain has been divided out.
    """
    per_n = {} if out is None else out
    ntor = list(NTOR_DEFAULT if ntor is None else np.atleast_1d(ntor))
    time_interval = (TIME_INTERVAL_DEFAULT if time_interval is None
                     else np.asarray(time_interval))
    nt = len(time_interval)
    freq_min = np.array([FMIN_DEFAULT] * nt) if freq_min is None else np.asarray(freq_min)
    freq_max = np.array([FMAX_DEFAULT] * nt) if freq_max is None else np.asarray(freq_max)

    sd = load_omaha_slow(shot)
    if sd.time.size < 2:
        raise RuntimeError(f"no OMAHA data for shot {shot}")

    sp = sd.spectrum(NFFT)
    rm = sp.n_detection(np.asarray(ntor))

    for n in ntor:
        n = int(n)
        sel = AmplitudeSelector(n, time_interval, freq_min, freq_max, v_min)
        t, freq, Bp, Ba, coh = amplitude_with_coherence(rm, sel)
        per_n[n] = {
            "amp": Ba,
            "damp_dt": Bp,
            "freq": freq,
            "time": t,
            "coherence": coh,
            "noise_floor": noise_floor(t, Ba, t_pre),
            "LMlabels": (freq < LM_FREQ_HZ).astype(int),
        }

    meta = {
        "shot": shot,
        "NFFT": NFFT,
        "ntor": [int(n) for n in ntor],
        "n_coils": int(sd.phi.size),
        "phi": np.asarray(sd.phi),
        "fs_hz": float(1.0 / np.mean(np.diff(sd.time))) if sd.time.size > 1 else np.nan,
        "freq_band": (float(freq_min.min()), float(freq_max.max())),
        "v_min": v_min,
    }
    return per_n, meta
