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
This is why NTOR_DEFAULT carries both signs, matching Giovannozzi's own
instruction that n_detection takes "a list of toroidal mode number (positive
and negative)".  Which half carries the coherence tells you the sense of
rotation.
"""

import numpy as np

import giopath  # noqa: F401  -- puts Giovannozzi's modules on sys.path

from saddle_data import AmplitudeSelector, load_omaha_slow
from mode_functions import zeros_spectrum

from saddle_analysis import (
    NFFT_DEFAULT, NTOR_DEFAULT, TIME_INTERVAL_DEFAULT,
    FMIN_DEFAULT, FMAX_DEFAULT, VMIN_DEFAULT, LM_FREQ_HZ,
)


# Coherence gate, measured on 93 shots of 47000-47099 (10.1M in-band bins,
# 5 OMAHA coils).  coherence_survey.py gave:
#
#     gate   bins kept   power kept   noise through
#     0.20     80.5%        92.4%        100%
#     0.40     15.9%        53.5%         52%
#     0.60      2.8%        48.8%          0.005%
#     0.80      0.76%       19.3%          0%
#
# Power kept is flat across 0.40-0.70 (53.5 -> 45.3%) while bins kept falls
# ninefold and noise leakage collapses: half the in-band power sits in a small
# highly-coherent population, the rest is noise across millions of bins.  At
# 0.80 power kept drops to 19.3%, which is the gate cutting real signal, so the
# shelf ends near 0.70.  0.60 leaks ~500 noise bins in 10M and keeps 48.8%.
COHERENCE_GATE_SUGGESTED = 0.6


def _select_bins(rm, sel: AmplitudeSelector, coh_min, absolute_n):
    """The (time, freq) bins that count toward sel.ntor, and their coherence.

    Shared by both amplitude estimators so that they sum over exactly the
    same bins.  Returns (tsel, Fsel, power, id_mask, coh), id_mask True =
    excluded; or None for a record too short to analyse.
    """
    F, time = rm.freq, rm.time
    if time.size < 2:
        return None

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

    # per-bin coherence for this n, over the same selection
    searched = np.asarray(rm.n_tor_searched)
    if absolute_n:
        n_abs = abs(int(sel.ntor))
        coh = np.full(power_spectrum.shape, np.nan)
        for s in (n_abs, -n_abs):
            m = np.flatnonzero(searched == s)
            if m.size:
                c_s = rm.coherence[int(m[0])].T[:, idf][idt, :]
                coh = np.where(Ntor == s, c_s, coh)
    else:
        match = np.flatnonzero(searched == sel.ntor)
        coh = (rm.coherence[int(match[0])].T[:, idf][idt, :] if match.size
               else np.full(power_spectrum.shape, np.nan))

    # The gate belongs INSIDE the frequency sum, as in DIII-D's get_amplitude:
    # a bin that fails coherence must contribute zero to the amplitude, not
    # merely disqualify the time slice after the fact.  Gating slices instead
    # leaves the noise bins in the sum, so the amplitude carries an
    # irreducible floor (~1.8 mG on the OMAHA subset) that no gate setting can
    # lower -- which is what made a 0.2 gate fire on every low rung and a 0.8
    # gate fire on nothing.
    id_mask = (np.abs(Ntor) != abs(int(sel.ntor))) if absolute_n \
        else (Ntor != sel.ntor)
    if coh_min > 0:
        id_mask = id_mask | ~(np.isfinite(coh) & (coh >= coh_min))

    return tsel, Fsel, power_spectrum, id_mask, coh


def amplitude_with_coherence(rm, sel: AmplitudeSelector, coh_min=0.0,
                             absolute_n=False):
    """RecognizedModes.amplitude() plus the power-weighted coherence.

    Mirrors the upstream method exactly for amplitude and frequency, then
    reduces rm.coherence over the same (time, freq) selection and n-mask:

        coh(t) = sum_f coherence(f,t) P(f,t) / sum_f P(f,t)

    absolute_n=True selects on |n| instead of signed n, as DIII-D does
    (modespyec `abolute-n`, i.e. get_amplitude(nsigned=False)): a bin counts
    toward |sel.ntor| if the argmax assigned it +|n| or -|n|, so a mode counts
    whichever way it rotates.  The coherence gate reads the row of the sign
    the bin was actually assigned.  With False (default) the result is
    bit-identical to the upstream method, which check_equivalence.py asserts.

    Returns (time, freq, Bp, Ba, coh), coh in [0, 1], NaN where no bin
    survived the mask.
    """
    sel_ = _select_bins(rm, sel, coh_min, absolute_n)
    if sel_ is None:
        z = np.zeros(0)
        return z, z, z, z, z
    tsel, Fsel, power_spectrum, id_mask, coh = sel_

    power_masked = np.ma.array(power_spectrum, mask=id_mask)
    power_sum = power_masked.sum(axis=1)

    Bp = np.sqrt(power_sum).filled(fill_value=np.nan)
    Fa = ((power_masked * Fsel).sum(axis=1) / power_sum).filled(fill_value=np.nan)
    Fa[Bp < sel.v_min] = np.nan
    Ba = Bp / Fa / 2 / np.pi

    # slice-level coherence, power-weighted over the surviving bins -- reported
    # for diagnostics, no longer the thing the gate acts on
    coh_masked = np.ma.array(coh, mask=id_mask)
    coh_w = ((coh_masked * power_masked).sum(axis=1) / power_sum).filled(np.nan)

    return tsel, Fa, Bp, Ba, coh_w


def hann_enbw(nfft):
    """Equivalent noise bandwidth of the Hann window, in bins: N sum(w^2)/sum(w)^2.

    1.5 for large N.  Converts the spectrum that matplotlib's specgram returns
    to an RMS amplitude, see amplitude_diiid.
    """
    w = np.hanning(nfft)
    return float(nfft * np.sum(w * w) / np.sum(w) ** 2)


def amplitude_diiid(rm, sel: AmplitudeSelector, coh_min=0.0, absolute_n=True):
    """RMS field amplitude [input units * s], in the DIII-D form.

    DIII-D (modespyec.get_amplitude, integrated=True) converts dB/dt to B one
    bin at a time:

        A = Ts * sqrt( ff * sum_f  W(f) / omegahat_f^2 * 0.5 (X11 + X22) )

    with omegahat = omega * Ts and ff = 2 / (sum(w^2) * nfft), which makes
    ff * sum|X|^2 an RMS.  Giovannozzi's Ba divides the band total by one
    power-weighted centroid frequency instead, which agrees only for a
    narrowband mode.

    Here X = FFT(w x), but matplotlib's specgram(mode='complex') returns
    S = X / sum(w).  Substituting,

        A^2 = (2 / ENBW) * sum_f  W(f) P(f) / omega_f^2 ,   P = mean_c |S_c|^2

    ENBW = N sum(w^2) / sum(w)^2 being 1.5 for Hann.  A pure sinusoid of
    peak amplitude a then gives a / sqrt(2) whatever the leakage, which
    tests/test_coherence.py checks.  W is the same gated, |n|-selected bin set
    that amplitude_with_coherence sums over -- both call _select_bins.

    Assumes specgram's defaults (Hann window, no detrend, complex scaled by
    sum(w)); true of matplotlib >= 1.x.  If the input is dB/dt in T/s this
    is B in tesla; if it is anything else the number is only relative.

    Returns (time, B_rms), NaN where no bin survived.
    """
    sel_ = _select_bins(rm, sel, coh_min, absolute_n)
    if sel_ is None:
        z = np.zeros(0)
        return z, z
    tsel, Fsel, power, id_mask, _ = sel_
    nfft = 2 * (rm.freq.size - 1)          # one-sided spectrum: N/2 + 1 bins
    omega = 2.0 * np.pi * Fsel             # Fsel > 0: bin 0 is excluded upstream
    weighted = np.ma.array(power / omega ** 2, mask=id_mask).sum(axis=1)
    B2 = (2.0 / hann_enbw(nfft)) * weighted
    return tsel, np.sqrt(B2).filled(np.nan)


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
                 t_pre=0.020, out=None, coh_min=0.0, absolute_n=True):
    """Per-shot analysis with coherence and noise floor.

    absolute_n=True (default, as DIII-D): per_n is keyed by |n|, and each entry
    pools the +|n| and -|n| bins.  Both signs are always searched, whatever
    `ntor` holds, since a bin must be able to be assigned to either.
    absolute_n=False keeps the signed keys of the original routine.

    Returns (per_n, meta):
        per_n[n] = {amp, damp_dt, freq, time, coherence, noise_floor, LMlabels}
        meta     = {shot, NFFT, ntor, n_coils, phi, fs_hz, freq_band, v_min}

    On naming, inherited from saddle_data.ModeAmplitude:
        damp_dt = sqrt(sum of masked power)  -- dB/dt spectral amplitude, T/s
        amp     = damp_dt / (2*pi*freq)      -- time-integrated, T

    UNITS.  Giovannozzi labels these T/s and T, but adds: "I'm saying T and
    T/s but, really, I'm not sure that the data I'm reading are in absolute
    units, so use it for comparing shots."  So treat them as a consistent
    relative scale -- valid for ranking and for shot-to-shot comparison, not
    for quoting an absolute field.  That is enough for a threshold ladder,
    which only ever needs shots to be comparable with each other; it is not
    enough to quote a threshold in Gauss or to compare against DIII-D.

    It also means the per-shot noise-floor normalisation in run_master is a
    choice, not a necessity: if the instrument gain is stable between shots,
    dividing by each shot's own floor removes real cross-shot variation.  Both
    numbers are stored -- 'amp' raw and 'noise_floor' -- so the ladder can be
    run either way.
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

    # n=0 is ALWAYS searched, and never reported.  It is a sink for the
    # common mode: a signal every coil sees identically (baseline wander, slow
    # drift) has no toroidal structure, but on a clustered array it does not
    # cancel against exp(-i n phi).  On the 5 OMAHA coils, spanning 60 deg,
    # a pure common-mode signal scores
    #     |sum exp(-i phi_c)|^2 / N^2 = 0.866
    # on n=1 -- above any gate worth setting.  Measured consequence before
    # this fix: at coh_min=0.8 the surviving pre-plasma bins were bin 1 alone
    # (Fa = 390 Hz = the FFT bin width), i.e. drift masquerading as n=1.
    # With 0 in the list the argmax sends those bins to n=0 (coherence 1.000
    # beats 0.866) and the |n|>=1 masks exclude them.  DIII-D is protected
    # instead by its integerness gate, which common mode fails at nhat ~ 0.
    if absolute_n:
        mags = sorted({abs(int(n)) for n in ntor if int(n) != 0})
        search = [0] + [s * m for m in mags for s in (-1, 1)]
        ntor = mags
    else:
        search = sorted({0} | {int(n) for n in ntor})

    sp = sd.spectrum(NFFT)
    rm = sp.n_detection(np.asarray(search))

    for n in ntor:
        n = int(n)
        sel = AmplitudeSelector(n, time_interval, freq_min, freq_max, v_min)
        t, freq, Bp, Ba, coh = amplitude_with_coherence(rm, sel, coh_min,
                                                        absolute_n=absolute_n)
        _, Bd = amplitude_diiid(rm, sel, coh_min, absolute_n=absolute_n)
        per_n[n] = {
            "amp_perbin": Bd,     # DIII-D form: per-bin 1/omega^2
            "amp_centroid": Ba,   # Giovannozzi: band total / centroid
            "amp": Ba,            # the default estimator (centroid)
            "noise_floor_perbin": noise_floor(t, Bd, t_pre),
            "noise_floor_centroid": noise_floor(t, Ba, t_pre),
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
        "ntor_searched": [int(n) for n in search],
        "absolute_n": bool(absolute_n),
        "n_coils": int(sd.phi.size),
        "phi": np.asarray(sd.phi),
        "fs_hz": float(1.0 / np.mean(np.diff(sd.time))) if sd.time.size > 1 else np.nan,
        "freq_band": (float(freq_min.min()), float(freq_max.max())),
        "v_min": v_min,
        "coh_min": coh_min,
    }
    return per_n, meta
