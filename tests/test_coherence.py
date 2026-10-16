"""saddle_extras.amplitude_with_coherence, against Giovannozzi's real classes.

Runs the genuine SaddleFull / Spectra / RecognizedModes pipeline on synthetic
data with a KNOWN injected mode, using stubs only for the Freya-only modules.

Checks three things:
  1. rm.coherence really has shape (n_searched, nfreq, ntime);
  2. amplitude and frequency are bit-identical to RecognizedModes.amplitude();
  3. the coherence rises to ~1 for the injected n and stays at the noise floor
     for every other n -- but ONLY for the toroidal handedness that
     Spectra.n_detection projects onto.  See the handedness note below.
"""
import _paths  # noqa: F401
import numpy as np
from saddle_data import SaddleFull, AmplitudeSelector
from saddle_extras import amplitude_with_coherence, amplitude_diiid, hann_enbw

NC, FS, T = 8, 200_000.0, 0.30
F_MODE, N_MODE, T_ON = 10_000.0, 2, 0.15
NTOR = np.array([1, 2, 3, 4])
#: a full evenly-spaced torus -- a clean testbed, NOT the OMAHA geometry
PHI = np.linspace(0, 360, NC, endpoint=False)


def build(sign, seed=0, mode_amp=20.0):
    """Synthetic array data: white noise, plus an n=2 mode after T_ON."""
    rng = np.random.default_rng(seed)
    phi = PHI
    t = np.arange(0, T, 1 / FS)
    data = rng.normal(0, 1.0, (NC, t.size))
    on = t > T_ON
    for c in range(NC):
        data[c, on] += mode_amp * np.sin(
            2 * np.pi * F_MODE * t[on] + sign * np.radians(phi[c]) * N_MODE)
    sd = SaddleFull(1, np.array([f"c{i}" for i in range(NC)]), t, data, phi,
                    np.full(NC, "OMAHA"))
    return sd


def main():
    ok = True
    ti = np.array([0.0, T])
    sel_kw = dict(time=ti, f_min=np.array([1e2] * 2),
                  f_max=np.array([5e4] * 2), v_min=0.0)

    sd = build(sign=-1)
    sp = sd.spectrum(512)
    rm = sp.n_detection(NTOR)

    # --- 1. shape assumption behind the coherence indexing ------------------
    want = (NTOR.size, rm.freq.size, rm.time.size)
    good = rm.coherence.shape == want
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  coherence shape {rm.coherence.shape} "
          f"== (n_searched, nfreq, ntime) {want}")

    # --- 2. amplitude path identical to upstream ---------------------------
    for n in NTOR:
        sel = AmplitudeSelector(ntor=int(n), **sel_kw)
        ref = rm.amplitude(sel)
        tt, freq, Bp, Ba, coh = amplitude_with_coherence(rm, sel)
        good = (np.array_equal(tt, ref.time)
                and np.allclose(Bp, ref.damplitude_dt, equal_nan=True)
                and np.allclose(Ba, ref.amplitude, equal_nan=True)
                and np.allclose(freq, ref.frequency, equal_nan=True))
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  n={n}: time/Bp/Ba/freq identical "
              f"to RecognizedModes.amplitude()")

    # --- 3. the gate separates the injected mode ---------------------------
    print(f"\n  pure-noise coherence floor 1/N_c = {1/NC:.4f} "
          f"(measured floor runs nearer 2/N_c)")
    for label, sign, expect_detect in [
            ("sin(wt - n*phi)  matches convention", -1, True),
            ("sin(wt + n*phi)  opposite handedness", +1, False)]:
        rm_s = build(sign=sign).spectrum(512).n_detection(NTOR)
        print(f"\n  {label}")
        for n in NTOR:
            sel = AmplitudeSelector(ntor=int(n), **sel_kw)
            tt, freq, Bp, Ba, coh = amplitude_with_coherence(rm_s, sel)
            pre, post = tt < T_ON, tt > T_ON
            c_noise, c_mode = np.nanmedian(coh[pre]), np.nanmedian(coh[post])
            ratio = np.nanmedian(Ba[post]) / np.nanmedian(Ba[pre])
            tag = "  <-- injected" if n == N_MODE else ""
            print(f"    n={n}  coh(noise)={c_noise:.4f}  coh(mode)={c_mode:.4f}"
                  f"  amp(mode)/amp(noise)={ratio:7.1f}{tag}")
            if n == N_MODE:
                good = (c_mode > 0.9) if expect_detect else (c_mode < 0.1)
                ok &= good
                print(f"    {'PASS' if good else 'FAIL'}  injected n detected"
                      f" = {expect_detect}")

    # --- 4. the gate belongs inside the frequency sum ------------------------
    # A bin failing coherence must contribute zero to the amplitude.  Gating
    # whole time slices instead leaves the noise bins in the sum, so the
    # amplitude keeps an irreducible floor no gate setting can lower.
    print("\n  gate inside the sum: floor must fall faster than the signal")
    rm_n = build(sign=-1, mode_amp=0.0).spectrum(512).n_detection(NTOR)   # no mode
    rm_m = build(sign=-1).spectrum(512).n_detection(NTOR)              # mode
    sel = AmplitudeSelector(ntor=N_MODE, **sel_kw)
    prev = None
    for g in (0.0, 0.4, 0.8):
        _, _, Bp_n, _, _ = amplitude_with_coherence(rm_n, sel, g)
        _, _, Bp_m, _, _ = amplitude_with_coherence(rm_m, sel, g)
        fn = np.nanmedian(Bp_n[np.isfinite(Bp_n)])
        fm = np.nanmedian(Bp_m[np.isfinite(Bp_m)])
        print(f"    coh_min={g:.1f}  floor={fn:8.4f}  signal={fm:9.4f}  SNR={fm/fn:7.1f}")
        if prev is not None:
            # NaN floor = every bin gated out, i.e. no coherent power at all.
            # That is the floor collapsing completely, not a failure.
            fell = (not np.isfinite(fn)) or fn < prev[0]
            held = fm > 0.95 * prev[1]
            ok &= (fell and held)
            note = " (floor fully suppressed)" if not np.isfinite(fn) else ""
            print(f"    {'PASS' if fell and held else 'FAIL'}  floor fell, "
                  f"signal held{note}")
        prev = (fn if np.isfinite(fn) else 0.0, fm)

    # --- 5. |n|: either rotation sense must be detected ----------------------
    # DIII-D selects on |n| (get_amplitude nsigned=False).  With both signs
    # searched and absolute_n=True, the injected mode must be found whichever
    # way it rotates, and the signed selection must still miss the wrong one.
    print("\n  |n| selection, both signs searched")
    both = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    sel2 = AmplitudeSelector(ntor=N_MODE, **sel_kw)
    for label, sign in [("sin(wt - n*phi)", -1), ("sin(wt + n*phi)", +1)]:
        rm_b = build(sign=sign).spectrum(512).n_detection(both)
        tt, _, _, Ba, coh = amplitude_with_coherence(rm_b, sel2, 0.6,
                                                     absolute_n=True)
        pre, post = tt < T_ON, tt > T_ON
        c_mode = np.nanmedian(coh[post])
        a_mode = np.nanmedian(Ba[post])
        a_pre = np.nanmedian(Ba[pre]) if np.isfinite(Ba[pre]).any() else 0.0
        good = (c_mode > 0.9) and (a_mode > 5 * a_pre)
        ok &= good
        print(f"    {'PASS' if good else 'FAIL'}  {label}: coh={c_mode:.3f}, "
              f"|n|=2 amp {a_mode:.3g} vs pre-mode {a_pre:.3g}")
    # signed selection of +2 only sees the matching handedness
    amps = {}
    for sign in (-1, +1):
        rm_b = build(sign=sign).spectrum(512).n_detection(both)
        tt, _, _, Ba, _ = amplitude_with_coherence(rm_b, sel2, 0.6)
        amps[sign] = np.nanmedian(Ba[tt > T_ON]) if np.isfinite(Ba[tt > T_ON]).any() else 0.0
    good = amps[-1] > 5 * amps[+1]
    ok &= good
    print(f"    {'PASS' if good else 'FAIL'}  signed +2 sees only one handedness "
          f"({amps[-1]:.3g} vs {amps[+1]:.3g})")

    # --- 6. DIII-D-form amplitude has the right ABSOLUTE scale ---------------
    # Treat the synthetic data as dB/dt.  A per-coil sinusoid of peak a at f
    # has RMS dB/dt a/sqrt(2), hence RMS B = a / (sqrt(2) * 2 pi f).  This is
    # the check on the 2/ENBW and Parseval bookkeeping, and it uses an
    # off-bin frequency (25.6 bins) so that window leakage is exercised.
    print("\n  DIII-D-form amplitude, absolute scale")
    expect = 20.0 / (np.sqrt(2) * 2 * np.pi * F_MODE)
    for sign in (-1, +1):
        rm_b = build(sign=sign).spectrum(512).n_detection(both)
        tt, Bd = amplitude_diiid(rm_b, sel2, 0.6, absolute_n=True)
        got = np.nanmedian(Bd[tt > T_ON + 0.02])
        rel = got / expect - 1.0
        good = abs(rel) < 0.05
        ok &= good
        print(f"    {'PASS' if good else 'FAIL'}  sign {sign:+d}: B_rms={got:.4g}"
              f" expected {expect:.4g} ({100*rel:+.1f}%)")
    # Giovannozzi's Ba differs by a fixed, computable factor for a line
    rm_b = build(sign=-1).spectrum(512).n_detection(both)
    tt, _, _, Ba_g, _ = amplitude_with_coherence(rm_b, sel2, 0.6, absolute_n=True)
    ratio = np.nanmedian(Ba_g[tt > T_ON + 0.02]) / expect
    print(f"    info  Gio Ba / true B_rms = {ratio:.3f} for this line "
          f"(sqrt(ENBW/2) = {np.sqrt(hann_enbw(512)/2):.3f})")

    # --- 5. n=0 absorbs the common mode --------------------------------------
    # A signal every coil sees identically has no toroidal structure, but on a
    # clustered array it does not cancel against exp(-i n phi): the 5 OMAHA
    # coils give it coherence 0.866 on n=1.  With 0 in the search list the
    # argmax sends those bins to n=0 instead.
    print("\n  n=0 sink: pure common mode must not read as n=1")
    rng2 = np.random.default_rng(7)
    OM = np.array([36.8, 26.5, 4.0, -5.0, -23.0])
    tt = np.arange(0, T, 1 / FS)
    cm = rng2.normal(0, 1.0, (OM.size, tt.size))
    for c in range(OM.size):
        cm[c] += 30.0 * np.sin(2 * np.pi * 400.0 * tt)       # identical in every coil
    sd_cm = SaddleFull(1, np.array([f"c{i}" for i in range(OM.size)]), tt, cm,
                       OM, np.full(OM.size, "X"))
    sel_cm = AmplitudeSelector(ntor=1, time=np.array([0.0, T]),
                               f_min=np.array([1e2] * 2),
                               f_max=np.array([5e4] * 2), v_min=0.0)
    res = {}
    for tag, NT in (("without", np.array([-4, -3, -2, -1, 1, 2, 3, 4])),
                    ("with   ", np.array([0, -4, -3, -2, -1, 1, 2, 3, 4]))):
        rm_cm = sd_cm.spectrum(512).n_detection(NT)
        _, _, Bp_cm, _, _ = amplitude_with_coherence(rm_cm, sel_cm, 0.8, True)
        f = np.isfinite(Bp_cm)
        res[tag] = np.nanmedian(Bp_cm[f]) if f.any() else 0.0
        print(f"    {tag} n=0: n=1 amplitude at gate 0.8 = {res[tag]:.4g}"
              f"   ({100*f.mean():.0f}% of slices survive)")
    good = res["with   "] < res["without"] / 10.0
    ok &= good
    print(f"    {'PASS' if good else 'FAIL'}  n=0 suppresses the common mode "
          f"({res['without']/max(res['with   '],1e-12):.0f}x)")

    print("\n  NOTE  the opposite-handedness block is the important one: the")
    print("        coherence correctly collapses, yet the UNGATED amplitude")
    print("        still rises ~10-17x for every n at once.  That is the false")
    print("        positive the gate exists to remove.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
