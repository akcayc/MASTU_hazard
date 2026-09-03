"""
Verify that amplitude_with_coherence() reproduces Giovannozzi's
RecognizedModes.amplitude() exactly, and only adds a coherence column.

If this passes, the reimplementation is faithful and the only new thing in the
pipeline is the gate. If it fails, the retrofit is wrong and not the original.

    python check_equivalence.py --shot 47000
"""
import argparse
import numpy as np

from saddle_data import load_omaha_slow, AmplitudeSelector
from saddle_extras import amplitude_with_coherence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int, default=47000)
    p.add_argument("--nfft", type=int, default=512)
    p.add_argument("--n-tor", type=int, nargs="+", default=[1, 2, 3, 4])
    a = p.parse_args()

    ntor = np.array(a.n_tor)
    ti = np.array([0.0, 2.0])
    sel_kw = dict(time=ti,
                  f_min=np.array([1e2] * len(ti)),
                  f_max=np.array([50e3] * len(ti)),
                  v_min=0.0)

    sd = load_omaha_slow(a.shot)
    print(f"shot {a.shot}: {sd.phi.size} coils, phi = {np.round(sd.phi, 1)}")
    print(f"  {sd.time.size} samples, Fs = {1/np.mean(np.diff(sd.time)):.1f} Hz")

    sp = sd.spectrum(a.nfft)
    rm = sp.n_detection(ntor)
    print(f"  coherence array shape = {rm.coherence.shape}  "
          f"(expect ({ntor.size}, {rm.freq.size}, {rm.time.size}))")

    ok = True
    for n in ntor:
        sel = AmplitudeSelector(ntor=n, **sel_kw)

        ref = rm.amplitude(sel)                                   # upstream
        t, freq, Bp, Ba, coh = amplitude_with_coherence(rm, sel)  # mine

        same_t = np.array_equal(t, ref.time)
        same_f = np.allclose(freq, ref.frequency, equal_nan=True)
        same_p = np.allclose(Bp, ref.damplitude_dt, equal_nan=True)
        same_a = np.allclose(Ba, ref.amplitude, equal_nan=True)
        good = same_t and same_f and same_p and same_a
        ok &= good

        c = coh[np.isfinite(coh)]
        print(f"  n={n}: time={same_t} freq={same_f} Bp={same_p} Ba={same_a}"
              f"  -> {'OK' if good else 'MISMATCH'}")
        if c.size:
            print(f"        coherence: min={c.min():.3f} med={np.median(c):.3f} "
                  f"max={c.max():.3f}  (pure noise ~ 1/Ncoils = {1/sd.phi.size:.3f})")
        else:
            print("        coherence: all NaN  <-- indexing is wrong")
            ok = False

    print("\nRESULT:", "faithful" if ok else "NOT faithful -- do not use the retrofit")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
