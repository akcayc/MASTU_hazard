"""--harmonics: an n=1 overtone against a separate n=2, on the OMAHA geometry.

Synthetic five-coil data runs through the real analyze_shot (load_omaha_slow
patched) and run_master.detect_ladder.  Two shots share one chirping n=1:

    overtone   n=2 at exactly twice the n=1 phase, amplitude tied to n=1
    separate   n=2 rotating 8% faster than 2 f1, its own phase and growth

Checks: the ladder gains hf/ha/hp for |n|=2 only, and the three tests come
out on the right side of the 1/sqrt(N_eff) noise level for each case.
"""
import _paths  # noqa: F401
import numpy as np

import saddle_extras
from saddle_data import SaddleFull
import run_master

FS, T, TC, W = 200_000.0, 0.30, 0.20, 0.020
PHI = np.array([36.8, 26.5, 4.0, -5.0, -23.0])     # OMAHA slow coils [deg]


def build(kind, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, 1 / FS)
    f1 = 5000.0 - 4000.0 * t                         # slow chirp, 5 -> 3.8 kHz
    th1 = 2 * np.pi * np.cumsum(f1) / FS
    a1 = 20.0 * (1 + 0.5 * np.sin(2 * np.pi * 15 * t))   # breathing envelope
    if kind == "overtone":
        th2, a2 = 2 * th1 + 0.7, 0.25 * a1
    else:
        th2 = 2 * np.pi * np.cumsum(2.08 * f1) / FS + rng.uniform(0, 2 * np.pi)
        th2 += np.cumsum(rng.normal(0, 0.02, t.size))     # wandering phase
        a2 = 5.0 * (1 + 0.5 * np.sin(2 * np.pi * 9 * t + 2.0))
    data = rng.normal(0, 1.0, (PHI.size, t.size))
    for c, p in enumerate(np.radians(PHI)):
        data[c] += a1 * np.sin(th1 - 1 * p) + a2 * np.sin(th2 - 2 * p)
    return SaddleFull(1, np.array([f"c{i}" for i in range(PHI.size)]), t, data,
                      PHI, np.full(PHI.size, "OMAHA"))


def traces(kind):
    sd = build(kind)
    saddle_extras.load_omaha_slow = lambda shot: sd
    per_n, _ = saddle_extras.analyze_shot(
        1, ntor=[1, 2, 3], time_interval=np.array([0.0, T]),
        freq_min=np.array([1e3, 1e3]), freq_max=np.array([5e4, 5e4]),
        coh_min=0.8, harmonics=True)
    return {"shot": 1, "traces": {n: {k: per_n[n][k] for k in
            ("time", "amp", "coherence", "freq", "pk_bin", "z_pk", "z_h")
            if k in per_n[n]} for n in per_n}}


def main():
    ok = True
    res = {}
    for kind in ("overtone", "separate"):
        sig = traces(kind)
        tr1, tr2 = sig["traces"][1], sig["traces"][2]
        res[kind] = run_master.harmonic_stats(tr2["time"], tr2, tr1, 2, TC, W)

        # the ladder carries the columns for |n|=2, and not for |n|=1
        lv = [float(np.nanmedian(tr2["amp"])) * 0.5]
        row2 = run_master.detect_ladder(sig, lv, 2, harmonic_window=W)
        row1 = run_master.detect_ladder(sig, lv, 1, harmonic_window=W)
        good = (all(f"{p}{lv[0]:.5f}" in row2 for p in ("hf", "ha", "hp"))
                and not any(k.startswith("h") for k in row1))
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {kind}: hf/ha/hp on |n|=2 only")

    n_eff = 2 * W / 1.28e-3 / 1.32
    noise = 1 / np.sqrt(n_eff)
    for kind, (hf, ha, hp) in res.items():
        print(f"    {kind:9s} hf {hf:.3f}  ha {ha:+.2f}  hp {hp:.2f}")
    hf, ha, hp = res["overtone"]
    good = hp > 0.9 and ha > 0.8 and hf < 0.1
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  overtone: hp > 0.9, ha > 0.8, hf < 0.1")
    # Only hp is asserted for the separate mode.  Its 8% offset is ~340 Hz at
    # 8.4 kHz, under one 390 Hz bin, so the peak bins coincide and hf reads 0;
    # and two smooth envelopes over 40 ms correlate whatever their source, so
    # ha reads high.  Both are printed, as a reminder that hp is the test.
    hf, ha, hp = res["separate"]
    good = hp < 2 * noise
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  separate: hp < {2 * noise:.2f} "
          f"(2/sqrt(N_eff))")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
