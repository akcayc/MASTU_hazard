"""compare_lf_hf.transfer / verdict, on signals whose relation is known.

The fast channel is a dense comb of tones at 10 MHz; the slow channel is built
from the same tones at 200 kHz with a chosen transfer function.  The verdict
must recover it -- in particular it must tell "same quantity" from "time
integral" from "derivative", which is the whole point of the script.
"""
import _paths  # noqa: F401
import numpy as np
from compare_lf_hf import transfer, verdict

FS_F, FS_S = 10.0e6, 200.0e3
T0, T1 = 0.0, 0.12
N_TONES = 220


_FAST = {}


def build(kind, seed=0, lag=0.0):
    rng = np.random.default_rng(seed)
    f_k = rng.uniform(1.5e3, 45e3, N_TONES)
    a_k = rng.uniform(0.5, 1.5, f_k.size)
    p_k = rng.uniform(0, 2 * np.pi, f_k.size)
    tf = np.arange(T0, T1, 1 / FS_F)
    # slow stamps carry a sub-sample offset, as two digitisers' clocks would
    ts = np.arange(T0, T1, 1 / FS_S) + 0.37 / FS_S
    tsl = ts - lag          # slow(t) = fast(t - lag): slow is `lag` behind

    def tones(t, gain, shift):
        out = np.zeros(t.size)
        for f, a, p, g, s in zip(f_k, a_k, p_k, gain, shift):
            out += g * a * np.cos(2 * np.pi * f * t + p + s)
        return out

    one = np.ones_like(f_k)
    zero = np.zeros_like(f_k)
    # the fast signal is the same for every kind: synthesise it once
    if seed not in _FAST:
        _FAST[seed] = tones(tf, one, zero)
    xf = _FAST[seed]
    if kind == "same":
        ys = tones(tsl, 2.0 * one, zero)                      # gain 2
    elif kind == "integral":
        ys = tones(tsl, 1 / (2 * np.pi * f_k) * 1e4, zero - np.pi / 2)
    elif kind == "derivative":
        ys = tones(tsl, 2 * np.pi * f_k * 1e-4, zero + np.pi / 2)
    elif kind == "flipped":
        ys = tones(tsl, -1.0 * one, zero)
    rng2 = np.random.default_rng(seed + 1)
    return tf, xf + 0.05 * rng2.normal(size=tf.size), ts, ys + 0.05 * rng2.normal(size=ts.size)


def main():
    ok = True
    for kind, want in [("same", "same quantity (flat gain)"),
                       ("integral", "slow = time integral of fast"),
                       ("derivative", "slow = derivative of fast"),
                       ("flipped", "same quantity, opposite sign")]:
        tf, xf, ts, ys = build(kind)
        res = transfer(tf, xf, ts, ys, T0 + 0.02, T1 - 0.02)
        v = verdict(res)
        good = v["kind"] == want
        ok &= good
        extra = (f"slope {v['slope']:+.2f} phase {v['phase_deg']:+.0f} "
                 f"gain {v['gain']:.3g}" if "slope" in v else v.get("note", ""))
        print(f"  {'PASS' if good else 'FAIL'}  {kind:10s} -> {v['kind']}  ({extra})")

    # gain of the 'same' case should come back as the 2.0 we put in
    tf, xf, ts, ys = build("same")
    v = verdict(transfer(tf, xf, ts, ys, T0 + 0.02, T1 - 0.02))
    good = abs(v["gain"] / 2.0 - 1) < 0.05
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  flat-band gain recovered "
          f"({v['gain']:.3f} vs 2.000)")

    # a clock offset between the digitisers: without the delay search this
    # reads "no common signal", which is exactly the failure to guard against
    for lag in (3.1e-3, -8.0e-3):
        tf, xf, ts, ys = build("same", lag=lag)
        res = transfer(tf, xf, ts, ys, T0 + 0.02, T1 - 0.02)
        v = verdict(res)
        good = (v["kind"] == "same quantity (flat gain)"
                and abs(res["delay"] - lag) < 10e-6
                and abs(v["gain"] / 2.0 - 1) < 0.05)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  slow {1e3*lag:+.1f} ms behind -> "
              f"found {1e3*res['delay']:+.3f} ms, {v['kind']}")

    # no common signal: independent noise must NOT be reported as a relation
    rng = np.random.default_rng(5)
    tf = np.arange(T0, T1, 1 / FS_F)
    ts = np.arange(T0, T1, 1 / FS_S)
    v = verdict(transfer(tf, rng.normal(size=tf.size), ts,
                         rng.normal(size=ts.size), T0 + 0.02, T1 - 0.02))
    good = v["kind"] == "no common signal"
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  independent noise -> {v['kind']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
