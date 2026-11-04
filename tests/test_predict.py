"""hazard_predict: calibration bins, AUC, and the event alignment.

Rows are drawn from known probabilities, so a calibrated predictor (p = truth)
must sit on the diagonal and a mis-scaled one (p = 3 x truth) must not.  The
AUC is checked against brute force, ties included.
"""
import _paths  # noqa: F401
import numpy as np

import hazard_predict as HP


def main():
    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")

    rng = np.random.default_rng(0)
    p = np.exp(rng.uniform(np.log(1e-4), np.log(0.2), 200_000))
    y = (rng.random(p.size) < p).astype(float)

    rows, tot = HP.calibration(p, y, 10)
    z = [(b["events"] - b["expected"]) / np.sqrt(b["expected"]) for b in rows]
    check(f"calibrated: every bin within 4 sigma (max |z| {max(map(abs, z)):.1f})",
          max(map(abs, z)) < 4)
    check(f"calibrated: HL {tot['hosmer_lemeshow']:.1f} below 30 (8 dof)",
          tot["hosmer_lemeshow"] < 30)
    check("equal-count bins", max(b["rows"] for b in rows)
          - min(b["rows"] for b in rows) <= 1)
    _, bad = HP.calibration(np.minimum(3 * p, 0.99), y, 10)
    check(f"3x over-predicting: HL {bad['hosmer_lemeshow']:.0f} far above",
          bad["hosmer_lemeshow"] > 300)

    pos, neg = rng.normal(1, 1, 300), rng.normal(0, 1, 400)
    pos[:20] = neg[:20]                                    # ties
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])
    check(f"AUC matches brute force ({HP.auc(pos, neg):.4f})",
          abs(HP.auc(pos, neg) - brute) < 1e-12)

    # two shots, dt = 5 ms: an event shot whose h rises to onset, a flat
    # censored one; the event median must be highest in the last bin
    dt = 5e-3
    T = np.r_[np.arange(40), np.arange(40)] * dt
    S = np.r_[np.zeros(40), np.ones(40)]
    Y = np.zeros(80); Y[39] = 1
    pp = np.r_[np.linspace(1e-3, 5e-2, 40), np.full(40, 1e-3)]
    al = HP.aligned(pp, Y, S, T, dt, 0.1, 2 * dt)
    e = al["event"]["median"]
    check("aligned: event hazard highest just before onset",
          np.nanargmax(e) == 0 and np.allclose(al["censored"]["median"], HP.rate(1e-3, dt)))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
