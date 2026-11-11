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

    # alarm split: 20 flat censored shots, one event shot that rises over its
    # last 4 slices (caught, lead 4 dt + dt to onset), one flat event shot (missed)
    dt = 5e-3
    n = 40
    pp, YY, SS, TT = [], [], [], []
    rng = np.random.default_rng(3)
    for s in range(20):
        pp.append(rng.uniform(1e-4, 2e-4, n)); YY.append(np.zeros(n))
        SS.append(np.full(n, s)); TT.append(np.arange(n) * dt)
    rise = np.full(n, 1.5e-4); rise[-4:] = 1e-2
    flat = np.full(n, 1.5e-4)
    for s, v in ((100, rise), (101, flat)):
        y = np.zeros(n); y[-1] = 1
        pp.append(v); YY.append(y); SS.append(np.full(n, s)); TT.append(np.arange(n) * dt)
    pp, YY, SS, TT = map(np.concatenate, (pp, YY, SS, TT))
    pk = HP.shot_peaks(pp, YY, SS, TT, dt)
    thr = HP.alarm_split(pk, 0.95)
    cls = {r["shot"]: r["class"] for r in pk}
    lead = {r["shot"]: r["lead"] for r in pk}
    check(f"alarm: rising event caught, flat event missed (thr {thr:.3g}/s)",
          cls[100] == "caught" and cls[101] == "missed")
    check(f"alarm: lead = onset - first alarm = 4 dt ({1e3 * lead[100]:.1f} ms)",
          abs(lead[100] - 4 * dt) < 1e-9)
    check("alarm: about 5% of censored shots alarm",
          sum(cls[s] == "false alarm" for s in range(20)) == 1)
    pe, pc, _ = HP.separation(pk)
    check("separation from shot_peaks", pe.size == 2 and pc.size == 20)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
