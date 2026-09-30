"""Where should the coherence gate go?  Measure, don't guess.

Two numbers decide it, and both are properties of your array and your data:

  the NOISE FLOOR -- what coherence pure noise produces on these coil angles,
      after the argmax assignment and the power weighting.  A gate below this
      admits everything and does nothing.
  the OBSERVED DISTRIBUTION -- what the real shots actually reach.  A gate
      above the bulk of this rejects everything.

The gate belongs between them.

NOTE the gate acts PER FREQUENCY BIN, inside the amplitude sum, so that is the
distribution reported here -- not the per-slice power-weighted value.  What
matters most is the last column: the fraction of summed POWER that survives.
A gate that works removes many noise bins while keeping the power, because the
bins carrying a real mode are the coherent ones.

    python coherence_survey.py --shot-min 47000 --shot-max 47020
"""

import argparse
import numpy as np

import giopath  # noqa: F401

try:
    from saddle_data import load_omaha_slow
    from saddle_analysis import NTOR_DEFAULT, TIME_INTERVAL_DEFAULT, \
        FMIN_DEFAULT, FMAX_DEFAULT
except ImportError:
    giopath.report()
    raise

GATES = [0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8]


def simulate_floor(phi, ntor, nf=257, nt=400, trials=12, seed=0):
    """Coherence that pure noise yields on THIS geometry, through the same
    argmax + power-weighting the real pipeline applies."""
    rng = np.random.default_rng(seed)
    nc = phi.size
    ap = np.exp(1j * np.radians(phi)[None, :] * np.asarray(ntor)[:, None])
    out = []
    for _ in range(trials):
        S = (rng.normal(size=(nc, nf, nt)) + 1j * rng.normal(size=(nc, nf, nt))) / np.sqrt(2)
        apS = np.tensordot(ap, S, axes=(1, 0))
        power = np.mean(np.abs(S) ** 2, axis=0)
        coh = np.abs(apS) ** 2 / nc ** 2 / power
        nfound = np.asarray(ntor)[np.abs(apS).argmax(axis=0)]
        for i, n in enumerate(ntor):
            m = nfound == n
            num = (coh[i] * power * m).sum(axis=0)
            den = (power * m).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                r = (num / den)[den > 0]
            out.append(r[np.isfinite(r)])
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shots", type=int, nargs="+")
    p.add_argument("--shot-min", type=int)
    p.add_argument("--shot-max", type=int)
    p.add_argument("--nfft", type=int, default=512)
    p.add_argument("--n-tor", type=int, nargs="+", default=NTOR_DEFAULT)
    a = p.parse_args()
    shots = a.shots or list(range(a.shot_min, a.shot_max))

    pooled, pw, per_n, phi, nshot = [], [], {int(n): [] for n in a.n_tor}, None, 0
    for shot in shots:
        try:
            sd = load_omaha_slow(shot)
            if sd.time.size < 2:
                continue
            rm = sd.spectrum(a.nfft).n_detection(np.asarray(a.n_tor))
            if phi is None:
                phi = np.asarray(sd.phi, float)
            F = rm.freq
            idf = (F >= FMIN_DEFAULT) & (F <= FMAX_DEFAULT)
            idf[0] = idf[-1] = False
            for i, n in enumerate(a.n_tor):
                # the bins this n actually claimed, in band -- exactly the set
                # the amplitude sums over, and exactly what the gate filters
                m = (rm.ntor.astype(int) == int(n)) & idf[:, None]
                if not m.any():
                    continue
                c = rm.coherence[i][m]
                p_ = rm.power[m]
                good = np.isfinite(c) & np.isfinite(p_)
                if good.any():
                    pooled.append(c[good])
                    pw.append(np.column_stack([c[good], p_[good]]))
                    per_n[int(n)].append(c[good])
            nshot += 1
        except Exception as e:
            print(f"  skip {shot}: {type(e).__name__}: {e}")

    if not pooled:
        print("no coherence data")
        return 1
    obs = np.concatenate(pooled)
    print(f"\n{nshot} shots, {obs.size} in-band bins, "
          f"{phi.size} coils at phi = {np.round(phi,1)}")

    sim = simulate_floor(phi, a.n_tor)
    print(f"\nNOISE FLOOR on this geometry (simulated, pure noise)")
    print(f"  median {np.median(sim):.3f}   p95 {np.percentile(sim,95):.3f}"
          f"   p99 {np.percentile(sim,99):.3f}   max {sim.max():.3f}")
    print(f"  (1/N_c = {1/phi.size:.3f} is the unconditioned expectation; the"
          f" argmax raises it)")

    print(f"\nOBSERVED")
    print("  " + "  ".join(f"p{q}={np.percentile(obs,q):.3f}"
                           for q in [50, 90, 99, 99.9]) + f"   max={obs.max():.3f}")

    CP = np.concatenate(pw, axis=0)
    tot_p = CP[:, 1].sum()
    print(f"\n{'gate':>6} {'bins kept':>10} {'power kept':>11} {'noise thru':>11}"
          f"   verdict")
    for g in GATES:
        keep = float((obs >= g).mean())
        pkeep = float(CP[CP[:, 0] >= g, 1].sum() / tot_p) if tot_p > 0 else 0.0
        leak = float((sim >= g).mean())
        if keep == 0:
            v = "rejects everything"
        elif leak > 0.05:
            v = "below the noise floor -- gates nothing"
        elif pkeep < 0.02:
            v = "throws away the power too"
        else:
            v = "USABLE"
        print(f"  {g:4.2f} {100*keep:9.3f}% {100*pkeep:10.3f}% {100*leak:10.3f}%"
              f"   {v}")
    print("\n  bins kept  = fraction of in-band bins passing the gate")
    print("  power kept = fraction of SUMMED POWER passing it  <-- the one to watch")
    print("  noise thru = fraction of pure-noise bins passing it")
    print("\n  A working gate drops most BINS while keeping most POWER: the bins")
    print("  carrying a real mode are the coherent ones.  If power falls as fast")
    print("  as bins, there is no coherent population to separate.")

    print(f"\nper n (per-bin coherence)\n{'n':>4} {'median':>8} {'p99':>8} {'max':>8}")
    for n in sorted(per_n):
        if per_n[n]:
            c = np.concatenate(per_n[n])
            print(f"  {n:>3} {np.median(c):8.3f} {np.percentile(c,99):8.3f} {c.max():8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
