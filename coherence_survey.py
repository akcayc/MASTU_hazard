"""Where should the coherence gate go?  Measure, don't guess.

Two numbers decide it, and both are properties of your array and your data:

  the NOISE FLOOR -- what coherence pure noise produces on these coil angles,
      after the argmax assignment and the power weighting.  A gate below this
      admits everything and does nothing.
  the OBSERVED DISTRIBUTION -- what the real shots actually reach.  A gate
      above the bulk of this rejects everything.

The gate belongs between them.  This script prints both, plus the fraction of
slices surviving each candidate gate, over as many shots as you give it.

    python coherence_survey.py --shot-min 47000 --shot-max 47020
"""

import argparse
import numpy as np

import giopath  # noqa: F401

try:
    from saddle_data import load_omaha_slow, AmplitudeSelector
    from saddle_extras import amplitude_with_coherence
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

    ti = TIME_INTERVAL_DEFAULT
    sel_kw = dict(time=ti, f_min=np.array([FMIN_DEFAULT] * len(ti)),
                  f_max=np.array([FMAX_DEFAULT] * len(ti)), v_min=0.0)

    pooled, per_n, phi, nshot = [], {int(n): [] for n in a.n_tor}, None, 0
    for shot in shots:
        try:
            sd = load_omaha_slow(shot)
            if sd.time.size < 2:
                continue
            rm = sd.spectrum(a.nfft).n_detection(np.asarray(a.n_tor))
            if phi is None:
                phi = np.asarray(sd.phi, float)
            for n in a.n_tor:
                _, _, _, _, coh = amplitude_with_coherence(
                    rm, AmplitudeSelector(ntor=int(n), **sel_kw))
                c = coh[np.isfinite(coh)]
                if c.size:
                    pooled.append(c)
                    per_n[int(n)].append(c)
            nshot += 1
        except Exception as e:
            print(f"  skip {shot}: {type(e).__name__}: {e}")

    if not pooled:
        print("no coherence data")
        return 1
    obs = np.concatenate(pooled)
    print(f"\n{nshot} shots, {obs.size} coherence samples, "
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

    print(f"\n{'gate':>6} {'survive':>8} {'noise thru':>11}   verdict")
    for g in GATES:
        keep = float((obs >= g).mean())
        leak = float((sim >= g).mean())
        if keep == 0:
            v = "rejects everything"
        elif leak > 0.05:
            v = "below the noise floor -- gates nothing"
        elif keep < 1e-4:
            v = "almost everything rejected"
        else:
            v = "USABLE"
        print(f"  {g:4.2f} {100*keep:7.3f}% {100*leak:10.3f}%   {v}")
    print("\n  survive   = fraction of real slices passing the gate")
    print("  noise thru = fraction of pure-noise slices passing it")
    print("  Pick the lowest gate whose noise leakage is negligible and which"
          "\n  still passes enough real slices to populate the ladder.")

    print(f"\n{'n':>4} {'median':>8} {'p99':>8} {'max':>8}")
    for n in sorted(per_n):
        if per_n[n]:
            c = np.concatenate(per_n[n])
            print(f"  {n:>3} {np.median(c):8.3f} {np.percentile(c,99):8.3f} {c.max():8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
