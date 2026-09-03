"""Compare every magnetics array Gio's code can feed to n_detection().

saddle_data offers five sources, all feeding the same spectrum -> n_detection
chain (see model_saddle.py's selector).  `saddle_analysis.py` currently uses
only `load_omaha_slow`, which on shot 47000 turned out to be 5 coils spanning
59.8 deg -- a cluster, not a toroidal array, on which the projections for
n=1..4 are 93% parallel and the mode number is not measurable.

This script reports, for each source: coil count, toroidal coverage, and how
well-conditioned the n-projections actually are.  Run it before choosing.

    python compare_arrays.py --shot 47000
"""

import argparse
import numpy as np

import giopath  # noqa: F401

try:
    from saddle_data import load as load_saddle, load_omaha_slow, load_omaha_fast
except ImportError:
    giopath.report()
    raise


def describe(name, sd, ntor):
    print(f"\n=== {name} ===")
    if sd is None or sd.phi.size == 0:
        print("  unavailable / empty")
        return
    phi = np.asarray(sd.phi, float)
    order = np.argsort(phi)
    span = phi.max() - phi.min()
    print(f"  channels : {sd.phi.size}")
    print(f"  names    : {list(np.asarray(sd.names)[order][:12])}"
          + (" ..." if sd.phi.size > 12 else ""))
    print(f"  phi      : {np.round(phi[order], 1)}")
    print(f"  span     : {span:.1f} deg of 360")
    if sd.phi.size > 1:
        gaps = np.diff(np.sort(phi))
        print(f"  spacings : min {gaps.min():.1f}  max {gaps.max():.1f} deg")
        print(f"  Nyquist  : |n| < {180.0/max(gaps.min(), 1e-9):.1f}")
    if sd.time.size > 1:
        fs = 1.0 / np.mean(np.diff(sd.time))
        print(f"  samples  : {sd.time.size}   Fs = {fs/1e3:.1f} kHz   "
              f"t = [{sd.time[0]:.4f}, {sd.time[-1]:.4f}] s")

    # how independent are the n-projections on THIS geometry?
    A = np.exp(1j * np.radians(phi)[None, :] * np.asarray(ntor)[:, None])
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    G = np.abs(A @ A.conj().T)
    off = G[~np.eye(len(ntor), dtype=bool)]
    print(f"  n-projection overlap: max off-diagonal = {off.max():.3f}"
          f"   (0 = independent, 1 = indistinguishable)")
    print("     " + "".join(f"  n={m:<3d}" for m in ntor))
    for i, n in enumerate(ntor):
        print(f"  n={n:<2d}" + "".join(f" {G[i, j]:.3f}" for j in range(len(ntor))))
    verdict = ("n is measurable" if off.max() < 0.5 else
               "MARGINAL -- adjacent n are hard to separate" if off.max() < 0.8 else
               "n is NOT measurable on this array")
    print(f"  verdict  : {verdict}")
    print(f"  noise floor 1/N_c = {1/sd.phi.size:.3f}  "
          f"(operative floor after argmax+weighting runs ~1.5x this)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int, default=47000)
    p.add_argument("--n-tor", type=int, nargs="+", default=[1, 2, 3, 4])
    a = p.parse_args()

    sources = [
        ("Saddle Out All  (saddle_data.load)", lambda s: load_saddle(s)),
        ("Saddle Out A    (load -> block_a)", lambda s: load_saddle(s).block_a()),
        ("Saddle Out B    (load -> block_b)", lambda s: load_saddle(s).block_b()),
        ("Omaha           (load_omaha_slow)", load_omaha_slow),
        ("Omaha fast      (load_omaha_fast)", load_omaha_fast),
    ]
    print(f"shot {a.shot}: comparing every source model_saddle.py can select")
    for name, fn in sources:
        try:
            describe(name, fn(a.shot), a.n_tor)
        except Exception as e:
            print(f"\n=== {name} ===\n  FAILED: {type(e).__name__}: {e}")

    print("\nPick the source with the largest span and the lowest off-diagonal")
    print("overlap, then set it in saddle_extras.analyze_shot.")


if __name__ == "__main__":
    raise SystemExit(main())
