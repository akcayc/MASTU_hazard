"""extract_features.midplane_width: the nhat chord, R_out - R_in at Z = 0."""
import _paths  # noqa: F401
import numpy as np

from extract_features import midplane_width


def ellipse(R, a, kappa, z0, n=181):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return R + a * np.cos(th), z0 + kappa * a * np.sin(th)


def main():
    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")

    a, k = 0.6, 2.0
    w = midplane_width(*ellipse(0.9, a, k, 0.0))
    check(f"centred: width 2a = {2*a:.3f} (got {w:.4f})", abs(w - 2 * a) < 1e-3)

    z0 = 0.3                         # shifted up: Z = 0 cuts below the widest point
    want = 2 * a * np.sqrt(1 - (z0 / (k * a)) ** 2)
    w = midplane_width(*ellipse(0.9, a, k, z0))
    check(f"shifted: width {want:.4f} (got {w:.4f}), narrower than 2a",
          abs(w - want) < 2e-3 and w < 2 * a)

    w = midplane_width(*ellipse(0.9, a, k, 1.5))   # boundary entirely above Z = 0
    check("boundary missing Z = 0 gives NaN", np.isnan(w))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
