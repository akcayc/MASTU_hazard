"""flattop.find_flattop on synthetic Ip traces."""
import _paths  # noqa: F401
import numpy as np
from flattop import find_flattop, FlatTopConfig


def main():
    t = np.linspace(0, 1.0, 4000)
    trap = np.clip(np.minimum(t / 0.1, (0.9 - t) / 0.1), 0, 1)
    ok = True

    for label, ip, expect_ok in [
        ("positive Ip, 0.8 MA", 0.8 * trap, True),
        ("negative Ip, 0.8 MA", -0.8 * trap, True),
        ("below min_abs_ip", 0.01 * trap, False),
        ("no plasma", np.zeros_like(t), False),
    ]:
        ft = find_flattop(t, ip, FlatTopConfig())
        good = (ft.ok == expect_ok)
        if ft.ok:
            # flat-top of a trapezoid must sit inside the flat part
            good &= (0.09 < ft.t_a < 0.15) and (0.75 < ft.t_b < 0.91)
            good &= abs(abs(ft.ip_mean) - 0.8) < 1e-6
            good &= ft.nrmse < 1e-6
        ok &= good
        detail = (f"window [{ft.t_a:.4f}, {ft.t_b:.4f}] <Ip>={ft.ip_mean:+.4f} MA"
                  if ft.ok else f"rejected: {ft.reason}")
        print(f"  {'PASS' if good else 'FAIL'}  {label:22s} {detail}")

    # polarity must not be decided by whichever ramp is steeper
    asym = np.clip(np.minimum(t / 0.30, (0.95 - t) / 0.02), 0, 1) * 0.8
    ft = find_flattop(t, asym, FlatTopConfig())
    good = ft.ok and ft.t_a < ft.t_b
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {'asymmetric ramps':22s} "
          + (f"window [{ft.t_a:.4f}, {ft.t_b:.4f}]" if ft.ok else f"rejected: {ft.reason}"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
