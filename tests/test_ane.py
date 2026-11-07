"""check_ane_thomson: the path-length fit tells a line integral from a line
average on a synthetic shot whose plasma width varies in time."""
import _paths  # noqa: F401
import numpy as np

import check_ane_thomson as CA


def shot(kind, seed=0):
    rng = np.random.default_rng(seed)
    t_ts = np.arange(0.05, 0.80, 0.0167)               # ~60 Hz Thomson
    t_eq = np.arange(0.025, 1.02, 5.077e-3)
    L = 0.6 + 0.6 * np.clip(t_eq / 0.3, 0, 1)          # width grows over the ramp
    rc = 0.85 + 0.0 * t_eq
    rin, rout = rc - L / 2, rc + L / 2
    R = np.linspace(0.25, 1.50, 120)
    ne = np.zeros((t_ts.size, R.size))
    I_true = np.zeros(t_ts.size)
    for k, tk in enumerate(t_ts):
        j = np.argmin(np.abs(t_eq - tk))
        x = (R - rc[j]) / (L[j] / 2)
        n0 = 3e19 * (0.5 + tk)
        ne[k] = np.where(np.abs(x) < 1, n0 * (1 - x ** 2), np.nan)
        I_true[k] = n0 * (4 / 3) * (L[j] / 2)
    ne *= 1 + 0.03 * rng.standard_normal(ne.shape)
    t_ane = np.arange(0.0, 1.0, 1e-4)
    It = np.interp(t_ane, t_ts, I_true)
    Lt = np.interp(t_ane, t_eq, L)
    ane = 2.0 * It if kind == "integral" else It / Lt
    ane *= 1 + 0.01 * rng.standard_normal(ane.size)
    return t_ts, R, ne, t_eq, rin, rout, t_ane, ane


def main():
    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")

    r, z = 0.85 + 0.5 * np.cos(np.linspace(0, 2 * np.pi, 200)), \
        2 * 0.5 * np.sin(np.linspace(0, 2 * np.pi, 200))
    rin, rout = CA.midplane_crossings(r, z)
    check(f"midplane crossings of an ellipse ({rin:.3f}, {rout:.3f})",
          abs(rin - 0.35) < 1e-3 and abs(rout - 1.35) < 1e-3)

    for kind, want in (("integral", 1.0), ("average", 0.0)):
        c = CA.cross_check(*shot(kind))
        u = c["use"]
        b, se, cc = CA.fit_beta(c["L"][u], c["ane"][u], c["A_ts"][u])
        v = CA.verdict(b, se)
        check(f"{kind}: beta = {b:.3f} +/- {se:.3f}, c = {cc:.2f}, '{v[:14]}'",
              abs(b - want) < 0.1 and v.startswith("line " + kind.upper()))
        if kind == "integral":
            check("integral: double-pass factor recovered as c ~ 2", abs(cc - 2) < 0.1)

    c = CA.cross_check(*shot("integral"), t0=0.4, t1=0.8)   # flattop only: L fixed
    u = c["use"]
    b, se, _ = CA.fit_beta(c["L"][u], c["ane"][u], c["A_ts"][u])
    check(f"constant L gives no verdict ({CA.verdict(b, se)[:12]})",
          CA.verdict(b, se).startswith("undetermined"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
