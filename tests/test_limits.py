"""tabular_limits: the DIII-D screening functions as ported."""
import _paths  # noqa: F401
import numpy as np

import tabular_limits as TL


def main():
    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")

    X = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, np.nan], [3.0, 4.0]])
    names = ["a", "b"]
    k = TL.apply_screening(X, names, {"a_min": 0.0, "a_max": 2.0})
    check("x == min fails, x == max passes", list(k) == [False, True, True, False])
    k = TL.apply_screening(X, names, {"b_max": 10.0})
    check("NaN in a screened column fails", list(k) == [True, True, False, True])
    check("empty set passes everything", TL.apply_screening(X, names, {}).all())

    L = {"a_min": 0.0, "a_max": 5.0}
    TL.adjust_limits(L, {"a_min": 1.0, "a_max": 9.0, "c_max": 1.0})
    check("adjust: only the more restrictive, absent keys ignored",
          L == {"a_min": 1.0, "a_max": 5.0})
    TL.adjust_limits(L, {"a_max": 9.0}, override=True)
    check("adjust override", L["a_max"] == 9.0)

    rng = np.random.default_rng(0)
    Z = rng.normal(size=(20000, 2))
    q = TL.quantile_limits(Z, ["u", "v"])
    frac = 1 - TL.apply_screening(Z, ["u", "v"], q).mean()
    check(f"quantile limits remove ~0.1% per column ({100*frac:.2f}% for 2)",
          0.0015 < frac < 0.0025)

    s = TL.parse_set_limits(["nhat::5", "q95:-20:-1"])
    check("parse_set_limits", s == {"nhat_max": 5.0, "q95_min": -20.0, "q95_max": -1.0})
    check("unknown keys reported", TL.unknown_keys({"a_min": 0, "zz_max": 1}, names)
          == ["zz_max"])
    check("MAST-U fixed set has no q limits (signed q)",
          not any(k_.startswith(("q0", "q95", "q33", "q67", "dq67"))
                  for k_ in TL.MASTU_FIXED_LIMITS))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
