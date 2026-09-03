"""run_master.schmitt_first_crossing: threshold + debounce + back-dating."""
import _paths  # noqa: F401
import numpy as np
from run_master import schmitt_first_crossing as sc


def main():
    t = np.arange(0, 1.0, 0.001)
    ok = True

    def check(label, sig, level, expect, debounce=0.008):
        nonlocal ok
        got = sc(t, sig, level, debounce)
        good = (np.isnan(got) and np.isnan(expect)) or \
               (np.isfinite(got) and abs(got - expect) < 1.5e-3)
        ok &= good
        shown = "nan" if np.isnan(got) else f"{got:.4f}"
        print(f"  {'PASS' if good else 'FAIL'}  {label:34s} -> {shown}")

    step = np.zeros_like(t); step[400:] = 10.0
    check("step at 0.400 s", step, 5.0, 0.400)

    glitch = np.zeros_like(t); glitch[300:303] = 10.0; glitch[600:] = 10.0
    check("3 ms glitch, then step at 0.600", glitch, 5.0, 0.600)

    check("never crossed", np.zeros_like(t), 5.0, np.nan)

    holed = step.copy(); holed[450:460] = np.nan
    check("NaNs after the crossing", holed, 5.0, 0.400)

    # the reported time is the crossing, not the confirmation
    got = sc(t, step, 5.0, 0.050)
    good = np.isfinite(got) and abs(got - 0.400) < 1.5e-3
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {'back-dated over a 50 ms debounce':34s} "
          f"-> {got:.4f} (not 0.450)")

    # a run shorter than the debounce must not fire
    brief = np.zeros_like(t); brief[400:405] = 10.0
    check("5 ms run vs 8 ms debounce", brief, 5.0, np.nan)

    ok &= (test_n_detect_guard() == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


def test_n_detect_guard():
    """--n-detect outside --n-tor must fail loudly, not return NaN."""
    from run_master import build_rm_detector, detect_ladder
    ok = True
    try:
        build_rm_detector(shots=[47000], client=None, ntor_list=[1, 2], n_detect=3)
        print("  FAIL  build_rm_detector accepted n_detect outside ntor")
        ok = False
    except ValueError as e:
        print(f"  PASS  build_rm_detector rejects it: {str(e)[:58]}...")
    try:
        detect_ladder({"shot": 1, "traces": {1: None}}, [2.0], n_detect=9)
        print("  FAIL  detect_ladder returned instead of raising")
        ok = False
    except ValueError as e:
        print(f"  PASS  detect_ladder rejects it: {str(e)[:64]}...")
    return 0 if ok else 1
