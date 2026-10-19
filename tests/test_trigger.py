"""run_master.schmitt_first_crossing: threshold + debounce + back-dating."""
import _paths  # noqa: F401
import numpy as np
from run_master import schmitt_first_crossing as sc


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


def test_freq_continuity():
    """The optional Fa-continuity mask: off by default, keeps chirps, cuts noise."""
    import numpy as np
    from run_master import schmitt_first_crossing as sc
    t = np.arange(0, 1.0, 0.00128)              # the real 1.28 ms stride
    amp = np.zeros_like(t); amp[300:] = 10.0    # amplitude crosses at t=0.384
    ok = True

    def chk(label, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")

    base = sc(t, amp, 5.0, 0.008)
    chk(f"off by default: fires at {base:.4f} s",
        np.isfinite(base) and abs(base - t[300]) < 2e-3)

    # a mode sliding 4 -> 1 kHz over 20 ms: 192 Hz per slice
    lock = np.full_like(t, 4000.0)
    k = np.flatnonzero((t >= t[300]) & (t <= t[300] + 0.020))
    lock[k] = np.linspace(4000.0, 1000.0, k.size)
    lock[k[-1]:] = 1000.0
    got = sc(t, amp, 5.0, 0.008, freq=lock, freq_step_max=1000.0)
    chk(f"locking transition survives a 1 kHz tolerance ({got:.4f} s)",
        np.isfinite(got) and abs(got - base) < 2e-3)

    # noise Fa: hops across the band every slice
    rng = np.random.default_rng(0)
    noisy = rng.uniform(1e3, 5e4, t.size)
    chk("scattering Fa is rejected",
        not np.isfinite(sc(t, amp, 5.0, 0.008, freq=noisy, freq_step_max=1000.0)))
    chk("...but the same trace fires with the mask off",
        np.isfinite(sc(t, amp, 5.0, 0.008)))

    # monotonicity in level must survive the extra mask
    steady = np.full_like(t, 4000.0)
    ramp = np.zeros_like(t); ramp[300:] = np.linspace(4.0, 40.0, t.size - 300)
    times = [sc(t, ramp, L, 0.008, freq=steady, freq_step_max=1000.0)
             for L in (5.0, 10.0, 20.0, 30.0)]
    fin = [x for x in times if np.isfinite(x)]
    chk(f"ladder stays non-decreasing in level {[round(x,3) for x in fin]}",
        all(b >= a - 1e-9 for a, b in zip(fin, fin[1:])))
    return 0 if ok else 1


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
    ok &= (test_freq_continuity() == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
