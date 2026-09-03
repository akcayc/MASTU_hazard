"""Check find_flattop against real MAST-U Ip traces, in isolation.

The flat-top constants in FlatTopConfig were scaled from DIII-D by pulse
length, not measured.  If they are wrong, every shot is silently rejected and
the ladder table comes out empty with no obvious cause.  Run this before
anything else that needs a window.

    python check_flattop.py --shots 47000 47001 47002 47003 47004
    python check_flattop.py --shot-min 47000 --shot-max 47020 --min-tb 0.05

Tune from the command line rather than editing flattop.py, then put whatever
works into FlatTopConfig.
"""

import argparse
import numpy as np

import giopath  # noqa: F401

try:
    import pyuda
    from flattop import find_flattop, FlatTopConfig
except ImportError:
    giopath.report()
    raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shots", type=int, nargs="+")
    p.add_argument("--shot-min", type=int)
    p.add_argument("--shot-max", type=int)
    # every FlatTopConfig knob, overridable
    p.add_argument("--smooth-time", type=float, default=0.030)
    p.add_argument("--ramp-frac", type=float, default=0.20)
    p.add_argument("--min-ta", type=float, default=0.020)
    p.add_argument("--min-tb", type=float, default=0.100)
    p.add_argument("--min-ip", type=float, default=0.050)
    a = p.parse_args()

    shots = a.shots or list(range(a.shot_min, a.shot_max))
    cfg = FlatTopConfig(smooth_time=a.smooth_time, ramp_frac=a.ramp_frac,
                        min_ta=a.min_ta, min_tb=a.min_tb, min_abs_ip=a.min_ip)
    print(f"config: smooth={cfg.smooth_time}s ramp_frac={cfg.ramp_frac} "
          f"min_ta={cfg.min_ta}s min_tb={cfg.min_tb}s min|Ip|={cfg.min_abs_ip}MA\n")

    client = pyuda.Client()
    print(f"{'shot':>7} {'ok':>3} {'t_a':>8} {'t_b':>8} {'len':>7} "
          f"{'<Ip>MA':>8} {'nrmse':>7}  {'Ip span':>15}  reason")
    ok_n, lens, reasons = 0, [], {}
    for shot in shots:
        try:
            v = client.get("/AMC/PLASMA_CURRENT", shot)
            t, ip = np.asarray(v.time.data), 1e-3 * np.asarray(v.data)
        except Exception as e:
            print(f"{shot:>7}   -  {'':8} {'':8} {'':7} {'':8} {'':7}"
                  f"  {'':15}  no Ip: {e}")
            reasons["no Ip"] = reasons.get("no Ip", 0) + 1
            continue

        ft = find_flattop(t, ip, cfg)
        span = f"[{t[0]:.2f},{t[-1]:.2f}]"
        if ft.ok:
            ok_n += 1
            lens.append(ft.t_b - ft.t_a)
            print(f"{shot:>7} {'Y':>3} {ft.t_a:8.4f} {ft.t_b:8.4f} "
                  f"{ft.t_b-ft.t_a:7.4f} {ft.ip_mean:8.4f} {ft.nrmse:7.4f}"
                  f"  {span:>15}  ok")
        else:
            key = ft.reason.split("=")[0].split("<")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
            print(f"{shot:>7} {'n':>3} {'':8} {'':8} {'':7} "
                  f"{'':8} {'':7}  {span:>15}  {ft.reason}")

    n = len(shots)
    print(f"\naccepted {ok_n}/{n} ({100*ok_n/max(n,1):.0f}%)")
    if lens:
        lens = np.array(lens)
        print(f"  flat-top length: min {lens.min():.3f}  median "
              f"{np.median(lens):.3f}  max {lens.max():.3f} s")
        print(f"  OMAHA slow covers ~1.0 s -- shots with t_b > 1.0 lose exposure")
    if reasons:
        print("  rejections:")
        for k, c in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {c:4d}  {k}")
        print("\n  A high rejection rate almost certainly means the constants,"
              "\n  not the shots.  Loosen --min-tb / --min-ip first.")


if __name__ == "__main__":
    raise SystemExit(main())
