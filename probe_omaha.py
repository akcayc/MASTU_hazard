"""What are the OMAHA signals, physically?  Print what the archive says.

The amplitude pipeline treats the OMAHA slow channels as dB/dt in T/s.  Nothing
in saddle_data.load_omaha applies a coil area or an amplifier gain -- only the
sign `pol` -- so either the archive already holds calibrated dB/dt, or the
scale is missing.  This prints the evidence for the first and leaves the
decision to you; paste the output to whoever knows the hardware.

    python probe_omaha.py --shot 47000
    python probe_omaha.py --shot 47000 --all-coils
"""

import argparse
import numpy as np

import giopath  # noqa: F401

try:
    import pyuda
    import omaha_coils as oc
except ImportError:
    giopath.report()
    raise


def describe(v, label):
    """Everything the UDA object volunteers about a signal."""
    data = np.asarray(v.data, float)
    t = np.asarray(v.time.data, float)
    print(f"  {label}")
    for attr in ("units", "label", "description", "name", "source", "type"):
        val = getattr(v, attr, None)
        if val not in (None, ""):
            print(f"      {attr:12s} {val}")
    tu = getattr(v.time, "units", None)
    print(f"      shape        {data.shape}   time units {tu}")
    if t.size > 1:
        dt = np.median(np.diff(t))
        print(f"      time         {t[0]:.5f} .. {t[-1]:.5f}   "
              f"dt {dt*1e6:.3f} us  (Fs {1/dt/1e3:.1f} kHz)")
    pre = t < 0.0
    if pre.sum() > 10:
        print(f"      pre-trigger  mean {data[pre].mean():+.4g}  "
              f"std {data[pre].std():.4g}  (t<0, {pre.sum()} samples)")
    print(f"      whole record mean {np.nanmean(data):+.4g}  "
          f"std {np.nanstd(data):.4g}  |max| {np.nanmax(np.abs(data)):.4g}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shot", type=int, default=47000)
    p.add_argument("--all-coils", action="store_true",
                   help="every coil, not just the first radial one")
    a = p.parse_args()

    # --- 1. the coil table: is there a calibration column? ----------------
    coils = oc.coils
    print("omaha_coils.coils columns:", list(coils.columns))
    print(f"  {len(coils)} rows; orient values: "
          f"{sorted(set(coils.orient)) if 'orient' in coils else 'n/a'}")
    cal_like = [c for c in coils.columns
                if any(k in c.lower() for k in
                       ("area", "na", "gain", "calib", "cal", "turn", "scale", "sens"))]
    print(f"  calibration-looking columns: {cal_like or 'NONE'}")
    others = [x for x in dir(oc) if not x.startswith("_") and x != "coils"]
    print(f"  other names in omaha_coils: {others}")
    print("\nradial (used) coils:")
    print(coils[coils.orient == "R"].to_string())

    # --- 2. what the archive says about the channels -----------------------
    client = pyuda.Client()
    radial = coils[coils.orient == "R"]
    rows = radial if a.all_coils else radial.iloc[:1]
    print(f"\nshot {a.shot}")
    for line in rows.itertuples():
        for kind in ("name_slow", "name_fast"):
            name = getattr(line, kind)
            try:
                v = client.get(name, a.shot)
            except Exception as e:
                print(f"  {name}: {type(e).__name__}: {e}")
                continue
            describe(v, f"{kind}: {name}  (phi {line.phi}, pol {line.pol})")

    print("\nHow to read this:")
    print("  units 'T/s' or similar, or a description saying calibrated -> the")
    print("  archive is already dB/dt and 1/omega gives tesla.")
    print("  units 'V' / 'counts' / empty, and no calibration column above ->")
    print("  the coil area and gain are missing, and the amplitude is only")
    print("  relative.  Units 'T' would mean already integrated: dividing by")
    print("  omega would then be wrong.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
