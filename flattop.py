"""
MAST-U flat-top windowing -- the analogue of DIII-D `iptimestamp.py`.

The hazard framework needs, per shot, an analysis window [t_a, t_b] over which
the plasma is a well-defined object.  Everything downstream depends on it:
the detector only triggers inside it, the dwell time in the survival table is
(t_event - t_a), and a shot that fails the window test is dropped rather than
contributing a spurious censored record.

The DIII-D version is hard-wired to that machine (500 ms smoothing, t_a > 0.4 s,
t_b > 1.0 s, |<Ip>| > 0.5 MA).  Those magnitudes silently reject essentially
every MAST-U pulse, so the algorithm is reproduced here with the geometry
factored out into `FlatTopConfig`.

NOT YET VALIDATED -- the defaults below are scaled from DIII-D by pulse-length
ratio and must be checked against real MAST-U Ip traces before use.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class FlatTopConfig:
    """Machine-specific constants.  MAST-U defaults; see module docstring."""
    smooth_time: float = 0.030      # boxcar width [s]   (DIII-D: 0.500)
    ramp_frac: float = 0.20         # back-off fraction of peak dIp/dt (DIII-D: 0.20)
    min_ta: float = 0.020           # earliest allowed t_a [s]   (DIII-D: 0.40)
    min_tb: float = 0.100           # earliest allowed t_b [s]   (DIII-D: 1.00)
    min_abs_ip: float = 0.050       # minimum |<Ip>| over window [MA] (DIII-D: 0.50)
    disruption_t2ratio: float = 1.05
    disruption_blend: float = 0.55  # t_b -> blend*t20 + (1-blend)*t21


@dataclass
class FlatTop:
    ok: bool
    t_a: float
    t_b: float
    ip_mean: float
    slope: float          # linear-fit slope over the window [MA/s]
    nrmse: float          # normalised RMSE of that fit -- flat-top quality
    reason: str = ""


def _boxcar(y, t, width):
    if width <= 0:
        return y
    dt = np.median(np.diff(t))
    n = max(1, int(round(width / dt)))
    k = np.ones(n) / n
    return np.convolve(y, k, mode="same")


def find_flattop(time, ip, cfg: FlatTopConfig = None) -> FlatTop:
    """Reproduce the DIII-D four-timestamp construction on a MAST-U Ip trace.

    `ip` in MA.  Returns the flat-top window plus its quality metrics.
    """
    cfg = cfg or FlatTopConfig()
    bad = lambda why: FlatTop(False, np.nan, np.nan, np.nan, np.nan, np.nan, why)

    time = np.asarray(time, float)
    ip = np.asarray(ip, float)
    good = np.isfinite(time) & np.isfinite(ip)
    time, ip = time[good], ip[good]
    if time.size < 10:
        return bad("too few samples")

    ips = _boxcar(ip, time, cfg.smooth_time)
    dip = np.gradient(ips, time, edge_order=2)

    # polarity comes from the current itself, not from the steeper ramp
    sgn = np.sign(ips[int(np.argmax(np.abs(ips)))])
    if sgn == 0:
        return bad("Ip identically zero")
    dip_s = sgn * dip                       # ramp-up now positive regardless of Ip sign
    i_up = int(np.argmax(dip_s))
    i_dn = int(np.argmin(dip_s))
    if i_dn <= i_up:
        return bad("ramp-down precedes ramp-up")

    up_pk, dn_pk = dip_s[i_up], dip_s[i_dn]

    # back-track from each extremum to where |dIp/dt| falls below ramp_frac * peak
    def _walk(i0, direction, peak):
        thr = cfg.ramp_frac * abs(peak)
        i = i0
        while 0 < i < dip_s.size - 1 and abs(dip_s[i]) > thr:
            i += direction
        return i

    t10 = time[_walk(i_up, -1, up_pk)]       # ramp-up start
    t11 = time[_walk(i_up, +1, up_pk)]       # ramp-up end   -> t_a
    t20 = time[_walk(i_dn, -1, dn_pk)]       # ramp-down start -> t_b
    t21 = time[_walk(i_dn, +1, dn_pk)]       # ramp-down end

    t_a, t_b = t11, t20

    # DIII-D disruption heuristic: an abrupt termination compresses the
    # ramp-down, so push t_b later rather than discard the pre-disruption data.
    t2ratio = (t21 - t10) / (t20 - t10) if (t20 - t10) > 0 else np.inf
    if t2ratio < cfg.disruption_t2ratio:
        b = cfg.disruption_blend
        t_b = b * t20 + (1.0 - b) * t21

    if not (t_a > cfg.min_ta):
        return bad(f"t_a={t_a:.4f} <= min_ta")
    if not (t_b > t_a):
        return bad("t_b <= t_a")
    if not (t_b > cfg.min_tb):
        return bad(f"t_b={t_b:.4f} <= min_tb")

    win = (time >= t_a) & (time <= t_b)
    if win.sum() < 5:
        return bad("window too short")

    ip_mean = float(np.mean(ip[win]))
    if abs(ip_mean) < cfg.min_abs_ip:
        return bad(f"|<Ip>|={abs(ip_mean):.4f} < min_abs_ip")

    # flat-top quality: linear fit residual, normalised by the mean
    p = np.polyfit(time[win], ip[win], 1)
    resid = ip[win] - np.polyval(p, time[win])
    nrmse = float(np.sqrt(np.mean(resid ** 2)) / abs(ip_mean))

    return FlatTop(True, float(t_a), float(t_b), ip_mean, float(p[0]), nrmse, "ok")
