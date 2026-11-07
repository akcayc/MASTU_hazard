"""Is ANE/DENSITY the line integral of n_e, or already the line average?

    python check_ane_thomson.py --shot 48123
    python check_ane_thomson.py --shot 48123 --t0 0.2 --t1 0.8 --out ane_48123.png

Nothing in this repository establishes which it is; the "line-integrated,
m^-2" comments in epm_nodes.py and extract_features.py are assumptions.  This
script tests both against Thomson scattering on one shot.

1. Metadata.  Units, label and description of ANE/DENSITY, its time bases
   (v.time and ANE/TIME1), and the Thomson signals, as pyuda reports them.

2. Thomson integrals.  At each Thomson time, n_e(R) from the radial
   Thomson chords is integrated in R between the inboard and outboard
   separatrix crossings at Z = 0 (EPM boundary, nearest EPM slice):

       I_TS = int n_e dR          [m^-2]
       A_TS = I_TS / L,  L = R_out - R_in     [m^-3]

   and ANE is averaged over the same Thomson interval.

3. The test.  Fit   ANE / A_TS = c * L^beta   over the slices:

       beta ~ 1   ANE scales with path length: a line INTEGRAL, and c is the
                  calibration (c ~ 2 for a double-pass reading reported raw)
       beta ~ 0   ANE does not: already a line AVERAGE, c ~ 1

   beta needs L to vary, so include the ramps; the script prints the L range
   and beta's standard error.  The medians of ANE/I_TS and ANE/A_TS are
   printed too, but on their own they differ only by the factor L ~ 1.2 m,
   so the calibration and the chord geometry can mimic either answer.

Assumptions, all printed: the interferometer chord is a horizontal midplane
line through the same plasma as the Thomson chords (check its real
geometry); n_e outside the separatrix is ignored; slices where Thomson
covers less than --min-coverage of [R_in, R_out] are dropped.
"""

import argparse

import numpy as np

EPM_TIME, RBDRY, ZBDRY = "TIME", "OUTPUT/SEPARATRIXGEOMETRY/RBOUNDARY", \
    "OUTPUT/SEPARATRIXGEOMETRY/ZBOUNDARY"


# ---------------------------------------------------------------- geometry
def midplane_crossings(r, z, z0=0.0):
    """(R_in, R_out) where the closed boundary crosses Z = z0; NaNs if it
    crosses fewer than twice.  Same construction as
    extract_features.midplane_width."""
    ok = np.isfinite(r) & np.isfinite(z) & (r > 0)
    r, z = r[ok], z[ok]
    if r.size < 3:
        return np.nan, np.nan
    r, z = np.append(r, r[0]), np.append(z, z[0])
    dz = z - z0
    i = np.flatnonzero((dz[:-1] * dz[1:] <= 0) & (dz[:-1] != dz[1:]))
    if i.size < 2:
        return np.nan, np.nan
    w = dz[i] / (dz[i] - dz[i + 1])
    rc = r[i] + w * (r[i + 1] - r[i])
    return float(rc.min()), float(rc.max())


# ---------------------------------------------------------------- core
def thomson_integrals(R, ne, rin, rout, min_points=5):
    """I = int n_e dR over [rin, rout] from the valid chords inside it, the
    fraction of [rin, rout] those chords span, and the point count."""
    ok = np.isfinite(R) & np.isfinite(ne) & (ne > 0) & (R >= rin) & (R <= rout)
    if ok.sum() < min_points or not (rout > rin):
        return np.nan, 0.0, int(ok.sum())
    o = np.argsort(R[ok])
    r, n = R[ok][o], ne[ok][o]
    return float(np.sum(0.5 * (n[1:] + n[:-1]) * np.diff(r))), float((r[-1] - r[0]) / (rout - rin)), int(ok.sum())


def window_mean(t, y, tc, half):
    """Mean of y over |t - tc| <= half; linear interpolation if empty."""
    m = np.abs(t - tc) <= half
    if m.any():
        return float(np.nanmean(y[m]))
    return float(np.interp(tc, t, y, left=np.nan, right=np.nan))


def cross_check(t_ts, R_ts, ne_ts, t_eq, rin_eq, rout_eq, t_ane, ane,
                t0=-np.inf, t1=np.inf, min_coverage=0.9):
    """Per Thomson slice: L, I_TS, A_TS, ANE, coverage.  R_ts is (Nr,) or
    (Nt, Nr); ne_ts is (Nt, Nr)."""
    R_ts = np.broadcast_to(R_ts, ne_ts.shape)
    half = 0.5 * np.median(np.diff(t_ts)) if t_ts.size > 1 else 0.0
    rows = []
    for k, tk in enumerate(t_ts):
        if not (t0 <= tk <= t1) or tk < t_eq[0] or tk > t_eq[-1]:
            continue
        j = int(np.argmin(np.abs(t_eq - tk)))
        rin, rout = rin_eq[j], rout_eq[j]
        if not np.isfinite(rin + rout):
            continue
        I, cov, npts = thomson_integrals(R_ts[k], ne_ts[k], rin, rout)
        a = window_mean(t_ane, ane, tk, half)
        rows.append((tk, rin, rout, rout - rin, I, cov, npts, a))
    names = ("t", "rin", "rout", "L", "I_ts", "coverage", "npts", "ane")
    out = {n: np.array([r[i] for r in rows], float) for i, n in enumerate(names)}
    if not rows:
        out["use"] = np.zeros(0, bool)
        return out
    out["A_ts"] = out["I_ts"] / out["L"]
    out["use"] = (np.isfinite(out["I_ts"]) & np.isfinite(out["ane"])
                  & (out["ane"] > 0) & (out["coverage"] >= min_coverage))
    return out


def fit_beta(L, ane, A):
    """log(ANE/A) = log c + beta log L by least squares; (beta, se, c)."""
    x, y = np.log(L), np.log(ane / A)
    if x.size < 3 or np.ptp(x) == 0:
        return np.nan, np.nan, np.nan
    X = np.c_[np.ones_like(x), x]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    s2 = res @ res / max(x.size - 2, 1)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(coef[1]), float(se), float(np.exp(coef[0]))


def verdict(beta, se):
    if not np.isfinite(beta):
        return "undetermined (too few slices or no variation in L)"
    if se > 0.3:
        return f"undetermined: beta's error {se:.2f} is too large; L does not vary enough"
    if abs(beta - 1) < 3 * se and abs(beta) > 3 * se:
        return "line INTEGRAL (ANE scales with path length)"
    if abs(beta) < 3 * se and abs(beta - 1) > 3 * se:
        return "line AVERAGE (ANE independent of path length)"
    return "ambiguous: neither beta = 0 nor beta = 1 fits within 3 sigma"


# ---------------------------------------------------------------- fetching
def meta(v):
    out = {}
    for k in ("units", "label", "description"):
        try:
            x = getattr(v, k, None)
        except Exception:
            x = None
        if x not in (None, ""):
            out[k] = str(x)
    return out


def show(name, v, extra=""):
    d = np.asarray(v.data)
    m = meta(v)
    print(f"  {name:22s} shape {str(d.shape):>12s}  "
          + "  ".join(f"{k}={m[k]!r}" for k in m) + extra)


def time_first(d, nt, name):
    if d.ndim >= 2 and d.shape[0] != nt and d.shape[-1] == nt:
        return np.moveaxis(d, -1, 0)
    if d.shape[0] != nt:
        raise RuntimeError(f"{name}: shape {d.shape} has no axis of length {nt}")
    return d


def fetch(client, shot, tag, ane_node, ne_node, r_node, z_node):
    print(f"\n  METADATA, shot {shot}")
    v = client.get(ane_node, shot)
    t_ane = np.asarray(v.time.data, float)
    ane = np.asarray(v.data, float)
    show(ane_node, v, f"  t[{t_ane.size}] {t_ane[0]:.4f}..{t_ane[-1]:.4f}")
    try:
        t1 = np.asarray(client.get(ane_node.rsplit("/", 1)[0] + "/TIME1", shot).data, float)
        same = t1.size == t_ane.size and np.allclose(t1, t_ane)
        print(f"  {'ANE/TIME1':22s} t[{t1.size}] {t1[0]:.4f}..{t1[-1]:.4f}  "
              f"{'same as v.time' if same else 'DIFFERS from v.time'}")
    except Exception as e:
        print(f"  {'ANE/TIME1':22s} not read ({type(e).__name__})")

    v = client.get(ne_node, shot)
    t_ts = np.asarray(v.time.data, float)
    ne = np.asarray(v.data, float)
    show(ne_node, v, f"  t[{t_ts.size}]")
    ne = time_first(ne, t_ts.size, ne_node)
    v = client.get(r_node, shot)
    R = np.asarray(v.data, float)
    show(r_node, v)
    if R.ndim == 2:
        R = time_first(R, t_ts.size, r_node)
    zinfo = None
    try:
        v = client.get(z_node, shot)
        zinfo = np.asarray(v.data, float)
        show(z_node, v, f"  median Z {np.nanmedian(zinfo):+.3f} m")
    except Exception as e:
        print(f"  {z_node:22s} not read ({type(e).__name__})")

    t_eq = np.asarray(client.get(f"{tag}/{EPM_TIME}", shot).data, float)
    rb = time_first(np.atleast_2d(np.asarray(client.get(f"{tag}/{RBDRY}", shot).data, float)),
                    t_eq.size, "RBOUNDARY")
    zb = time_first(np.atleast_2d(np.asarray(client.get(f"{tag}/{ZBDRY}", shot).data, float)),
                    t_eq.size, "ZBOUNDARY")
    cr = np.array([midplane_crossings(rb[k], zb[k]) for k in range(t_eq.size)])
    return dict(t_ane=t_ane, ane=ane, t_ts=t_ts, ne=ne, R=R,
                t_eq=t_eq, rin=cr[:, 0], rout=cr[:, 1])


# ---------------------------------------------------------------- output
def report(c, ane_scale):
    u = c["use"]
    print(f"\n  CROSS-CHECK: {u.sum()} of {u.size} Thomson slices used "
          f"(coverage >= threshold, ANE > 0)")
    if u.sum() == 0:
        return np.nan, np.nan, np.nan
    L, a, I, A = c["L"][u], c["ane"][u] * ane_scale, c["I_ts"][u], c["A_ts"][u]
    print(f"  L = R_out - R_in: {L.min():.3f} .. {L.max():.3f} m  "
          f"(range {100 * np.ptp(L) / np.median(L):.0f}% of median)")
    print(f"  Thomson coverage of [R_in, R_out]: median {np.median(c['coverage'][u]):.2f}")
    for nm, r in (("ANE / I_TS  (=1 if integral)", a / I),
                  ("ANE / A_TS  (=1 if average) ", a / A)):
        q = np.percentile(r, [25, 50, 75])
        print(f"  {nm}: median {q[1]:.3g}, IQR {q[0]:.3g}..{q[2]:.3g} "
              f"(spread {100 * (q[2] - q[0]) / q[1]:.0f}%)")
    beta, se, cc = fit_beta(L, a, A)
    print(f"\n  fit ANE/A_TS = c L^beta:  beta = {beta:.2f} +/- {se:.2f},  c = {cc:.3g}")
    print(f"  -> {verdict(beta, se)}")
    print("\n  caveats: assumes a horizontal midplane interferometer chord; SOL "
          "density ignored; c absorbs calibration and any double-pass factor")
    return beta, se, cc


def plot(c, ane_scale, path, title, beta, se):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u = c["use"]
    fig, ax = plt.subplots(3, 1, figsize=(7, 8), sharex=False)
    t = c["t"]
    a = c["ane"] * ane_scale
    ax[0].plot(t, a, ".", color="k", ms=3, label="ANE (x scale)")
    ax[0].plot(t, c["I_ts"], ".", color="C0", ms=3, label="Thomson int n_e dR  [m^-2]")
    ax[0].plot(t, c["A_ts"], ".", color="C3", ms=3, label="Thomson average  [m^-3]")
    ax[0].set_ylabel("density"); ax[0].legend(fontsize=7); ax[0].set_xlabel("t [s]")
    ax[1].plot(t[u], (a / c["I_ts"])[u], ".", color="C0", ms=3, label="ANE / I_TS")
    ax[1].plot(t[u], (a / c["A_ts"])[u], ".", color="C3", ms=3, label="ANE / A_TS")
    b = ax[1].twinx()
    b.plot(t, c["L"], color="0.5", lw=1)
    b.set_ylabel("L [m]", fontsize=8)
    ax[1].set_ylabel("ratio"); ax[1].legend(fontsize=7); ax[1].set_xlabel("t [s]")
    if u.any():
        L = c["L"][u]
        ax[2].loglog(L, (a / c["A_ts"])[u], ".", color="C3", ms=4, label="data")
        Lg = np.linspace(L.min(), L.max(), 2)
        r0 = np.median((a / c["A_ts"])[u])
        ax[2].loglog(Lg, r0 * Lg / np.median(L), "C0--", label="beta = 1 (integral)")
        ax[2].loglog(Lg, r0 + 0 * Lg, "k:", label="beta = 0 (average)")
    ax[2].set_xlabel("L = R_out - R_in [m]"); ax[2].set_ylabel("ANE / A_TS")
    ax[2].set_title(f"beta = {beta:.2f} +/- {se:.2f}", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shot", type=int, required=True)
    p.add_argument("--tag", default="EPM", help="equilibrium tree for the boundary")
    p.add_argument("--ane", default="/ANE/DENSITY")
    p.add_argument("--ne", default="/AYC/N_E", help="Thomson n_e, (time, channel)")
    p.add_argument("--r", default="/AYC/R", help="Thomson chord major radius")
    p.add_argument("--z", default="/AYC/Z", help="Thomson chord height (reported only)")
    p.add_argument("--ane-scale", type=float, default=1.0,
                   help="multiply ANE by this before comparing (it is used raw)")
    p.add_argument("--t0", type=float, default=-np.inf)
    p.add_argument("--t1", type=float, default=np.inf)
    p.add_argument("--min-coverage", type=float, default=0.9)
    p.add_argument("--out", default=None, help="figure path (ane_thomson_<shot>.png)")
    a = p.parse_args()

    import pyuda
    client = pyuda.Client()
    d = fetch(client, a.shot, a.tag.upper(), a.ane, a.ne, a.r, a.z)
    c = cross_check(d["t_ts"], d["R"], d["ne"], d["t_eq"], d["rin"], d["rout"],
                    d["t_ane"], d["ane"], a.t0, a.t1, a.min_coverage)
    beta, se, _ = report(c, a.ane_scale)
    out = a.out or f"ane_thomson_{a.shot}.png"
    plot(c, a.ane_scale, out, f"#{a.shot}: ANE vs Thomson ({a.tag.upper()} boundary)",
         beta, se)
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
