"""EPM/EPQ descriptor extraction for MAST-U -- the analogue of DIII-D's
extractEqDescriptor.py.

Produces, per shot, a (time x feature) table of DIMENSIONLESS equilibrium
descriptors chopped to the shot's own annotation window.  That dimensionless
normalisation is what lets a DIII-D feature set port to a spherical tokamak at
all, so it is reproduced exactly rather than approximated.

Nine details carried over from the DIII-D code because each is load-bearing
and none is obvious from the feature list:

  1. The table is chopped to [tsof, tevent], NOT [tsof, teof].  Rows after the
     event are discarded -- a survival model must not see the aftermath.
  2. `pres` is interpolated against sqrt(psi_n); `qpsi` against psi_n.  The
     derivatives therefore mean different things, deliberately.
  3. The pressure scale is Bt0^2/(2 mu0)/100, the /100 included: the fixed
     screening limits are calibrated to it.
  4. The boundary arrays are padded; they are truncated per slice before the
     bounding box is taken.
  5. Units are asserted, not trusted.
  6. Time bases are asserted equal across signals -- no silent interpolation.
  7. R0 is asserted constant, and every length is divided by it.
  8. `error`, `chisq` and `glitch` are carried but are reconstruction quality,
     not physics.  Screening uses them; the model should not.
  9. ipsign / btsign are recorded, unused -- they are what the toroidal
     handedness question needs.

    python extract_features.py --probe 47000
    python extract_features.py --annotation rm_ladder_joint.csv --tag EPM \
        --out feat_EPM.pickle
"""

import argparse
import pickle

import numpy as np
import scipy.interpolate as scinterp

import giopath  # noqa: F401
import epm_nodes as NODES

try:
    import pyuda
except ImportError:
    giopath.report()
    raise

MU0 = 4.0e-7 * np.pi


# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------
def _get(client, tag, path, shot):
    """Read <tag>/<path>; return (data, time, units) or None if absent."""
    try:
        v = client.get(f"{tag}/{path}" if path != "TIME" else f"{tag}/TIME", shot)
    except Exception:
        return None
    t = None
    try:
        t = np.asarray(v.time.data)
    except Exception:
        pass
    units = getattr(v, "units", None)
    return np.asarray(v.data), t, units


def probe(shot, tag, client):
    """Report which nodes resolve, with shape, units and time base."""
    groups = [("globals", NODES.GLOBALS), ("sources", NODES.SOURCES),
              ("quality", NODES.QUALITY), ("flags", NODES.FLAGS),
              ("profiles", NODES.PROFILES), ("shaping", NODES.SHAPING_SCALAR)]
    print(f"shot {shot}, tree {tag}\n")
    missing = []
    for gname, g in groups:
        print(f"  [{gname}]")
        for name, path in g.items():
            if path is None:
                print(f"    {name:9s} -- no node defined")
                continue
            r = _get(client, tag, path, shot)
            if r is None:
                print(f"    {name:9s} MISSING   {tag}/{path}")
                missing.append(f"{gname}.{name}")
                continue
            d, t, u = r
            tinfo = f"t[{t.size}] {t[0]:.4f}..{t[-1]:.4f}" if t is not None and t.size else "no time"
            print(f"    {name:9s} {str(d.shape):>14s}  {str(u):>12s}  {tinfo}")
    for name, path in NODES.EXTERNAL.items():
        try:
            v = client.get(path, shot)
            print(f"  [external] {name:9s} {np.asarray(v.data).shape}  {getattr(v,'units',None)}")
        except Exception as e:
            print(f"  [external] {name:9s} MISSING  {path}  ({type(e).__name__})")
            missing.append(f"external.{name}")
    for path in NODES.R0_CANDIDATES:
        r = _get(client, tag, path, shot)
        print(f"  [R0 cand ] {path:46s} {'ok' if r else 'missing'}"
              + (f"  value {np.atleast_1d(r[0])[0]:.4f}" if r else ""))
    print(f"\n  {len(missing)} missing: {missing}" if missing else "\n  all nodes resolved")
    return missing


def fetch_record(shot, tag, client):
    """Pull everything for one shot, asserting units and time bases."""
    rec = {"shot": int(shot), "tag": tag}

    tv = _get(client, tag, NODES.SOURCES["time"], shot)
    if tv is None:
        raise RuntimeError(f"no {tag}/TIME for shot {shot}")
    rec["time"] = np.asarray(tv[0], float)
    nt = rec["time"].size
    if nt < 2:
        raise RuntimeError(f"{tag}/TIME has {nt} points")

    def need(group, name, path):
        r = _get(client, tag, path, shot)
        if r is None:
            raise RuntimeError(f"missing {group}.{name} ({tag}/{path})")
        d = np.asarray(r[0], float)
        # (6) time bases must agree -- no silent interpolation
        if d.ndim >= 1 and d.shape[0] != nt:
            raise RuntimeError(
                f"{name}: leading axis {d.shape[0]} != {tag}/TIME {nt}")
        return d

    for n, p in NODES.GLOBALS.items():
        rec[n] = need("globals", n, p)
    for n, p in NODES.SOURCES.items():
        if n != "time":
            rec[n] = need("sources", n, p)
    for n, p in NODES.QUALITY.items():
        rec[n] = need("quality", n, p)
    for n, p in NODES.PROFILES.items():
        rec[n] = need("profiles", n, p)

    for n, p in NODES.FLAGS.items():                 # optional
        r = _get(client, tag, p, shot)
        rec[n] = np.asarray(r[0], float) if r is not None else None

    # external: line-integrated density, on its own time base
    try:
        v = client.get(NODES.EXTERNAL["ne_bar"], shot)
        rec["ne_bar"] = (np.asarray(v.time.data), np.asarray(v.data, float))
    except Exception:
        rec["ne_bar"] = None

    # (7) reference major radius, asserted constant
    rec["R0"] = None
    for path in NODES.R0_CANDIDATES:
        r = _get(client, tag, path, shot)
        if r is None:
            continue
        d = np.atleast_1d(np.asarray(r[0], float))
        if np.all(np.isfinite(d)) and np.ptp(d) / max(abs(np.mean(d)), 1e-12) < 1e-6:
            rec["R0"] = float(d[0])
            rec["R0_source"] = path
            break
    if rec["R0"] is None:
        rec["R0"] = NODES.R0_FALLBACK
        rec["R0_source"] = "fallback"
    return rec


# ---------------------------------------------------------------------------
# derived quantities
# ---------------------------------------------------------------------------
def derive(rec):
    nt = rec["time"].size

    # (4) boundary arrays are padded -- truncate per slice
    rb, zb = np.atleast_2d(rec["rbdry"]), np.atleast_2d(rec["zbdry"])
    bbox = np.full((nt, 4), np.nan)
    for k in range(nt):
        r, z = rb[k], zb[k]
        ok = np.isfinite(r) & np.isfinite(z) & (r > 0)
        if ok.sum() < 3:
            continue
        bbox[k] = [r[ok].min(), r[ok].max(), z[ok].min(), z[ok].max()]
    rec["bbox"] = bbox

    a = np.asarray(rec["aminor"], float)
    rec["aminor_alt"] = (bbox[:, 1] - bbox[:, 0]) / 2.0
    with np.errstate(invalid="ignore", divide="ignore"):
        rec["kappa_alt"] = (bbox[:, 3] - bbox[:, 2]) / (bbox[:, 1] - bbox[:, 0])
        rec["aspr"] = rec["R0"] / a
        rec["aspr_alt"] = rec["R0"] / rec["aminor_alt"]
        # inverse-q-like current measure
        rec["iphat"] = np.abs(MU0 * rec["ip"] / (rec["bt0"] * a))

    # (9) recorded, unused
    rec["ipsign"] = float(np.sign(np.nanmean(rec["ip"])))
    rec["btsign"] = float(np.sign(np.nanmean(rec["bt0"])))

    # Greenwald-normalised density.  n_GW = |Ip[MA]| / (pi a^2) in 1e20 m^-3;
    # ANE/DENSITY is line-INTEGRATED (m^-2), so a chord length is needed to
    # reach a line-averaged density.  Using the outboard-inboard width of the
    # boundary as that chord.
    rec["nhat"] = np.full(nt, np.nan)
    if rec["ne_bar"] is not None:
        tne, ne = rec["ne_bar"]
        ne_i = np.interp(rec["time"], tne, ne, left=np.nan, right=np.nan)
        chord = bbox[:, 1] - bbox[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            n_bar = 1.0e-20 * ne_i / chord                   # 1e20 m^-3
            ngw = np.abs(rec["ip"] / 1.0e6) / (np.pi * a ** 2)
            rec["nhat"] = n_bar / ngw

    # shaping: profile nodes are (time, psi_n); the boundary is the last column
    for nm in ("kappa", "tritop", "tribot"):
        v = np.asarray(rec[nm], float)
        rec[nm + "_b"] = v[:, -1] if v.ndim == 2 else v
    return rec


# (2) pres on sqrt(psi_n); (3) scale carries the /100
def featurize_pres(rec):
    psin = np.asarray(rec["psin"], float)
    psin = psin[0] if psin.ndim == 2 else psin
    pscale = (np.asarray(rec["bt0"], float) ** 2) / (2 * MU0) / 100.0
    f = scinterp.PchipInterpolator(np.sqrt(psin), np.asarray(rec["pres"], float), axis=1)
    return (np.column_stack([f(0.0) / pscale, f(0.50) / pscale,
                             f(0.50, nu=1) / pscale]),
            ["pres0", "pres50", "dpres50"])


# (2) qpsi on psi_n directly
def featurize_qpsi(rec):
    psin = np.asarray(rec["psin"], float)
    psin = psin[0] if psin.ndim == 2 else psin
    f = scinterp.PchipInterpolator(psin, np.asarray(rec["qpsi"], float), axis=1)
    cols = [f(0.33), f(0.67), f(0.67, nu=1)]
    names = ["q33", "q67", "dq67"]
    if NODES.Q95_FROM_PROFILE:          # q95 is absent from GLOBALPARAMETERS
        cols.insert(0, f(0.95))
        names.insert(0, "q95")
    return np.column_stack(cols), names


def make_descriptor_table(rec, use_alts=False, include_glitch=True):
    nt = rec["time"].size
    R0 = rec["R0"]
    cols, names = [], []

    def add(nm, v):
        names.append(nm)
        cols.append(np.asarray(v, float).reshape(nt))

    add("rmin", rec["bbox"][:, 0] / R0)
    add("rmax", rec["bbox"][:, 1] / R0)
    add("zmin", rec["bbox"][:, 2] / R0)
    add("zmax", rec["bbox"][:, 3] / R0)
    add("aspr", rec["aspr_alt"] if use_alts else rec["aspr"])
    add("kappa", rec["kappa_alt"] if use_alts else rec["kappa_b"])
    add("iphat", rec["iphat"])
    add("nhat", rec["nhat"])
    add("q0", rec["q0"])
    add("elli", rec["elli"])
    add("rcur", rec["rcur"] / R0)
    add("zcur", rec["zcur"] / R0)
    add("betap", rec["betap"])
    add("betat", rec["betat"])
    add("tritop", rec["tritop_b"])
    add("tribot", rec["tribot_b"])
    add("shaf1", rec["shaf1"])
    add("shaf2", rec["shaf2"])
    add("shaf3", rec["shaf3"])

    if include_glitch:       # (8) Shafranov residual: reconstruction self-consistency
        beplih = np.asarray(rec["betap"], float) + np.asarray(rec["elli"], float) / 2
        s1s2rc = (np.asarray(rec["shaf1"], float) / 4
                  + (np.asarray(rec["shaf2"], float) / 4)
                  * (1 + np.asarray(rec["rcur"], float) / R0))
        add("glitch", s1s2rc - beplih)

    P, pn = featurize_pres(rec)
    for i, n in enumerate(pn):
        add(n, P[:, i])
    Q, qn = featurize_qpsi(rec)
    for i, n in enumerate(qn):
        add(n, Q[:, i])

    add("error", rec["error"])        # (8) screening only
    add("chisq", rec["chisq"])
    return rec["time"], np.column_stack(cols), names


def extract_shot(shot, tag, client, ta=None, tb=None, use_alts=False,
                 include_glitch=True, min_timestamps=5):
    rec = derive(fetch_record(shot, tag, client))
    T, X, names = make_descriptor_table(rec, use_alts, include_glitch)

    # (1) chop to [tsof, tevent] -- NOT to end of flat-top
    if ta is not None and tb is not None:
        keep = (T >= ta) & (T <= tb)
        T, X = T[keep], X[keep, :]
    if T.size < min_timestamps:
        raise RuntimeError(f"only {T.size} slices in window (need {min_timestamps})")

    return {"shot": int(shot), "tag": tag, "time": T, "features": X,
            "feature-names": names, "R0": rec["R0"], "R0_source": rec["R0_source"],
            "ipsign": rec["ipsign"], "btsign": rec["btsign"],
            "ta": ta, "tb": tb,
            "badchi2": rec.get("badchi2"), "eqstat": rec.get("eqstat")}


def read_annotation(path):
    """shot, tsof, teof, trm, tlm, tevent, ievent -- window is [tsof, tevent]."""
    D = np.genfromtxt(path, delimiter=",", skip_header=1)
    D = np.atleast_2d(D)
    assert D.shape[1] == 7, f"expected 7 columns, got {D.shape[1]}"
    out = {}
    for row in D:
        s = int(row[0])
        out[s] = {"shot": s, "ta": float(row[1]), "tb": float(row[5]),
                  "event": int(row[6])}
        assert out[s]["ta"] < out[s]["tb"], f"shot {s}: ta >= tb"
        assert out[s]["event"] in (0, 1, 2)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=int, help="report node availability for this shot")
    p.add_argument("--tag", default="EPM", help="EPM or EPQ")
    p.add_argument("--annotation", help="joint annotation CSV (7 columns)")
    p.add_argument("--shots", type=int, nargs="+", help="override the CSV shot list")
    p.add_argument("--out", help="pickle to write")
    p.add_argument("--use-alts", action="store_true",
                   help="aspr/kappa from the bounding box instead of the nodes")
    p.add_argument("--no-glitch", action="store_true")
    p.add_argument("--min-timestamps", type=int, default=5)
    a = p.parse_args()

    client = pyuda.Client()
    if a.probe:
        return 0 if not probe(a.probe, a.tag.upper(), client) else 1

    if not a.annotation:
        p.error("--annotation is required (or use --probe)")
    ann = read_annotation(a.annotation)
    shots = a.shots or sorted(ann)

    out, dropped = [], {}
    for s in shots:
        if s not in ann:
            dropped[s] = "not in annotation file"
            continue
        try:
            r = extract_shot(s, a.tag.upper(), client, ann[s]["ta"], ann[s]["tb"],
                             a.use_alts, not a.no_glitch, a.min_timestamps)
            r["event"] = ann[s]["event"]
            out.append(r)
            print(f"{s}: {r['features'].shape[0]} slices x "
                  f"{r['features'].shape[1]} features, R0={r['R0']:.4f} "
                  f"({r['R0_source']})")
        except Exception as e:
            dropped[s] = f"{type(e).__name__}: {e}"
            print(f"{s}: dropped -- {e}")

    print(f"\n{len(out)} kept, {len(dropped)} dropped")
    if out:
        dt = np.median(np.diff(out[0]["time"]))
        print(f"  slice cadence (shot {out[0]['shot']}): {1e3*dt:.3f} ms"
              f"  <- this is the Delta in the hazard link")
    if a.out:
        with open(a.out, "wb") as f:
            pickle.dump({"records": out, "dropped": dropped, "tag": a.tag.upper()}, f)
        print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
