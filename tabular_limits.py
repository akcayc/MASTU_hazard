"""Screening limits for the survival table: the port of DIII-D's
fdp-demo-hazard/tabular_limits.py.

A limit set is a dict keyed "<feature>_min" / "<feature>_max".  A row passes
when x > min and x <= max on every feature that has a key; a feature with no
key is not screened.  hazard_model.py --limits picks the set:

  fixed   MASTU_FIXED_LIMITS (DIII-D's default mode, STANDARD_FIXED_LIMITS)
  auto    the 0.05% / 99.95% quantiles of every column, tightened by
          MASTU_MANUAL_LIMITS wherever the manual limit is the more
          restrictive (DIII-D: SCREEN_MANUAL_LIMITS)
  none    no screening

quantile_limits, adjust_limits and apply_screening are DIII-D's.  Added:
parse_set_limits (--set-limit overrides), unknown_keys (DIII-D ignores a key
naming no column silently; here it is reported) and screening_report (what
each limit removes).  As in DIII-D, a NaN fails every comparison, so a row
with a NaN in a screened column fails.
"""

import numpy as np

#: DIII-D's values, verbatim -- tokamak values, kept for reference and for
#: feature_diagnostics.py's reference lines.  Not used for MAST-U.
STANDARD_FIXED_LIMITS = {
    "chisq_max": 90.0,
    "error_max": 5.0e-3,
    "nhat_min": 0.0,
    "nhat_max": 2.0,
    "betap_min": 0.0,
    "betat_min": 0.0,
    "q0_min": 0.0,
    "q0_max": 10.0,
    "q95_min": 1.0,
    "q95_max": 20.0,
    "pres0_min": -20.0,
    "pres0_max": 50.0,
    "pres50_min": -20.0,
    "pres50_max": 50.0,
    "dpres50_min": -50.0,
    "dpres50_max": 20.0,
}

#: DIII-D's SCREEN_MANUAL_LIMITS, verbatim, for reference
DIIID_MANUAL_LIMITS = {
    "chisq_max": 90.0,
    "error_max": 5.0e-3,
    "nhat_min": 0.0,
    "nhat_max": 2.0,
    "betap_min": 0.0,
    "betat_min": 0.0,
}

SCREEN_AUTO_QUANTILE_LOW = 0.0005
SCREEN_AUTO_QUANTILE_HIGH = 0.9995

#: MAST-U fixed limits.  Only what holds by definition is set:
#:   betap, betat   positive-definite (pressure over field energy)
#:   nhat           a density; now 1e-20 * ANE/DENSITY, not Greenwald-normalised
#: Deliberately NOT set until feature_diagnostics.py has been read on real
#: data:
#:   chisq, error   EPM's scales are not EFIT's; DIII-D's 90 and 5e-3 do not port
#:   nhat_max       DIII-D's 2.0 was a Greenwald fraction
#:   q0, q95, q33, q67, dq67
#:                  SIGNED in our features (featurize_qpsi takes no abs) and
#:                  MAST-U q can be negative (left-handed), so DIII-D's
#:                  q0_min 0 / q95_min 1 would remove every such shot
#:   pres*          same normalisation as DIII-D, but the bounds are DIII-D's
#:                  judgement on DIII-D distributions
MASTU_FIXED_LIMITS = {
    "betap_min": 0.0,
    "betat_min": 0.0,
    "nhat_min": 0.0,
}

#: MAST-U counterpart of SCREEN_MANUAL_LIMITS, used by --limits auto
MASTU_MANUAL_LIMITS = dict(MASTU_FIXED_LIMITS)


def quantile_limits(X, Xnames, qa=SCREEN_AUTO_QUANTILE_LOW,
                    qb=SCREEN_AUTO_QUANTILE_HIGH):
    """{name_min, name_max} at the qa / qb quantiles of each column."""
    L = dict()
    for x in Xnames:
        ix = Xnames.index(x)
        xa, xb = np.quantile(X[:, ix], [qa, qb])
        L["%s_min" % x] = float(xa)
        L["%s_max" % x] = float(xb)
    return L


def adjust_limits(L, scr, override=False):
    """In place: for keys already in L, take scr's value if override, else
    only where scr is the more restrictive.  Keys absent from L are ignored,
    as in DIII-D."""
    for k in L.keys():
        if k not in scr.keys():
            continue
        if override:
            L[k] = scr[k]
        else:
            if k.endswith("_min") and scr[k] > L[k]:
                L[k] = scr[k]
            if k.endswith("_max") and scr[k] < L[k]:
                L[k] = scr[k]


def apply_screening(X, Xnames, scr=None):
    """Rows passing every limit in scr: x > min and x <= max."""
    assert scr is not None, "Must provide dict with screen limits (can be empty)"
    ok = np.tile(True, (X.shape[0],))
    for x in Xnames:
        if ("%s_min" % x) in scr.keys():
            ok = np.logical_and(ok, X[:, Xnames.index(x)] > scr["%s_min" % x])
        if ("%s_max" % x) in scr.keys():
            ok = np.logical_and(ok, X[:, Xnames.index(x)] <= scr["%s_max" % x])
    return ok


def unknown_keys(scr, Xnames):
    """Limit keys naming no column -- DIII-D ignores them silently."""
    return [k for k in scr
            if not (k.endswith(("_min", "_max")) and k[:-4] in Xnames)]


def parse_set_limits(specs):
    """['nhat::5', 'q95:-20:-1'] -> {'nhat_max': 5.0, 'q95_min': -20.0,
    'q95_max': -1.0}.  An empty side sets nothing."""
    out = {}
    for sp in specs or ():
        parts = sp.split(":")
        if len(parts) != 3 or not parts[0] or not (parts[1] or parts[2]):
            raise SystemExit(f"--set-limit {sp!r}: expected NAME:MIN:MAX with "
                             f"at least one bound")
        try:
            if parts[1]:
                out[parts[0] + "_min"] = float(parts[1])
            if parts[2]:
                out[parts[0] + "_max"] = float(parts[2])
        except ValueError:
            raise SystemExit(f"--set-limit {sp!r}: bounds must be numbers")
        if parts[1] and parts[2] and not float(parts[1]) < float(parts[2]):
            raise SystemExit(f"--set-limit {sp!r}: MIN must be below MAX")
    return out


def screening_report(X, Y, S, Xnames, scr):
    """Per limit key: rows, event rows and shots it fails on its own."""
    rows = []
    for k in sorted(scr):
        n = k[:-4]
        if n not in Xnames:
            continue
        x = X[:, Xnames.index(n)]
        fail = ~(x > scr[k]) if k.endswith("_min") else ~(x <= scr[k])
        rows.append({"key": k, "value": float(scr[k]), "rows": int(fail.sum()),
                     "events": int((fail & (Y > 0)).sum()),
                     "shots": int(np.unique(S[fail]).size)})
    return rows
