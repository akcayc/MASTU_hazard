"""MAST-U EPM/EPQ node map, and the DIII-D feature it stands in for.

Every descriptor in the DIII-D set has an EPM counterpart, including the three
that looked likely to be lost: S1/S2/S3 (so `glitch` is computable), and
CHISQUARED / POLOIDALFLUXERROR (so the screening ports).  EPM additionally
offers two quality signals DIII-D has no analogue for -- BADCHI2FLAG and
EQUILIBRIUMSTATUSINTEGER -- which are a more direct convergence test than a
chi-squared threshold.

Node paths are relative to the tree tag (`EPM` or `EPQ`), so the same map
serves both halves of the matched pair.

UNVERIFIED on a live shot: every entry below is taken from a node listing, not
from a successful read.  Run `python extract_features.py --probe <shot>` first;
it reports which nodes resolve, their shapes, units and time bases.
"""

#: scalar-vs-time global parameters: feature name -> node path under <TAG>/
GLOBALS = {
    "q0":     "OUTPUT/GLOBALPARAMETERS/Q0",
    "elli":   "OUTPUT/GLOBALPARAMETERS/LI3",      # LI1/LI2/LI3 all exist; LI3 is the usual
    "betap":  "OUTPUT/GLOBALPARAMETERS/BETAP",
    "betat":  "OUTPUT/GLOBALPARAMETERS/BETAT",
    "rcur":   "OUTPUT/GLOBALPARAMETERS/CURRENTCENTROID/R",
    "zcur":   "OUTPUT/GLOBALPARAMETERS/CURRENTCENTROID/Z",
    "shaf1":  "OUTPUT/GLOBALPARAMETERS/S1",
    "shaf2":  "OUTPUT/GLOBALPARAMETERS/S2",
    "shaf3":  "OUTPUT/GLOBALPARAMETERS/S3",
}

#: inputs to derived quantities, not features themselves
SOURCES = {
    "ip":      "OUTPUT/GLOBALPARAMETERS/PLASMACURRENT",
    "bt0":     "OUTPUT/GLOBALPARAMETERS/BVACRMAG",
    "aminor":  "OUTPUT/SEPARATRIXGEOMETRY/MINORRADIUS",
    "rbdry":   "OUTPUT/SEPARATRIXGEOMETRY/RBOUNDARY",
    "zbdry":   "OUTPUT/SEPARATRIXGEOMETRY/ZBOUNDARY",
    "time":    "TIME",
}

#: reconstruction quality -- screening only, never model features.
#: CHISQUARED and POLOIDALFLUXERROR are (time, 30): one value per solver
#: iteration, not per slice.  The converged value is what screening wants, so
#: the FINAL* nodes are used instead.
QUALITY = {
    "chisq":  "OUTPUT/NUMERICALDETAILS/FINALCHISQUARED",
    "error":  "OUTPUT/NUMERICALDETAILS/FINALPOLOIDALFLUXERROR",
}

#: the per-iteration versions, kept for diagnostics (convergence history)
QUALITY_ITER = {
    "chisq_iter": "OUTPUT/NUMERICALDETAILS/CHISQUARED",
    "error_iter": "OUTPUT/NUMERICALDETAILS/POLOIDALFLUXERROR",
}

#: EPM-only convergence flags; no DIII-D analogue
FLAGS = {
    "badchi2": "BADCHI2FLAG",
    "eqstat":  "EQUILIBRIUMSTATUSINTEGER",
}

#: 2-D flux-function profiles, (time, psi_norm)
PROFILES = {
    "psin":   "OUTPUT/FLUXFUNCTIONPROFILES/NORMALIZEDPOLOIDALFLUX",
    "pres":   "OUTPUT/FLUXFUNCTIONPROFILES/STATICPRESSURE",
    "qpsi":   "OUTPUT/FLUXFUNCTIONPROFILES/Q",
    "kappa":  "OUTPUT/FLUXFUNCTIONPROFILES/ELONGATION",
    "tritop": "OUTPUT/FLUXFUNCTIONPROFILES/UPPERTRIANGULARITY",
    "tribot": "OUTPUT/FLUXFUNCTIONPROFILES/LOWERTRIANGULARITY",
}

#: shaping also exists as scalars under SEPARATRIXGEOMETRY.  Preferred when it
#: reads cleanly: FLUXFUNCTIONPROFILES needs [:, -1] to reach the boundary, and
#: the last profile node is not always the separatrix.
#: Confirmed on shot 47002: the SEPARATRIXGEOMETRY triangularities are (time,)
#: scalars and are preferred.  There is no scalar ELONGATION, so kappa must
#: come from the profile's boundary column.
SHAPING_SCALAR = {
    "kappa":  None,
    "tritop": "OUTPUT/SEPARATRIXGEOMETRY/UPPERTRIANGULARITY",
    "tribot": "OUTPUT/SEPARATRIXGEOMETRY/LOWERTRIANGULARITY",
}

#: Reference major radius.  DIII-D reads RZERO and asserts it constant; every
#: length feature is divided by it, so a wrong value rescales the entire
#: geometric half of the feature set.
#:
#: On shot 47002 BOTH obvious candidates returned NaN -- RT and RMIDPLANEOUT
#: resolve as nodes but carry no value.  BVACRADIUSPRODUCT is the B*R product
#: at the reference radius, the same convention as DIII-D's BCENTR-at-RZERO,
#: so R0 = BVACRADIUSPRODUCT / B_vac is the principled route if B_vac at that
#: same radius can be identified.  Until that is settled, R0 falls back to the
#: machine constant and the fallback is recorded per shot.
R0_CANDIDATES = [
    "OUTPUT/GLOBALPARAMETERS/RT",
    "INPUT/BVACRADIUSPRODUCT",          # B*R0; needs dividing by B at R0
]
R0_FALLBACK = 0.85      # m, nominal MAST-U major radius

#: cross-check: the median geometric centre (rmin+rmax)/2 over the campaign
#: should sit near R0.  Reported, never used as the normalisation -- a
#: time-varying R0 would fold plasma motion into every length feature.
R0_CROSSCHECK = True

#: non-EPM signals
EXTERNAL = {
    "ip_meas": "/AMC/PLASMA_CURRENT",     # kA
    "ne_bar":  "/ANE/DENSITY",            # line-integrated, m^-2
}

#: q95 is absent from GLOBALPARAMETERS (Q0, QMIN, Q1/2/3RADIUS only), so it is
#: computed from the Q profile at psi_n = 0.95 -- which is what DIII-D's
#: featurize_qpsi does anyway.
Q95_FROM_PROFILE = True

#: the DIII-D descriptor set, in order, for reference
DIIID_FEATURES = [
    "rmin", "rmax", "zmin", "zmax", "aspr", "kappa", "iphat", "nhat",
    "q0", "q95", "elli", "rcur", "zcur", "betap", "betat",
    "tritop", "tribot", "shaf1", "shaf2", "shaf3",
    "glitch", "pres0", "pres50", "dpres50", "q33", "q67", "dq67",
]
