"""Boosted-tree discrete-time hazard model on the merged survival table.

The MAST-U port of fdp-demo-hazard's xgb_hazard_common.py.  One row is one
equilibrium slice; Y = 1 only on the last pre-event slice of an event shot
(merge_pair.table_stack).  A per-row logistic loss on that table IS the
discrete-time survival likelihood: the model's p(x) is the probability of
onset within the next slice, h(x)*dt.

    python hazard_model.py --features feat_EPM.pickle --out hazard_EPM.pickle

--features stacks the harvest pickle itself (merge_pair.build_table), so an
EPM-only run needs no merged file; --table reads a merge_pair output instead,
for when EPQ has been paired on.

What carries over from DIII-D, and what does not:

  * Objective.  DIII-D supplies its own gradient and Hessian, g = z - y and
    h = z(1 - z).  Those are exactly XGBoost's built-in binary:logistic, so the
    built-in is used.
  * Base-rate anchoring.  The boosting starts from the training base rate.
    DIII-D passes base_score as LOG-ODDS, which is right for a custom
    objective; with binary:logistic base_score is a PROBABILITY.  Copying the
    DIII-D line here would anchor the model at the wrong rate.
  * Shot-respecting folds.  Slices of one shot are strongly correlated, so
    every split -- the outer test folds and the inner early-stopping folds --
    is by shot, never by row.  A shot is never on both sides of a split.
  * Booster settings: fdp-run-study-xgb.sh's (depth 6, eta 0.10, lambda 0.010,
    min_child_weight 5), hist trees.
  * Calibration is in hazard_predict.py, and goes further than DIII-D's:
    equal-count bins of predicted p with a binomial interval on each observed
    rate, calibration-in-the-large (events observed against sum of p), and a
    Hosmer-Lemeshow statistic against its reference distribution.  DIII-D
    scattered the binned rates against a reference curve and read it by eye.
  * Not ported yet: the screening limits (STANDARD_FIXED_LIMITS are DIII-D
    values, and an ST needs its own), SHAP, elimination paths.

Rows: EPM alone, those with every feature finite (merge_pair's usable1).  The
event ledger merge_pair prints says how many events that costs.

Score: mean test log-likelihood per row against the base-rate model (the
training base rate applied to every test row), per outer fold.  A model that
has learned nothing scores at or below the base; --shuffle-targets is the
control that should.
"""

import argparse
import pickle
import time

import numpy as np

import merge_pair as MP

try:
    import xgboost as xgb
except ImportError:
    xgb = None

#: fdp-run-study-xgb.sh's settings
PARAMS = {
    "booster": "gbtree",
    "tree_method": "hist",
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "min_child_weight": 5.0,
    "eta": 0.10,
    "lambda": 0.010,
    "alpha": 0.0,
    "gamma": 0.0,
    "subsample": 1.0,
}

EPS = 1e-12


def load_table(path, exclude=(), features=False):
    """X, Y, S (shot), T, feature names and the table dict, EPM rows only.

    `path` is a merge_pair output, or with features=True a harvest pickle
    that is stacked here.
    """
    if features:
        d = MP.build_table(path)
    else:
        with open(path, "rb") as fh:
            d = pickle.load(fh)
    names = [n for n in d["Xnames"] if n not in exclude]
    for n in exclude:
        if n not in d["Xnames"]:
            print(f"  note: --exclude {n}: no such feature")
    cols = [d["Xnames"].index(n) for n in names]
    ok = np.asarray(d["usable1"], bool)
    X = np.asarray(d["X1"], float)[np.ix_(ok, cols)]
    Y = (np.asarray(d["Y"])[ok] > 0).astype(float)    # rotating or locked
    S = np.asarray(d["ID"])[ok]
    T = np.asarray(d["T"])[ok]
    lost = int((np.asarray(d["Y"]) > 0).sum() - Y.sum())
    return X, Y, S, T, names, d, lost


def shot_folds(S, K, rng):
    """K arrays of row indices, splitting the SHOTS of S into K groups."""
    shots = np.unique(S)
    rng.shuffle(shots)
    groups = np.array_split(shots, K)
    return [np.flatnonzero(np.isin(S, g)) for g in groups]


def load_model(raw):
    """A fold record's `model` bytes back to a Booster."""
    bst = xgb.Booster()
    bst.load_model(bytearray(raw))
    return bst


def mean_llikl(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit(X, Y, S, params, inner_k, max_rounds, rng, early_stop=25):
    """Fit on these rows; rounds chosen by shot-respecting inner CV."""
    p0 = float(Y.mean())
    if not 0 < p0 < 1:
        raise SystemExit("training rows hold no events (or only events)")
    par = dict(params, base_score=p0)
    d = xgb.DMatrix(X, label=Y)
    folds = []
    for te in shot_folds(S, inner_k, rng):
        tr = np.setdiff1d(np.arange(Y.size), te)
        folds.append((tr, te))
    cv = xgb.cv(par, d, num_boost_round=max_rounds, folds=folds,
                early_stopping_rounds=early_stop, verbose_eval=False)
    rounds = len(cv)            # early stopping truncates cv to the best round
    bst = xgb.train(par, d, num_boost_round=rounds)
    return bst, rounds, p0, float(cv["test-logloss-mean"].iloc[-1])


def evaluate(X, Y, S, params, outer_k, inner_k, repeats, max_rounds, rng,
             verbose=True):
    """Outer shot-respecting K-fold, `repeats` times.

    Returns the per-fold records and the out-of-fold predictions, one column
    per repeat: every row is predicted by a model that never saw its shot.
    Each record carries its booster (`model`, xgboost raw bytes; see
    load_model) and the shots it was fitted on (`train_shots`).
    """
    folds, oof = [], np.full((Y.size, repeats), np.nan)
    for b in range(repeats):
        for k, te in enumerate(shot_folds(S, outer_k, rng)):
            tr = np.setdiff1d(np.arange(Y.size), te)
            assert not np.intersect1d(S[tr], S[te]).size, "shot on both sides"
            t0 = time.time()
            bst, rounds, p0, cvloss = fit(X[tr], Y[tr], S[tr], params,
                                          inner_k, max_rounds, rng)
            p = bst.predict(xgb.DMatrix(X[te]))
            oof[te, b] = p
            rec = {"repeat": b, "fold": k, "rounds": rounds, "cv_logloss": cvloss,
                   "base_rate": p0, "rows": int(te.size),
                   "events": int(Y[te].sum()), "shots": int(np.unique(S[te]).size),
                   "llikl": mean_llikl(Y[te], p),
                   "llikl0": mean_llikl(Y[te], np.full(te.size, p0)),
                   "gain": bst.get_score(importance_type="gain"),
                   "seconds": time.time() - t0,
                   # for inference: the booster, and the shots it must not
                   # be scored on (DIII-D's model_keys)
                   "model": bytes(bst.save_raw()),
                   "train_shots": np.unique(S[tr])}
            folds.append(rec)
            if verbose:
                warn = "  <-- hit max rounds" if rounds >= max_rounds else ""
                print(f"  repeat {b} fold {k}: {rec['events']:4d} events / "
                      f"{rec['rows']:6d} rows, {rounds:4d} rounds, "
                      f"llikl {rec['llikl']:.6f} vs base {rec['llikl0']:.6f}"
                      f"  ({rec['seconds']:.0f} s){warn}")
    return folds, oof


def summarize(folds, names, Y, dt):
    d = np.array([f["llikl"] - f["llikl0"] for f in folds])
    rows = np.array([f["rows"] for f in folds])
    ev = np.array([f["events"] for f in folds])
    l0 = np.array([f["llikl0"] for f in folds])
    p0 = np.mean([f["base_rate"] for f in folds])
    print(f"\n  base rate {p0:.3e} per slice"
          + (f" = {-np.log1p(-p0) / dt:.3f} /s at dt = {1e3 * dt:.3f} ms" if dt else ""))
    print(f"  test llikl gain over base, per row:  {d.mean():.3e} "
          f"+/- {d.std(ddof=1) if d.size > 1 else 0:.1e}  "
          f"(> 0 in {int((d > 0).sum())}/{d.size} folds)")
    print(f"                    per event:         "
          f"{np.mean(d * rows / np.maximum(ev, 1)):.3f} nats")
    print(f"  fraction of the base-rate loss removed: "
          f"{np.mean(d / -l0):.3f}")
    g = np.zeros(len(names))
    for f in folds:
        tot = sum(f["gain"].values()) or 1.0
        for key, v in f["gain"].items():      # keys are f0, f1, ... by column
            g[int(key[1:])] += v / tot
    g /= len(folds)
    order = np.argsort(g)[::-1]
    print("\n  gain importance (share, mean over folds):")
    for i in order[:12]:
        print(f"    {names[i]:8s} {g[i]:.3f}")
    return {"gain_share": dict(zip(names, g.tolist())),
            "llikl_gain_per_row": d.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--features", help="harvest pickle (feat_EPM.pickle); "
                                        "stacked here")
    src.add_argument("--table", help="merge_pair output (merged.pickle)")
    p.add_argument("--out", help="results pickle")
    p.add_argument("--outer-k", type=int, default=8)
    p.add_argument("--inner-k", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1,
                   help="outer K-fold repeats with fresh shot splits (DIII-D: 15)")
    p.add_argument("--max-rounds", type=int, default=1000)
    p.add_argument("--exclude", nargs="*", default=[], help="features to leave out")
    p.add_argument("--shuffle-targets", action="store_true",
                   help="control: permute Y over rows; should score at the base")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max-depth", type=int, default=PARAMS["max_depth"])
    p.add_argument("--eta", type=float, default=PARAMS["eta"])
    p.add_argument("--lambda", dest="lam", type=float, default=PARAMS["lambda"])
    p.add_argument("--min-child-weight", type=float,
                   default=PARAMS["min_child_weight"])
    a = p.parse_args()

    if xgb is None:
        raise SystemExit("xgboost is not installed:  pip install --user xgboost")
    rng = np.random.default_rng(a.seed)

    path = a.features or a.table
    X, Y, S, T, names, d, lost = load_table(path, a.exclude, bool(a.features))
    ann = ((d.get("provenance") or {}).get("primary") or {}).get("annotation") or {}
    print(f"  xgboost {xgb.__version__}")
    print(f"  table {path}: tree {d.get('tag1')}"
          + (f", n={ann.get('n')}, threshold {ann.get('threshold')} G" if ann else
             ", annotation unknown (no provenance)"))
    print(f"  {Y.size} rows, {int(Y.sum())} events, {np.unique(S).size} shots, "
          f"{len(names)} features")
    if lost:
        print(f"    {lost} labelled events are on rows with a NaN feature "
              f"and are not in the fit")
    if a.shuffle_targets:
        Y = rng.permutation(Y)
        print("  CONTROL: targets shuffled over rows")

    params = dict(PARAMS, max_depth=a.max_depth, eta=a.eta, min_child_weight=
                  a.min_child_weight, nthread=a.threads, seed=a.seed)
    params["lambda"] = a.lam
    folds, oof = evaluate(X, Y, S, params, a.outer_k, a.inner_k, a.repeats,
                          a.max_rounds, rng)
    summ = summarize(folds, names, Y, d.get("dt"))

    if a.out:
        with open(a.out, "wb") as fh:
            pickle.dump({"folds": folds, "oof": oof, "Y": Y, "S": S, "T": T,
                         "Xnames": names, "params": params, "args": vars(a),
                         "summary": summ, "provenance": d.get("provenance"),
                         "dt": d.get("dt"), "tag": d.get("tag1"),
                         "table": path, "xgboost": xgb.__version__}, fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
