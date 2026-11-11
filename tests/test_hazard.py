"""hazard_model: a synthetic survival table with a known hazard.

Each shot is a run of slices; the per-slice onset probability depends on one
feature (x0) through a logistic, the others are noise.  A shot ends at its
first onset (Y = 1 on that row) or is censored.  The model must beat the
base rate on held-out shots and rank x0 first; the shuffled-target control
must not beat it.  Skipped, not failed, when xgboost is not installed.
"""
import _paths  # noqa: F401
import numpy as np

import hazard_model as HM


def table(n_shots=400, n_slices=60, n_feat=5, seed=0):
    rng = np.random.default_rng(seed)
    X, Y, S = [], [], []
    for s in range(n_shots):
        x = rng.normal(size=(n_slices, n_feat))
        x[:, 0] = np.cumsum(rng.normal(0, 0.3, n_slices)) + rng.normal()  # drifts
        p = 1 / (1 + np.exp(-(-5.0 + 1.5 * x[:, 0])))
        hit = np.flatnonzero(rng.random(n_slices) < p)
        end = hit[0] + 1 if hit.size else n_slices
        y = np.zeros(end)
        if hit.size:
            y[-1] = 1
        X.append(x[:end]); Y.append(y); S.append(np.full(end, s))
    return np.vstack(X), np.concatenate(Y), np.concatenate(S)


def main():
    if HM.xgb is None:
        print("  SKIP  xgboost not installed")
        return 0
    ok = True

    def check(label, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")

    X, Y, S = table()
    names = [f"x{i}" for i in range(X.shape[1])]
    print(f"    {Y.size} rows, {int(Y.sum())} events, {np.unique(S).size} shots")
    par = dict(HM.PARAMS, nthread=2, seed=0)

    rng = np.random.default_rng(1)
    folds, oof = HM.evaluate(X, Y, S, par, 4, 3, 1, 300, rng, verbose=False)
    gain = np.array([f["llikl"] - f["llikl0"] for f in folds])
    check(f"beats the base rate on every held-out fold (gain {gain.mean():.2e}/row)",
          np.all(gain > 0))
    check("every row predicted out of fold", np.all(np.isfinite(oof)))
    # the stored booster reproduces the out-of-fold predictions it made
    f0 = folds[0]
    te = np.flatnonzero(~np.isin(S, f0["train_shots"]))
    p_re = HM.load_model(f0["model"]).predict(HM.xgb.DMatrix(X[te]))
    check("stored fold model reproduces its oof predictions",
          np.allclose(p_re, oof[te, 0], rtol=1e-6))
    check("train_shots and held-out shots are disjoint",
          not np.intersect1d(f0["train_shots"], S[te]).size)
    summ = HM.summarize(folds, names, Y, None)
    top = max(summ["gain_share"], key=summ["gain_share"].get)
    check(f"x0 ranked first (got {top})", top == "x0")

    # --features: per-shot records stacked by merge_pair.build_table
    import os, pickle, tempfile
    recs = []
    for s_ in np.unique(S):
        r = S == s_
        t = 0.1 + 5.025e-3 * np.arange(r.sum())
        recs.append({"shot": int(s_), "tag": "EPM", "time": t, "features": X[r],
                     "feature-names": names, "event": int(Y[r][-1]), "tb": t[-1]})
    path = os.path.join(tempfile.mkdtemp(), "feat_EPM.pickle")
    with open(path, "wb") as fh:
        pickle.dump({"records": recs, "dropped": {}, "tag": "EPM",
                     "provenance": None}, fh)
    Xf, Yf, Sf, _, nf, _, _ = HM.load_table(path, features=True)
    check("--features stacks to the same table",
          np.array_equal(Xf, X) and np.array_equal(Yf, Y) and np.array_equal(Sf, S)
          and nf == names)

    # screening in load_table: chisq/error kept in the table, screened on,
    # never fitted on; a failing row is dropped, event rows included (DIII-D)
    recs_q = []
    for r_ in recs:
        r2 = dict(r_)
        n_ = len(r_["time"])
        r2["features"] = np.c_[r_["features"], np.full(n_, 10.0), np.full(n_, 1e-4)]
        r2["feature-names"] = names + ["chisq", "error"]
        recs_q.append(r2)
    iev = next(i_ for i_, r_ in enumerate(recs_q) if r_["event"])
    recs_q[iev]["features"][-1, -2] = 500.0        # an event shot's LAST row: chisq
    path_q = os.path.join(tempfile.mkdtemp(), "feat_q.pickle")
    with open(path_q, "wb") as fh:
        pickle.dump({"records": recs_q, "dropped": {}, "tag": "EPM",
                     "provenance": None}, fh)
    Xq, Yq, Sq, _, nq, dq, _ = HM.load_table(path_q, features=True, limits="none",
                                             set_limits={"chisq_max": 100.0})
    s0 = recs_q[iev]["shot"]
    check("chisq/error screened on, not fitted on",
          "chisq" not in nq and "error" not in nq and nq == names)
    check("a failing row is dropped, its shot kept (event row too)",
          Yq.size == Y.size - 1 and (Sq == s0).sum() == (S == s0).sum() - 1
          and dq["screened"]["events"] - dq["screened"]["events_kept"] == 1
          and Yq[Sq == s0].sum() == 0)
    rep = {r_["key"]: r_ for r_ in dq["screening"]}
    check("report counts the failing row", rep["chisq_max"]["rows"] == 1)
    _, Ya, _, _, _, da, _ = HM.load_table(path_q, features=True, limits="auto")
    check("auto: quantile limits on every column, error included",
          {"x0_min", "x0_max", "error_max"} <= set(da["limits"]["values"])
          and Ya.size < Y.size)
    _, Yn, _, _, _, dn, _ = HM.load_table(path_q, features=True, limits="none")
    check("none: nothing screened", Yn.size == Y.size and not dn["limits"]["values"])

    rng = np.random.default_rng(2)
    Ys = rng.permutation(Y)
    folds, _ = HM.evaluate(X, Ys, S, par, 4, 3, 1, 300, rng, verbose=False)
    gain_s = np.array([f["llikl"] - f["llikl0"] for f in folds])
    check(f"shuffled targets do not beat the base (gain {gain_s.mean():.2e}/row)",
          gain_s.mean() <= 0.1 * gain.mean())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
