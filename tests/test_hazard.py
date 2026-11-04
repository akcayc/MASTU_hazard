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
    summ = HM.summarize(folds, names, Y, None)
    top = max(summ["gain_share"], key=summ["gain_share"].get)
    check(f"x0 ranked first (got {top})", top == "x0")

    rng = np.random.default_rng(2)
    Ys = rng.permutation(Y)
    folds, _ = HM.evaluate(X, Ys, S, par, 4, 3, 1, 300, rng, verbose=False)
    gain_s = np.array([f["llikl"] - f["llikl0"] for f in folds])
    check(f"shuffled targets do not beat the base (gain {gain_s.mean():.2e}/row)",
          gain_s.mean() <= 0.1 * gain.mean())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
