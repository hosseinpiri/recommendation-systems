"""
Probability-calibration diagnostics and flexible functional form
(review items 6 and 7), on the 3% impression sample with history lengths.

Fits (statsmodels, 60/40 fit/holdout split by row):
  M0  cat effects + kappa * align                  (baseline)
  M1  cat effects + piecewise-linear spline in align (knots 0.02/0.05/0.1/0.2/0.4)
  M2  cat effects + per-category slopes kappa_c
  M3  M0 + kappa x log history-length interaction (users with n >= 5)
  M4  M0 fit on ALL users incl. cold-start (n < 5), align from smoothed state

Holdout report per model: log-loss, Brier, AUC, calibration-in-the-large
(mean p vs mean y), calibration slope (logit of p as single regressor),
and for M0 a reliability table by alignment decile.
"""
import json
import os
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

SCRATCH = "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
KNOTS = [0.02, 0.05, 0.1, 0.2, 0.4]
SEED = 13


def metrics(y, p):
    eps = 1e-12
    ll = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    brier = float(np.mean((p - y) ** 2))
    auc = float(roc_auc_score(y, p))
    citl = {"mean_p": float(p.mean()), "mean_y": float(y.mean())}
    lp = np.log(p + eps) - np.log(1 - p + eps)
    slope_fit = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    return {"logloss": ll, "brier": brier, "auc": auc, "citl": citl,
            "cal_intercept": float(slope_fit.params[0]),
            "cal_slope": float(slope_fit.params[1])}


def main():
    d = np.load(os.path.join(SCRATCH, "hist_rows.npz"))
    cat, align, hl, y = (d["cat"].astype(int), d["align"].astype(float),
                         d["hist_len"].astype(int), d["y"].astype(int))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(y))
    nfit = int(0.6 * len(y))
    fit_i, hold_i = perm[:nfit], perm[nfit:]

    counts = np.bincount(cat, minlength=18)
    keep = np.where(counts >= 2000)[0]
    remap = {c: j for j, c in enumerate(keep)}
    K = len(keep)

    def onehot(idx):
        X = np.zeros((len(idx), K))
        ok = np.isin(cat[idx], keep)
        X[np.arange(len(idx))[ok],
          [remap[c] for c in cat[idx][ok]]] = 1.0
        return X, ok

    def run(idx_fit, idx_hold, cols_fn, label):
        Xf, okf = onehot(idx_fit)
        Xh, okh = onehot(idx_hold)
        idx_fit, idx_hold = idx_fit[okf], idx_hold[okh]
        Xf, Xh = Xf[okf], Xh[okh]
        Ef = cols_fn(idx_fit)
        Eh = cols_fn(idx_hold)
        Xf = np.column_stack([Xf, Ef])
        Xh = np.column_stack([Xh, Eh])
        res = sm.Logit(y[idx_fit], Xf).fit(disp=0, maxiter=300)
        ph = res.predict(Xh)
        m = metrics(y[idx_hold], ph)
        m["n_params"] = int(Xf.shape[1])
        m["llf_fit"] = float(res.llf)
        print(label, {k: round(v, 5) for k, v in m.items()
                      if isinstance(v, float)}, flush=True)
        return res, m, (idx_hold, ph)

    out = {}
    active = hl >= 5
    fa = fit_i[active[fit_i]]
    ha = hold_i[active[hold_i]]

    # M0 baseline
    res0, out["M0"], (hi, p0) = run(fa, ha, lambda i: align[i][:, None], "M0")
    # reliability by alignment decile
    qs = np.quantile(align[hi], np.linspace(0, 1, 11))
    rel = []
    for a, b in zip(qs[:-1], qs[1:]):
        m = (align[hi] >= a) & (align[hi] <= b)
        if m.sum() > 100:
            rel.append({"lo": float(a), "hi": float(b),
                        "mean_p": float(p0[m].mean()),
                        "mean_y": float(y[hi][m].mean()),
                        "n": int(m.sum())})
    out["M0"]["reliability_by_decile"] = rel

    # M1 spline
    def spline_cols(i):
        a = align[i]
        cols = [a]
        for k in KNOTS:
            cols.append(np.maximum(a - k, 0))
        return np.column_stack(cols)
    _, out["M1"], _ = run(fa, ha, spline_cols, "M1 spline")

    # M2 per-category slopes
    def cc_cols(i):
        X = np.zeros((len(i), K))
        ok = np.isin(cat[i], keep)
        X[np.arange(len(i))[ok], [remap[c] for c in cat[i][ok]]] = align[i][ok]
        return X
    res2, out["M2"], _ = run(fa, ha, cc_cols, "M2 kappa_c")
    out["M2"]["kappa_c"] = {str(int(c)): float(res2.params[K + j])
                            for j, c in enumerate(keep)}

    # M3 history interaction
    def hist_cols(i):
        lg = np.log(np.maximum(hl[i], 1)) - np.log(20.0)
        return np.column_stack([align[i], align[i] * lg])
    res3, out["M3"], _ = run(fa, ha, hist_cols, "M3 hist-interaction")
    out["M3"]["kappa_at_n20"] = float(res3.params[K])
    out["M3"]["kappa_log_slope"] = float(res3.params[K + 1])

    # M4 all users, smoothed alignment (Dirichlet c0=10)
    # align stored is raw share; smooth: (share*n + 10/18)/(n+10)
    def sm_align(i):
        n = hl[i].astype(float)
        return ((align[i] * n + 10.0 / 18) / (n + 10.0))[:, None]
    _, out["M4_allusers"], _ = run(fit_i, hold_i, sm_align, "M4 all users")

    with open(os.path.join(OUT, "p16_diagnostics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved p16_diagnostics.json")


if __name__ == "__main__":
    main()
