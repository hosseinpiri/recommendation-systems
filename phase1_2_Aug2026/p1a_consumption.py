"""
Phase 1a: structural estimation of the consumption model

    P(click | z, item i) = sigma( alpha_{cat(i)} + kappa * z' x_i ),
    x_i = e_{cat(i)}  =>  z' x_i = z[cat(i)]  (user's history share of the item's category).

Impression-item design built by p0_prepare.py from a 15% sample of train impressions
(users with >= 5 history clicks). Fits by ML logit (statsmodels) on a random subsample,
reports kappa with CI, per-category alpha with CI, and AUC on a held-out subsample.
"""
import json
import os
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

SCRATCH = os.environ.get(
    "SCRATCH",
    "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind",
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

FIT_N = 2_000_000
TEST_N = 1_000_000
SEED = 7


def main():
    import pickle
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        cat_list = pickle.load(f)["cat_list"]
    K = len(cat_list)

    d = np.load(os.path.join(SCRATCH, "impr_1a.npz"))
    cat, align, y = d["cat"].astype(int), d["align"], d["y"].astype(int)
    n = len(y)
    print(f"{n} rows, click rate {y.mean():.4f}")

    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    fit_idx = perm[:FIT_N]
    test_idx = perm[FIT_N:FIT_N + TEST_N]

    # drop categories too rare to carry their own intercept; keep those with >= 200 fit rows
    counts = np.bincount(cat[fit_idx], minlength=K)
    keep = np.where(counts >= 200)[0]
    keep_set = set(keep.tolist())
    print(f"categories kept: {len(keep)} of {K} "
          f"(dropped: {[cat_list[i] for i in range(K) if i not in keep_set and counts[i] > 0]})")
    remap = {c: j for j, c in enumerate(keep)}

    def design(idx):
        m = np.isin(cat[idx], keep)
        idx = idx[m]
        X = np.zeros((len(idx), len(keep) + 1))
        for r, c in enumerate(cat[idx]):
            X[r, remap[c]] = 1.0
        X[:, -1] = align[idx]
        return X, y[idx], idx

    Xf, yf, _ = design(fit_idx)
    model = sm.Logit(yf, Xf)
    res = model.fit(disp=1, maxiter=200)

    kappa = res.params[-1]
    kse = res.bse[-1]
    alpha = res.params[:-1]
    ase = res.bse[:-1]

    Xt, yt, _ = design(test_idx)
    p = res.predict(Xt)
    auc = roc_auc_score(yt, p)
    ll_model = res.llf / len(yf)

    # null model: intercepts only (no alignment) for LR test of kappa
    res0 = sm.Logit(yf, Xf[:, :-1]).fit(disp=0, maxiter=200)
    lr = 2 * (res.llf - res0.llf)

    out = {
        "n_fit": int(len(yf)),
        "n_test": int(len(yt)),
        "click_rate_fit": float(yf.mean()),
        "kappa": float(kappa),
        "kappa_se": float(kse),
        "kappa_ci95": [float(kappa - 1.96 * kse), float(kappa + 1.96 * kse)],
        "LR_kappa_vs_null": float(lr),
        "auc_holdout": float(auc),
        "mean_loglik_fit": float(ll_model),
        "alpha_by_category": {
            cat_list[c]: {"alpha": float(alpha[j]), "se": float(ase[j]),
                          "n_fit": int(counts[c])}
            for c, j in remap.items()
        },
    }
    with open(os.path.join(OUT, "p1a_consumption.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "alpha_by_category"}, indent=2))
    print("alpha range:", float(alpha.min()), "to", float(alpha.max()))


if __name__ == "__main__":
    main()
