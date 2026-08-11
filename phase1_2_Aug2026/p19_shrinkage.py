"""
Hierarchical (Dirichlet-smoothed) latent state including cold-start users
(review item 5). State z(c0) = (counts + c0/18) / (n + c0). The concentration
c0 is chosen by held-out likelihood; kappa is re-estimated under the best c0
on ALL users (no history filter) and on cold-start users (n < 5) separately;
the rho profile re-runs with smoothed initial states to check robustness.
"""
import json
import os
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

SCRATCH = "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
C0_GRID = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
RHOS = [0.0, 0.02, 0.035, 0.05]


def fit_ll(cats, ys, aligns, keep, holdout_frac=0.4, seed=3):
    remap = {int(c): j for j, c in enumerate(keep)}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ys))
    nf = int((1 - holdout_frac) * len(ys))
    fi, hi = perm[:nf], perm[nf:]
    R = len(ys)
    onehot = sp.csr_matrix((np.ones(R), (np.arange(R),
                            [remap[c] for c in cats])), shape=(R, len(keep)))
    X = sp.hstack([onehot, sp.csr_matrix(aligns[:, None])], format="csr")
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000,
                             fit_intercept=False, tol=1e-7)
    clf.fit(X[fi], ys[fi])
    p = 1 / (1 + np.exp(-(X[hi] @ clf.coef_[0])))
    eps = 1e-12
    ll = float(np.mean(ys[hi] * np.log(p + eps)
                       + (1 - ys[hi]) * np.log(1 - p + eps)))
    return ll, float(clf.coef_[0][-1])


def main():
    d = np.load(os.path.join(SCRATCH, "hist_rows.npz"))
    cat, align, hl, y = (d["cat"].astype(int), d["align"].astype(float),
                         d["hist_len"].astype(float), d["y"].astype(int))
    counts_c = align * hl                    # raw category count for the row
    keep = np.where(np.bincount(cat, minlength=18) >= 2000)[0]
    m = np.isin(cat, keep)
    cat, counts_c, hl, y = cat[m], counts_c[m], hl[m], y[m]

    out = {"c0_grid": C0_GRID, "holdout_ll_per_row": {}, "kappa": {}}
    best = (-1e18, None)
    for c0 in C0_GRID:
        denom = np.maximum(hl + c0, 1e-9)
        sm_align = np.where(hl + c0 > 0,
                            (counts_c + c0 / 18.0) / denom, 1.0 / 18)
        ll, kap = fit_ll(cat, y, sm_align, keep)
        out["holdout_ll_per_row"][str(c0)] = ll
        out["kappa"][str(c0)] = kap
        if ll > best[0]:
            best = (ll, c0)
        print(f"c0={c0}: holdout ll/row={ll:.6f} kappa={kap:.3f}", flush=True)
    c0h = best[1]
    out["c0_hat"] = c0h

    cold = hl < 5
    sm_align = (counts_c + c0h / 18.0) / (hl + c0h)
    ll_c, kap_c = fit_ll(cat[cold], y[cold], sm_align[cold],
                         np.where(np.bincount(cat[cold], minlength=18) >= 500)[0])
    out["cold_start"] = {"n_rows": int(cold.sum()), "kappa": kap_c,
                         "share_rows": float(cold.mean())}
    print(f"cold-start rows {cold.sum()}: kappa={kap_c:.3f}", flush=True)

    # rho profile with smoothed z0 (established users, standard design)
    import p1b_transition as T
    cat_list, hist, clicks, shown, dev, users = T.load()
    rng = np.random.default_rng(7)
    users = sorted(rng.choice(users, 60000, replace=False).tolist())
    ev = T.stack_events(users, clicks, shown)
    with open(os.path.join(SCRATCH, "states_all.pkl"), "rb") as f:
        counts_all = pickle.load(f)
    hist_sm = {}
    for u in users:
        cnt = counts_all[u].astype(np.float64)
        hist_sm[u] = ((cnt + c0h / 18.0) / (cnt.sum() + c0h)).astype(np.float32)
    all_cats = np.concatenate([dev[u][0] for u in users])
    keep_cats = np.where(np.bincount(all_cats, minlength=18) >= 500)[0]
    remap = {int(c): j for j, c in enumerate(keep_cats)}
    D = T.Design(users, hist_sm, ev, dev, keep_cats, remap)
    lls = []
    for rho in RHOS:
        ll, kap = D.maxll(D.aligns(float(rho), 0.0))
        lls.append(ll)
        print(f"smoothed-z0 rho={rho}: {ll:.1f}", flush=True)
    i = int(np.argmax(lls))
    out["rho_profile_smoothed"] = {"rhos": RHOS, "loglik": lls,
                                   "rho_hat": RHOS[i],
                                   "LR_vs_zero": 2 * (lls[i] - lls[0])}
    with open(os.path.join(OUT, "p19_shrinkage.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("c0_hat", c0h, "rho_hat(smoothed)", RHOS[i])


if __name__ == "__main__":
    main()
