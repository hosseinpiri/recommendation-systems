"""
Rolling day-pair out-of-time validation of the transition (review item 8).
For each evaluation day d in {12, 13, 14}: state z0 from pre-window history,
rolled through clicks on days 9..d-1, scored on day-d impressions (from the
10% day-sampled rows), refitting (alpha, kappa) per rho. The Nov 15 pair is
the original design. Reports rho_hat and LR per pair.
"""
import json
import os
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

SCRATCH = "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
RHOS = [0.0, 0.02, 0.035, 0.05, 0.08]


def main():
    with open(os.path.join(SCRATCH, "user_hist.pkl"), "rb") as f:
        hist = pickle.load(f)
    with open(os.path.join(SCRATCH, "clicks_by_day.pkl"), "rb") as f:
        cbd = pickle.load(f)
    with open(os.path.join(SCRATCH, "day_rows.pkl"), "rb") as f:
        day_rows = pickle.load(f)

    out = {"rhos": RHOS, "pairs": {}}
    for eval_day in (12, 13, 14):
        users = [u for u in day_rows
                 if eval_day in day_rows[u] and u in hist]
        users = sorted(users)[:120000]
        cats, ys, uidx = [], [], []
        for k, u in enumerate(users):
            cs, yy = day_rows[u][eval_day]
            cats.append(np.array(cs, dtype=int))
            ys.append(np.array(yy, dtype=int))
            uidx.append(np.full(len(cs), k))
        cats = np.concatenate(cats); ys = np.concatenate(ys)
        uidx = np.concatenate(uidx)
        cnt = np.bincount(cats, minlength=18)
        keep = np.where(cnt >= 300)[0]
        m = np.isin(cats, keep)
        cats, ys, uidx = cats[m], ys[m], uidx[m]
        remap = {int(c): j for j, c in enumerate(keep)}
        R = len(ys)
        onehot = sp.csr_matrix((np.ones(R), (np.arange(R),
                                [remap[c] for c in cats])), shape=(R, len(keep)))
        lls = []
        for rho in RHOS:
            Z = np.zeros((len(users), 18))
            for k, u in enumerate(users):
                z = hist[u].astype(np.float64).copy()
                if rho > 0:
                    for (day, t, c) in cbd.get(u, []):
                        if day < eval_day:
                            z *= (1 - rho)
                            z[c] += rho
                Z[k] = z
            align = Z[uidx, cats]
            X = sp.hstack([onehot, sp.csr_matrix(align[:, None])], format="csr")
            clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000,
                                     fit_intercept=False, tol=1e-7)
            clf.fit(X, ys)
            u_ = X @ clf.coef_[0]
            p = 1 / (1 + np.exp(-u_))
            eps = 1e-12
            ll = float(np.sum(ys * np.log(p + eps)
                              + (1 - ys) * np.log(1 - p + eps)))
            lls.append(ll)
            print(f"day {eval_day} rho={rho}: {ll:.1f}", flush=True)
        i = int(np.argmax(lls))
        out["pairs"][str(eval_day)] = {
            "n_users": len(users), "n_rows": int(R), "loglik": lls,
            "rho_hat": RHOS[i], "LR_vs_zero": 2 * (lls[i] - lls[0])}
    with open(os.path.join(OUT, "p18_rolling.json"), "w") as f:
        json.dump(out, f, indent=2)
    print({d: v["rho_hat"] for d, v in out["pairs"].items()})


if __name__ == "__main__":
    main()
