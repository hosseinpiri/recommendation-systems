"""
Coarse profile likelihood for rho under DENSE features, 30k-user subsample.
Same temporal identification as p1b: z0 from history (dense mean vector),
rolled through train-window clicked-article vectors, scored on dev items with
a logit of category effects + kappa * (z . x).
"""
import json
import os
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

SCRATCH = os.environ.get(
    "SCRATCH",
    "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind",
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
RHO_GRID = [0.0, 0.02, 0.035, 0.05, 0.08, 0.12]


def main():
    E = np.load(os.path.join(SCRATCH, "news_vec.npy"))
    with open(os.path.join(SCRATCH, "dense_rho_pieces.pkl"), "rb") as f:
        prof = pickle.load(f)
    users = sorted(prof)
    print(f"{len(users)} users")

    dev_j, dev_c, dev_y, uidx = [], [], [], []
    for k, u in enumerate(users):
        for (j, c, y) in prof[u]["dev"]:
            dev_j.append(j); dev_c.append(c); dev_y.append(y); uidx.append(k)
    dev_j = np.array(dev_j); dev_c = np.array(dev_c)
    dev_y = np.array(dev_y); uidx = np.array(uidx)
    cnt = np.bincount(dev_c)
    keep = np.where(cnt >= 500)[0]
    m = np.isin(dev_c, keep)
    dev_j, dev_c, dev_y, uidx = dev_j[m], dev_c[m], dev_y[m], uidx[m]
    remap = {int(c): i for i, c in enumerate(keep)}
    R = len(dev_y)
    onehot = sp.csr_matrix((np.ones(R), (np.arange(R),
                            [remap[c] for c in dev_c])), shape=(R, len(keep)))
    Xdev = E[dev_j]
    print(f"{R} dev rows, {len(keep)} categories")

    def zmat(rho):
        Z = np.zeros((len(users), E.shape[1]))
        for k, u in enumerate(users):
            z = prof[u]["z0"].astype(np.float64).copy()
            if rho > 0:
                for j in prof[u]["clicks"]:
                    z = (1 - rho) * z + rho * E[j]
            Z[k] = z
        return Z

    out = {"rho_grid": RHO_GRID, "loglik": [], "kappa": []}
    for rho in RHO_GRID:
        Z = zmat(rho)
        align = np.einsum("rd,rd->r", Z[uidx], Xdev)
        X = sp.hstack([onehot, sp.csr_matrix(align[:, None])], format="csr")
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000,
                                 fit_intercept=False, tol=1e-7)
        clf.fit(X, dev_y)
        u = X @ clf.coef_[0]
        p = 1 / (1 + np.exp(-u))
        eps = 1e-12
        ll = float(np.sum(dev_y * np.log(p + eps) + (1 - dev_y) * np.log(1 - p + eps)))
        out["loglik"].append(ll)
        out["kappa"].append(float(clf.coef_[0][-1]))
        print(f"rho={rho}  logL={ll:.1f}  kappa_dense={clf.coef_[0][-1]:.3f}",
              flush=True)

    i = int(np.argmax(out["loglik"]))
    out["rho_hat_dense"] = RHO_GRID[i]
    out["LR_vs_zero"] = float(2 * (out["loglik"][i] - out["loglik"][0]))
    out["n_users"] = len(users)
    out["n_dev_rows"] = int(R)
    with open(os.path.join(OUT, "p5_rho_dense.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("rho_hat_dense", out["rho_hat_dense"], "LR", round(out["LR_vs_zero"], 1))


if __name__ == "__main__":
    main()
