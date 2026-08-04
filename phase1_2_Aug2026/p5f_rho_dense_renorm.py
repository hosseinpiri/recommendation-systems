"""
Norm-preserving dense-rho profiles (audit correction).

The raw update (1-rho)z + rho x on unit vectors shrinks |z_T| heterogeneously
across users, which penalizes every rho > 0 mechanically when a single global
kappa is refit. This version renormalizes z_T to unit norm before evaluation,
for both the article-vector and category-centroid step variants.
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
RHO_GRID = [0.0, 0.01, 0.02, 0.035, 0.05, 0.08]


def main():
    E = np.load(os.path.join(SCRATCH, "news_vec.npy"))
    C = np.load(os.path.join(SCRATCH, "cat_centroids.npy"))
    with open(os.path.join(SCRATCH, "news_vec_idx.pkl"), "rb") as f:
        vi = pickle.load(f)
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        nc = pickle.load(f)
    cat_idx = {c: i for i, c in enumerate(nc["cat_list"])}
    row_cat = np.zeros(E.shape[0], dtype=int)
    for nid, row in vi["idx"].items():
        row_cat[row] = cat_idx[vi["cat"][nid]]

    with open(os.path.join(SCRATCH, "dense_rho_pieces.pkl"), "rb") as f:
        prof = pickle.load(f)
    users = sorted(prof)

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

    def zmat(rho, variant):
        Z = np.zeros((len(users), E.shape[1]))
        for k, u in enumerate(users):
            z = prof[u]["z0"].astype(np.float64).copy()
            if rho > 0:
                for j in prof[u]["clicks"]:
                    x = E[j] if variant == "article" else C[row_cat[j]]
                    z = (1 - rho) * z + rho * x
            n = np.linalg.norm(z)
            Z[k] = z / n if n > 1e-9 else z
        return Z

    out = {"rho_grid": RHO_GRID}
    for variant in ("article", "centroid"):
        lls, kaps = [], []
        for rho in RHO_GRID:
            Z = zmat(rho, variant)
            align = np.einsum("rd,rd->r", Z[uidx], Xdev)
            X = sp.hstack([onehot, sp.csr_matrix(align[:, None])], format="csr")
            clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000,
                                     fit_intercept=False, tol=1e-7)
            clf.fit(X, dev_y)
            u = X @ clf.coef_[0]
            p = 1 / (1 + np.exp(-u))
            eps = 1e-12
            ll = float(np.sum(dev_y * np.log(p + eps)
                              + (1 - dev_y) * np.log(1 - p + eps)))
            lls.append(ll)
            kaps.append(float(clf.coef_[0][-1]))
            print(f"{variant} rho={rho}  logL={ll:.1f}", flush=True)
        i = int(np.argmax(lls))
        out[variant] = {"loglik": lls, "kappa": kaps,
                        "rho_hat": RHO_GRID[i],
                        "LR_vs_zero": float(2 * (lls[i] - lls[0]))}
        print(variant, "rho_hat", RHO_GRID[i], "LR",
              round(out[variant]["LR_vs_zero"], 1))

    with open(os.path.join(OUT, "p5_rho_dense_renorm.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
