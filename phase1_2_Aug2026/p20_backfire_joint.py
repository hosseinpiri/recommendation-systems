"""
Joint backfire estimation on the FULL impression scale (review item 12).
30k-user subsample with every shown-not-clicked event retained (no 1-in-10
subsampling). Joint 2-D profile over (rho, rho_ns_per_impression), two
resistance specifications:

  affine  z <- (1 - rho_ns) z + rho_ns e_c, rho_ns < 0, per-step clip+renorm
  prop    proportional redistribution (simplex-native):
          z_c <- (1 - delta) z_c;  z_k <- z_k + delta z_c z_k / (1 - z_c)

Evaluation on the users' dev-day impressions, refitting (alpha, kappa) per
grid point, exactly the temporal design of the main estimation.
"""
import json
import os
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

SCRATCH = "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
RHOS = [0.02, 0.035, 0.05]
NS_GRID = [0.0, -0.0005, -0.001, -0.002, -0.005]
DELTA_GRID = [0.0, 0.001, 0.002, 0.005, 0.0075, 0.01, 0.015, 0.02]


def z_final_correct(z0, events, rho, mag, spec):
    z = z0.astype(np.float64).copy()
    for _, c, isclick in events:
        if isclick:
            if rho > 0:
                z *= (1 - rho)
                z[c] += rho
        elif mag != 0.0:
            if spec == "affine":
                z *= (1 + mag)
                z[c] -= mag
                np.clip(z, 0.0, None, out=z)
                z /= z.sum()
            else:  # proportional redistribution
                zc = z[c]
                if 0.0 < zc < 1.0:
                    moved = mag * zc
                    z *= 1.0 + moved / (1.0 - zc)
                    z[c] = (zc - moved) * 1.0
    return z


def main():
    with open(os.path.join(SCRATCH, "user_hist.pkl"), "rb") as f:
        hist = pickle.load(f)
    with open(os.path.join(SCRATCH, "train_clicks.pkl"), "rb") as f:
        clicks = pickle.load(f)
    with open(os.path.join(SCRATCH, "shown_full.pkl"), "rb") as f:
        shown = pickle.load(f)
    with open(os.path.join(SCRATCH, "dev_rows.pkl"), "rb") as f:
        dev = pickle.load(f)
    users = sorted(u for u in shown if u in dev and u in hist)
    print(f"{len(users)} users, "
          f"{sum(len(shown[u]) for u in users)} shown events (full scale)")

    ev = {}
    for u in users:
        e = [(t, c, 1) for (t, c) in clicks.get(u, [])]
        e += [(t, c, 0) for (t, c) in shown.get(u, [])]
        e.sort()
        ev[u] = e

    cats, ys, uidx = [], [], []
    for k, u in enumerate(users):
        cs, yy = dev[u]
        cats.append(cs.astype(int)); ys.append(yy.astype(int))
        uidx.append(np.full(len(cs), k))
    cats = np.concatenate(cats); ys = np.concatenate(ys)
    uidx = np.concatenate(uidx)
    keep = np.where(np.bincount(cats, minlength=18) >= 500)[0]
    m = np.isin(cats, keep)
    cats, ys, uidx = cats[m], ys[m], uidx[m]
    remap = {int(c): j for j, c in enumerate(keep)}
    R = len(ys)
    onehot = sp.csr_matrix((np.ones(R), (np.arange(R),
                            [remap[c] for c in cats])), shape=(R, len(keep)))

    def maxll(aligns):
        X = sp.hstack([onehot, sp.csr_matrix(aligns[:, None])], format="csr")
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000,
                                 fit_intercept=False, tol=1e-7)
        clf.fit(X, ys)
        p = 1 / (1 + np.exp(-(X @ clf.coef_[0])))
        eps = 1e-12
        return float(np.sum(ys * np.log(p + eps)
                            + (1 - ys) * np.log(1 - p + eps)))

    out = {"rhos": RHOS, "n_users": len(users)}
    for spec, grid in (("affine", [abs(x) for x in NS_GRID]),
                       ("prop", DELTA_GRID)):
        prof = {}
        best = (-1e18, None)
        for rho in RHOS:
            for mag in grid:
                Z = np.stack([z_final_correct(hist[u], ev[u], rho, mag, spec)
                              for u in users])
                ll = maxll(Z[uidx, cats])
                prof[f"rho={rho}|mag={mag}"] = ll
                if ll > best[0]:
                    best = (ll, (rho, mag))
                print(f"{spec} rho={rho} mag={mag}: {ll:.1f}", flush=True)
        rho_h, mag_h = best[1]
        ll0 = prof[f"rho={rho_h}|mag=0.0"]
        out[spec] = {"profile": prof, "rho_hat": rho_h, "mag_hat": mag_h,
                     "LR_mag_vs_zero": 2 * (best[0] - ll0)}
    with open(os.path.join(OUT, "p20_backfire_joint.json"), "w") as f:
        json.dump(out, f, indent=2)
    print({s: (out[s]["rho_hat"], out[s]["mag_hat"],
               round(out[s]["LR_mag_vs_zero"], 1)) for s in ("affine", "prop")})


if __name__ == "__main__":
    main()
