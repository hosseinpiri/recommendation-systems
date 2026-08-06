"""
Spillover transition estimation on MIND (Xinyuan's fix a, made empirical).

Transition with cross-category spillover:
    z+ = z + rho [ (1-s) I + s S ] (e_c - z)
where S is the column-stochastic category-similarity matrix built from LSA
centroids (zero diagonal). s = 0 recovers the baseline linear pull; s > 0
lets a click on category c also pull taste toward similar categories.
Profile likelihood over (rho, s) with the p1b temporal design.
"""
import json
import os
import numpy as np

import p1b_transition as T

S18 = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output", "spill_S18.npy"))
RHO_GRID = [0.02, 0.035, 0.05]
S_GRID = [0.0, 0.1, 0.2, 0.3, 0.5]


def z_final_spill(z0, events, rho, s):
    z = z0.astype(np.float64).copy()
    M = (1 - s) * np.eye(18) + s * S18
    for _, c, isclick in events:
        if isclick and rho > 0:
            v = -z
            v[c] += 1.0
            z = z + rho * (M @ v)
            np.clip(z, 0.0, None, out=z)
            z /= z.sum()
    return z


def main():
    cat_list, hist, clicks, shown, dev, users = T.load()
    ev = T.stack_events(users, clicks, shown)
    all_cats = np.concatenate([dev[u][0] for u in users])
    cnt = np.bincount(all_cats, minlength=len(cat_list))
    keep_cats = np.where(cnt >= 500)[0]
    remap = {int(c): j for j, c in enumerate(keep_cats)}
    D = T.Design(users, hist, ev, dev, keep_cats, remap)

    out = {"rho_grid": RHO_GRID, "s_grid": S_GRID, "loglik": {}}
    best = (-1e18, None)
    for rho in RHO_GRID:
        for s in S_GRID:
            zmat = np.stack([z_final_spill(hist[u], ev[u], rho, s)
                             for u in D.users])
            align = zmat[D.uidx, D.cats]
            ll, kap = D.maxll(align)
            out["loglik"][f"rho={rho}|s={s}"] = ll
            if ll > best[0]:
                best = (ll, (rho, s, kap))
            print(f"rho={rho} s={s}: logL={ll:.1f} kappa={kap:.3f}", flush=True)
    (rho_h, s_h, kap_h) = best[1]
    ll0 = out["loglik"][f"rho={rho_h}|s=0.0"]
    out.update({"rho_hat": rho_h, "s_hat": s_h, "kappa_at_hat": kap_h,
                "LR_s_vs_zero": 2 * (best[0] - ll0)})
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output", "p11_spillover.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("BEST:", best[1], "LR_s:", round(out["LR_s_vs_zero"], 1))


if __name__ == "__main__":
    main()
