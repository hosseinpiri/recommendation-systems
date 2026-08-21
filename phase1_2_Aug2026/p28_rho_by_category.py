"""
Is rho similar across categories? Per-category transition step: clicking
an item of category c moves the state with step rho_c. For each of the 8
largest categories, profile rho_c over a grid holding all other categories
at the global 0.035, refitting (alpha, kappa) on the dev rows each time.
60k-user subsample of the standard temporal design.
"""
import json
import os
import numpy as np

import p1b_transition as T

TARGETS = ["news", "lifestyle", "sports", "finance", "foodanddrink",
           "entertainment", "travel", "health"]
GRID = [0.0, 0.02, 0.035, 0.05, 0.08]
N_USERS = 60000
RHO_BASE = 0.035


def z_final_percat(z0, events, rho_vec):
    z = z0.astype(np.float64).copy()
    for _, c, isclick in events:
        if isclick:
            r = rho_vec[c]
            if r > 0:
                z *= (1 - r)
                z[c] += r
    return z


def main():
    cat_list, hist, clicks, shown, dev, users = T.load()
    rng = np.random.default_rng(29)
    users = sorted(rng.choice(users, N_USERS, replace=False).tolist())
    ev = T.stack_events(users, clicks, shown)
    all_cats = np.concatenate([dev[u][0] for u in users])
    keep_cats = np.where(np.bincount(all_cats, minlength=18) >= 500)[0]
    remap = {int(c): j for j, c in enumerate(keep_cats)}
    D = T.Design(users, hist, ev, dev, keep_cats, remap)

    base = np.full(18, RHO_BASE)
    out = {"grid": GRID, "rho_base": RHO_BASE, "profiles": {}}
    for cname in TARGETS:
        ci = cat_list.index(cname)
        lls = []
        for r in GRID:
            rv = base.copy()
            rv[ci] = r
            Z = np.stack([z_final_percat(hist[u], ev[u], rv) for u in D.users])
            ll, _ = D.maxll(Z[D.uidx, D.cats])
            lls.append(ll)
            print(f"{cname} rho_c={r}: {ll:.1f}", flush=True)
        i = int(np.argmax(lls))
        out["profiles"][cname] = {
            "loglik": lls, "rho_c_hat": GRID[i],
            "LR_vs_base": 2 * (lls[i] - lls[GRID.index(RHO_BASE)])}
        print(f"== {cname}: rho_c_hat={GRID[i]} "
              f"LR_vs_0.035={out['profiles'][cname]['LR_vs_base']:.1f}",
              flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output", "p28_rho_by_category.json"), "w") as f:
        json.dump(out, f, indent=2)
    print({c: v["rho_c_hat"] for c, v in out["profiles"].items()})


if __name__ == "__main__":
    main()
