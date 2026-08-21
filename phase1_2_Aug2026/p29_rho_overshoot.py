"""
rho > 1 (overshoot) on MIND: extend the transition profile past full
adoption. In category space the update z+ = (1-rho) z + rho e_c leaves the
simplex for rho > 1, so we estimate the projected transition (per-step
clip to nonnegative, renormalize), under which the state overshoots toward
the clicked vertex and saturates. Profile over rho in {0.5, 0.8, 1.0,
1.2, 1.5, 1.8} plus the calibrated 0.035 as reference, 60k users,
standard temporal design. Also stores the exact-DP toy overshoot sweep
for the memo.
"""
import json
import os
import numpy as np

import p1b_transition as T
import p10_xinyuan_toy as X

GRID = [0.035, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8]
N_USERS = 60000


def z_final_proj(z0, events, rho):
    z = z0.astype(np.float64).copy()
    for _, c, isclick in events:
        if isclick and rho > 0:
            z *= (1 - rho)
            z[c] += rho
            if rho > 1:
                np.clip(z, 0.0, None, out=z)
                s = z.sum()
                if s > 0:
                    z /= s
    return z


def main():
    cat_list, hist, clicks, shown, dev, users = T.load()
    rng = np.random.default_rng(31)
    users = sorted(rng.choice(users, N_USERS, replace=False).tolist())
    ev = T.stack_events(users, clicks, shown)
    all_cats = np.concatenate([dev[u][0] for u in users])
    keep_cats = np.where(np.bincount(all_cats, minlength=18) >= 500)[0]
    remap = {int(c): j for j, c in enumerate(keep_cats)}
    D = T.Design(users, hist, ev, dev, keep_cats, remap)

    out = {"grid": GRID, "loglik": [], "kappa": []}
    for rho in GRID:
        Z = np.stack([z_final_proj(hist[u], ev[u], rho) for u in D.users])
        ll, kap = D.maxll(Z[D.uidx, D.cats])
        out["loglik"].append(ll)
        out["kappa"].append(kap)
        print(f"rho={rho}: logL={ll:.1f} kappa={kap:.3f}", flush=True)
    i = int(np.argmax(out["loglik"]))
    out["rho_hat_on_grid"] = GRID[i]
    ll035 = out["loglik"][0]
    out["LR_best_overshoot_vs_035"] = 2 * (max(out["loglik"][1:]) - ll035)
    out["LL_drop_at_1.5_vs_035"] = ll035 - out["loglik"][GRID.index(1.5)]

    # toy overshoot sweep for memo macros
    toy = {}
    for rho in (0.45, 1.0, 1.2, 1.5, 1.8, 1.95):
        p, succ, fail = X.make_model(kappa=3.0, rho=rho, mode="linear")
        v, vm, actions = X.exact_values(p, succ, fail, X.H, X.Z0)
        toy[str(rho)] = {"premium_pct": round(100 * (v - vm) / vm, 1),
                         "path": "-".join(actions)}
    out["toy_overshoot"] = toy
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "output", "p29_rho_overshoot.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("best on grid:", GRID[i], "| drop at 1.5:",
          round(out["LL_drop_at_1.5_vs_035"], 1))


if __name__ == "__main__":
    main()
