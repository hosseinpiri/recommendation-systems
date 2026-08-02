"""Refine the rho profile on a 0.005-step grid near the peak to get a proper
95% profile-LR CI; merges the result into p1b_transition.json."""
import json
import os
import numpy as np

import p1b_transition as T

FINE = np.round(np.arange(0.005, 0.085, 0.005), 4)


def main():
    cat_list, hist, clicks, shown, dev, users = T.load()
    ev = T.stack_events(users, clicks, shown)
    all_cats = np.concatenate([dev[u][0] for u in users])
    cnt = np.bincount(all_cats, minlength=len(cat_list))
    keep_cats = np.where(cnt >= 500)[0]
    remap = {int(c): j for j, c in enumerate(keep_cats)}
    D = T.Design(users, hist, ev, dev, keep_cats, remap)

    with open(os.path.join(T.OUT, "p1b_transition.json")) as f:
        out = json.load(f)
    grid = {round(r, 4): ll for r, ll in zip(out["rho_grid"], out["profile_loglik"])}

    for rho in FINE:
        r = round(float(rho), 4)
        if r in grid:
            continue
        ll, kap = D.maxll(D.aligns(r, 0.0))
        grid[r] = ll
        print(f"rho={r:.3f}  logL={ll:.1f}  kappa={kap:.3f}", flush=True)

    rhos = np.array(sorted(grid))
    lls = np.array([grid[r] for r in rhos])
    i = int(np.argmax(lls))
    rho_hat = float(rhos[i])
    ll_hat = lls[i]
    inside = rhos[lls >= ll_hat - 1.92]
    out.update({
        "rho_grid": rhos.tolist(),
        "profile_loglik": lls.tolist(),
        "rho_hat": rho_hat,
        "rho_ci95_profileLR": [float(inside.min()), float(inside.max())],
        "LR_rho_vs_zero": float(2 * (ll_hat - grid[0.0])),
    })
    with open(os.path.join(T.OUT, "p1b_transition.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("rho_hat", rho_hat, "CI", out["rho_ci95_profileLR"],
          "LR", out["LR_rho_vs_zero"])


if __name__ == "__main__":
    main()
