"""Post-audit fixes to p1b_transition.json:
1. drop the stale misaligned profile_kappa field (pass-1 only);
2. recompute kappa at the refined rho_hat (statsmodels, full 120k design);
3. recompute best_runmean_vs_best_const_rho against the refined optimum;
4. re-estimate the rho_ns profile AT rho = rho_hat (the audit showed the
   2-D coarse grid's rho_ns_hat = -0.01 was a grid artifact), with per-step
   simplex clipping, on the same 40k-user subsample as the original 2-D pass.
"""
import json
import os
import numpy as np
import scipy.sparse as sp

import p1b_transition as T

NS_GRID = [-0.04, -0.03, -0.02, -0.015, -0.01, -0.005, 0.0, 0.01]


def z_final_stepclip(z0, events, rho, rho_ns):
    z = z0.astype(np.float64).copy()
    for _, c, isclick in events:
        if isclick:
            if rho != 0.0:
                z *= (1.0 - rho)
                z[c] += rho
        elif rho_ns != 0.0:
            z *= (1.0 - rho_ns)
            z[c] += rho_ns
            np.clip(z, 0.0, None, out=z)
            s = z.sum()
            if s > 0:
                z /= s
    return z


def main():
    cat_list, hist, clicks, shown, dev, users = T.load()
    ev = T.stack_events(users, clicks, shown)
    all_cats = np.concatenate([dev[u][0] for u in users])
    cnt = np.bincount(all_cats, minlength=len(cat_list))
    keep_cats = np.where(cnt >= 500)[0]
    remap = {int(c): j for j, c in enumerate(keep_cats)}

    with open(os.path.join(T.OUT, "p1b_transition.json")) as f:
        out = json.load(f)
    rho_hat = out["rho_hat"]

    # (2) kappa at refined rho_hat with SE
    D = T.Design(users, hist, ev, dev, keep_cats, remap)
    import statsmodels.api as sm
    al = D.aligns(rho_hat, 0.0)
    X = sp.hstack([D.onehot, sp.csr_matrix(al[:, None])], format="csr").toarray()
    res = sm.Logit(D.y, X).fit(disp=0, maxiter=200)
    out["kappa_at_rho_hat_dev"] = float(res.params[-1])
    out["kappa_at_rho_hat_dev_se"] = float(res.bse[-1])

    # (3) runmean gap vs refined const-rho optimum
    out["best_runmean_vs_best_const_rho"] = float(
        max(out["runmean_loglik"]) - max(out["profile_loglik"]))

    # (1) drop stale field
    out.pop("profile_kappa", None)

    # (4) rho_ns profile at rho_hat, 40k subsample, per-step clipping
    rng = np.random.default_rng(T.SEED + 1)
    users2 = sorted(rng.choice(users, T.MAX_USERS_2D, replace=False).tolist())
    D2 = T.Design(users2, hist, ev, dev, keep_cats, remap)

    def aligns_ns(rns):
        zmat = np.stack([z_final_stepclip(hist[u], ev[u], rho_hat, rns)
                         for u in D2.users])
        return zmat[D2.uidx, D2.cats]

    prof = {}
    for rns in NS_GRID:
        ll, _ = D2.maxll(aligns_ns(float(rns)))
        prof[rns] = ll
        print(f"rho_ns={rns:+.3f}  logL={ll:.1f}", flush=True)
    b = max(prof, key=prof.get)
    out["ns_at_rhohat"] = {
        "rho": rho_hat,
        "rho_ns_grid": NS_GRID,
        "loglik": [prof[r] for r in NS_GRID],
        "rho_ns_hat": float(b),
        "LR_vs_zero": float(2 * (prof[b] - prof[0.0])),
        "note": ("per-step simplex clipping; shown events are a 10% subsample, "
                 "so the per-actual-impression step is ~10x smaller"),
    }

    with open(os.path.join(T.OUT, "p1b_transition.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("kappa at rho_hat:", round(out["kappa_at_rho_hat_dev"], 4),
          "runmean gap:", round(out["best_runmean_vs_best_const_rho"], 2),
          "rho_ns_hat:", b, "LR:", round(out["ns_at_rhohat"]["LR_vs_zero"], 1))


if __name__ == "__main__":
    main()
