"""One user-block bootstrap replicate of the rho profile (universe 18).

Resamples users with replacement, rebuilds every state inside the replicate,
refits (alpha, kappa) at each rho on a coarse grid, and records the argmax and
the likelihood-ratio against rho = 0. 300 replicates give a bootstrap CI for
rho_hat that respects user-level dependence.
"""
import json
import os
import sys
import numpy as np

import fitlib as F

GRID = [0.0, 0.02, 0.03, 0.035, 0.04, 0.05, 0.08]
N_USERS = 60000


def main():
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
    nc, states, clicks, dev = F.load_prep()
    map9 = F.cat18_to9(nc)
    base_users = sorted(dev.keys())
    rng = np.random.default_rng(500000 + rep)
    users = list(rng.choice(base_users, N_USERS, replace=True))
    # resampled users can repeat: build row structures on the multiset
    cats, ys, uidx = [], [], []
    for k, u in enumerate(users):
        cs, yy = dev[u]
        cats.append(cs.astype(int)); ys.append(yy.astype(int))
        uidx.append(np.full(len(cs), k))
    cats = np.concatenate(cats); ys = np.concatenate(ys)
    uidx = np.concatenate(uidx)
    cnt = np.bincount(cats, minlength=18)
    keepc = np.where(cnt >= 500)[0]
    m = np.isin(cats, keepc)
    cats, ys, uidx = cats[m], ys[m], uidx[m]

    lls = []
    for rho in GRID:
        Z = F.z_rolled(states, clicks, users, rho, 18, map9)
        ll, _ = F.max_loglik(cats, ys, Z[uidx, cats], keepc, None)
        lls.append(ll)
    i = int(np.argmax(lls))
    res = {"rep": rep, "grid": GRID, "loglik": lls,
           "rho_hat": GRID[i], "LR_vs_zero": 2 * (lls[i] - lls[0])}
    od = os.path.join(F.OUT, "bootcells")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f"rep_{rep:04d}.json"), "w") as f:
        json.dump(res, f)
    print(res["rep"], res["rho_hat"], round(res["LR_vs_zero"], 1), flush=True)


if __name__ == "__main__":
    main()
