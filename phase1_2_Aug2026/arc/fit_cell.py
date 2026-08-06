"""One profile-grid cell: (universe, rho, kappa_mode) -> max log likelihood.

Cell index comes from SLURM_ARRAY_TASK_ID over the cross product defined here.
"""
import json
import os
import sys
import numpy as np

import fitlib as F

RHO_GRID = sorted(set(
    [round(x, 3) for x in np.arange(0.0, 0.42, 0.02)]
    + [round(x, 3) for x in np.arange(0.005, 0.085, 0.005)]))
UNIVERSES = [18, 9]
KMODES = ["free", "k_train", "k_dev"]
KVALS = {"free": None, "k_train": 1.751, "k_dev": 2.141}
CAP = 120000
SEED = 11


def cells():
    out = []
    for uv in UNIVERSES:
        for km in KMODES:
            for rho in RHO_GRID:
                out.append((uv, km, rho))
    return out


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
    uv, km, rho = cells()[tid]
    nc, states, clicks, dev = F.load_prep()
    map9 = F.cat18_to9(nc)
    users = F.user_sample(dev, CAP, SEED)
    cats18, ecat, ys, uidx, keepc = F.build_rows(dev, users, uv, map9)
    Z = F.z_rolled(states, clicks, users, rho, uv, map9)
    aligns = Z[uidx, ecat] if uv == 9 else Z[uidx, cats18]
    ll, kap = F.max_loglik(ecat, ys, aligns, keepc, KVALS[km])
    res = {"cell": tid, "universe": uv, "kmode": km, "rho": rho,
           "loglik": ll, "kappa": kap, "n_rows": int(len(ys)),
           "n_users": len(users)}
    od = os.path.join(F.OUT, "fitcells")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f"cell_{tid:04d}.json"), "w") as f:
        json.dump(res, f)
    print(res, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "count":
        print(len(cells()))
    else:
        main()
