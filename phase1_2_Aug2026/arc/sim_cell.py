"""One simulation cell. The cell table crosses:

  family 'main'   : margins {uniform, hetero} x rho {6} x kappa {1.751, 2.141}
  family 'gate'   : kappa {1.751, 3, 5, 7} x recenter {raw, iso} x rho {0.035, 0.15}, hetero
  family 'back'   : kappa {1.751, 2.141, 5} x rho_ns {0, -0.002, -0.015},
                    tenure-dependent rho, empirical tenure, hetero

Each cell runs SEEDS_PER_CELL paired (CRN) aware/myopic comparisons with
per-user paired differences for equivalence bounds.
"""
import json
import os
import sys
import numpy as np

import sim_core as S

H = 30
N_EP = 3000
SEEDS_PER_CELL = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
RHOS = [0.0, 0.035, 0.05, 0.15, 0.25, 0.35]


def cells():
    out = []
    for mtag in ("uniform", "hetero"):
        for kap in (1.751, 2.141):
            for rho in RHOS:
                out.append(("main", mtag, kap, rho, None, None))
    for kap in (1.751, 3.0, 5.0, 7.0):
        for rec in ("raw", "iso"):
            for rho in (0.035, 0.15):
                out.append(("gate", "hetero", kap, rho, rec, None))
    for kap in (1.751, 2.141, 5.0):
        for rns in (0.0, -0.002, -0.015):
            out.append(("back", "hetero", kap, None, None, rns))
    return out


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
    fam, mtag, kap, rho, rec, rns = cells()[tid]
    pool, tenure = S.load_pool()
    alpha = S.ALPHA8.copy()
    m = (np.ones(8) if mtag == "uniform"
         else np.array([S.MARGIN_MAP[c] for c in S.ITEMS8]))
    if fam == "gate" and rec == "iso":
        alpha = S.recenter_alpha(alpha, kap, 1.751, pool)

    runs = []
    for s in SEEDS_PER_CELL:
        if fam == "back":
            r = S.run_paired(alpha, kap, m, pool, tenure, 9000 + s, N_EP, H,
                             rho_const=None, rho_ns=rns,
                             tenure_mode="empirical")
        else:
            r = S.run_paired(alpha, kap, m, pool, tenure,
                             1000 * s + int((rho or 0) * 100), N_EP, H,
                             rho_const=rho, rho_ns=0.0)
        runs.append(r)

    d_all = np.array([r["d_mean"] for r in runs])
    n_tot = sum(r["n"] for r in runs)
    # pooled paired stats across seeds
    d_mean = float(np.mean(d_all))
    d_se = float(np.std(d_all, ddof=1) / np.sqrt(len(runs)))
    myo = float(np.mean([r["myopic"] for r in runs]))
    res = {"cell": tid, "family": fam, "margins": mtag, "kappa": kap,
           "rho": rho, "recenter": rec, "rho_ns": rns, "H": H,
           "n_ep_per_seed": N_EP, "seeds": len(runs),
           "aware": float(np.mean([r["aware"] for r in runs])),
           "myopic": myo, "gap_pct": 100 * d_mean / myo,
           "gap_pct_se": 100 * d_se / myo,
           "gap_pct_ci95_hi": 100 * (d_mean + 1.645 * d_se) / myo,
           "n_total": n_tot}
    od = os.path.join(S.OUT, "simcells")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f"cell_{tid:04d}.json"), "w") as f:
        json.dump(res, f)
    print(res, flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "count":
        print(len(cells()))
    else:
        main()
