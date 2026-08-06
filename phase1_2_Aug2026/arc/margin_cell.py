"""One margin-sweep chunk: 100 margin vectors x 2 seeds, paired CRN gaps.
10,000 vectors over 100 array tasks; rho = 0.15, kappa = 1.751."""
import json
import os
import sys
import numpy as np

import sim_core as S

VEC_PER_TASK = 20
SEEDS = [11, 12]
RHO = 0.15
N_EP = 1500
H = 30


def vectors_for(task):
    rng = np.random.default_rng(700000 + task)
    out = []
    sigs = [0.3, 0.6, 1.0, 1.5]
    for i in range(VEC_PER_TASK):
        sig = sigs[i % len(sigs)]
        g = rng.standard_normal(8)
        m = np.exp(sig * g)
        out.append(m / m.mean())
    return out


def main():
    tid = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["SLURM_ARRAY_TASK_ID"])
    pool, tenure = S.load_pool()
    alpha = S.ALPHA8
    rows = []
    for vi, m in enumerate(vectors_for(tid)):
        gaps = []
        for s in SEEDS:
            r = S.run_paired(alpha, 1.751, m, pool, tenure, 5000 + s,
                             N_EP, H, rho_const=RHO, rho_ns=0.0)
            gaps.append(100 * r["d_mean"] / r["myopic"])
        rows.append({"task": tid, "vec": vi, "gap_pct": float(np.mean(gaps)),
                     "gap_se": float(np.std(gaps)),
                     "dispersion": float(np.std(np.log(m))),
                     "corr_alpha": float(np.corrcoef(m, alpha)[0, 1]),
                     "margins": np.round(m, 4).tolist()})
    od = os.path.join(S.OUT, "margincells")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, f"chunk_{tid:04d}.json"), "w") as f:
        json.dump(rows, f)
    print(f"task {tid}: max gap {max(r['gap_pct'] for r in rows):.2f}%",
          flush=True)


if __name__ == "__main__":
    main()
