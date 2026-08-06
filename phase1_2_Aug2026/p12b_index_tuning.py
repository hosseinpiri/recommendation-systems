"""
Index-policy hyperparameter tuning in the 5-item showcase (restored to a
proper script after the audit found the original existed only inline).

Grid: eta_mode 'h' (eta = h-1, horizon-scaled) or '1' (eta = min(h-1, 1));
beta in {0.3, 1.0}; potential 'a' (best myopic payoff max_j p_j m_j) or
'c' (geometric: negative squared distance to the target item's feature).
Each configuration run over SEEDS and reported as mean pct-of-oracle.
"""
import json
import os
import numpy as np

import p12_index_policy as P

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
X5 = np.array([[1.00, 0.25], [0.45, 0.60], [-0.20, 0.35],
               [-1.00, 0.00], [0.20, -0.30]])
A5 = np.array([-0.75, 0.2, 0.2, 0.5, 0.0])
M5 = np.array([12.0, 3.0, 1.5, 2.5, 4.0])
SEEDS = [5, 6, 7]
K = 400
H = 7
KAPPA, RHO = 3.0, 0.45


def run_index(eta_mode, beta, phi, seed):
    n = 5
    rng = np.random.default_rng(seed)
    L = P.Learner(n)
    _, _, _, scores_true = P.make_dp(X5, A5, M5, KAPPA, RHO)
    orc_rng = np.random.default_rng(seed)
    tot_o = tot = 0.0
    for k in range(K):
        z = np.array([-1.0, 0.0])
        for t in range(H):
            i = int(np.argmax(scores_true(H - t, z)))
            p = P.sigmoid(A5[i] + KAPPA * z @ X5[i])
            y = float(orc_rng.random() < p)
            tot_o += M5[i] * y
            if y:
                z = (1 - RHO) * z + RHO * X5[i]
    for k in range(K):
        z = np.array([-1.0, 0.0])
        ah = L.alpha_hat.copy()
        for t in range(H):
            h = H - t
            ph = np.array([P.sigmoid(ah[j] + KAPPA * z @ X5[j])
                           for j in range(n)])
            Phi_z = (float(np.max(ph * M5)) if phi == "a"
                     else -float(((z - X5[0]) ** 2).sum()))
            eta = (h - 1) if eta_mode == "h" else min(h - 1, 1)
            sc = np.zeros(n)
            for j in range(n):
                zs = (1 - RHO) * z + RHO * X5[j]
                if phi == "a":
                    ps = np.array([P.sigmoid(ah[l] + KAPPA * zs @ X5[l])
                                   for l in range(n)])
                    Phi_s = float(np.max(ps * M5))
                else:
                    Phi_s = -float(((zs - X5[0]) ** 2).sum())
                sc[j] = (ph[j] * M5[j] + eta * ph[j] * (Phi_s - Phi_z)
                         + beta * M5[j] * L.width(j))
            i = int(np.argmax(sc))
            p = P.sigmoid(A5[i] + KAPPA * z @ X5[i])
            y = float(rng.random() < p)
            tot += M5[i] * y
            L.add(i, KAPPA * float(z @ X5[i]), y)
            if y:
                z = (1 - RHO) * z + RHO * X5[i]
        L.refit()
    return 100 * tot / tot_o


def main():
    res = {}
    for eta_mode in ("h", "1"):
        for beta in (0.3, 1.0):
            for phi in ("a", "c"):
                key = f"eta={eta_mode}|beta={beta}|phi={phi}"
                vals = [run_index(eta_mode, beta, phi, s) for s in SEEDS]
                res[key] = {"mean": round(float(np.mean(vals)), 2),
                            "se": round(float(np.std(vals, ddof=1)
                                              / np.sqrt(len(SEEDS))), 2),
                            "per_seed": [round(v, 2) for v in vals]}
                print(key, res[key], flush=True)
    with open(os.path.join(OUT, "p12b_index_tuning.json"), "w") as f:
        json.dump(res, f, indent=2)
    best = max(res, key=lambda k: res[k]["mean"])
    print("BEST:", best, res[best])


if __name__ == "__main__":
    main()
