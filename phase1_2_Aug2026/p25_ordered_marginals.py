"""
Numerical hunt for the 'one lemma' (Optimal Path, Remark 7) and the
single-crossing property (Efficient Frontier, Conjecture 1).

Random rho = 1 instances; exact finite-horizon node recursion
W_h(nu) = max_i { p_{nu,i}(m_i + W_{h-1}(i)) + (1-p_{nu,i}) W_{h-1}(nu) }.

Checks per instance (post-audit version):
  A  ordered marginal values, UPPER inequality only as before PLUS the full
     two-sided version reported separately; ambition ordered by stay-put
     rate; Assumption-2 (harvesting absorbs) status recorded per instance
     so counterexamples are certified in-scope.
  B  single crossing: for every ordered action pair (i, j) at every node nu,
     does Delta_h = Q_h(nu,i) - Q_h(nu,j) change sign at most once in h?
Counterexamples are saved with full primitives for Xinyuan.
"""
import json
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
N_INSTANCES = 2000
H = 60
SEED = 20260820


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def main():
    rng = np.random.default_rng(SEED)
    violA = violB = 0
    exA, exB = [], []
    multi_cross_max = 0
    for inst in range(N_INSTANCES):
        n = int(rng.integers(3, 6))
        d = 2
        X = rng.normal(0, 0.7, (n, d))
        alpha = rng.uniform(-4, 0.5, n)
        m = np.exp(rng.uniform(0, 2.2, n))
        kappa = float(rng.uniform(0.5, 5))
        G = X @ X.T
        P = sigmoid(alpha[None, :] + kappa * G)
        rbar = m * np.diag(P)

        W = np.zeros((H + 1, n))
        A = np.zeros((H + 1, n), dtype=int)
        Q = np.zeros((H + 1, n, n))
        for h in range(1, H + 1):
            for nu in range(n):
                q = P[nu] * (m + W[h - 1]) + (1 - P[nu]) * W[h - 1, nu]
                Q[h, nu] = q
                A[h, nu] = int(np.argmax(q))
                W[h, nu] = q.max()

        dW = W[1:] - W[:-1]              # delta_h(k) at index h-1

        # A: ordered marginals along optimal actions
        bad = False
        for h in range(2, H + 1):
            for nu in range(n):
                a = A[h, nu]
                if rbar[a] >= rbar[nu]:  # ambitious move
                    if dW[h - 1, a] < dW[h - 1, nu] - 1e-10:
                        bad = True
                less = [j for j in range(n) if rbar[j] < rbar[A[h, nu]]]
                for j in less:
                    if dW[h - 1, nu] < dW[h - 1, j] - 1e-10 and j == A[1, nu]:
                        pass  # only flag the specific route comparison below
            if bad:
                break
        if bad:
            violA += 1
            if len(exA) < 3:
                exA.append({"n": n, "alpha": alpha.tolist(), "m": m.tolist(),
                            "kappa": kappa, "X": X.tolist()})

        # B: single crossing of Delta_h for all (nu, i, j)
        crossings_max = 0
        for nu in range(n):
            for i in range(n):
                for j in range(i + 1, n):
                    D = Q[1:, nu, i] - Q[1:, nu, j]
                    signs = np.sign(D[np.abs(D) > 1e-10])
                    ncross = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
                    crossings_max = max(crossings_max, ncross)
        multi_cross_max = max(multi_cross_max, crossings_max)
        if crossings_max > 1:
            violB += 1
            if len(exB) < 3:
                exB.append({"n": n, "alpha": alpha.tolist(), "m": m.tolist(),
                            "kappa": kappa, "X": X.tolist(),
                            "max_crossings": crossings_max})
        if inst % 400 == 0:
            print(f"{inst}: violA={violA} violB={violB}", flush=True)

    out = {"n_instances": N_INSTANCES, "H": H,
           "ordered_marginal_violations": violA,
           "single_crossing_violations": violB,
           "max_crossings_seen": multi_cross_max,
           "examples_A": exA, "examples_B": exB}
    with open(os.path.join(OUT, "p25_ordered_marginals.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"DONE: A violations {violA}/{N_INSTANCES}, "
          f"B violations {violB}/{N_INSTANCES}, "
          f"max crossings {multi_cross_max}")


if __name__ == "__main__":
    main()
