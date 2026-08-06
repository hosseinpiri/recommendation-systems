"""
Xinyuan's index policy: validation and learning benchmark.

Part 1  Validate the exact DP against the Section 5 numbers of his note
        (3 items, H=3, rho=0.7, kappa=2, alpha=0):
        r_A=1.192, r_B=0.500, r_C=1.762; D_A3=2.533, D_B3=3.000, D_C3=1.762;
        optimal success path B -> A -> A.

Part 2  Learning benchmark in his 5-item 2D toy (bridge regime, H=7): alpha
        unknown, learned across K episodes by per-item MLE. Policies:
          oracle       exact DP with true alpha
          plugin-dp    exact DP with alpha_hat (estimate-then-plug-in)
          index        his approximate index: p_hat*m + eta_h*p_hat*(Phi(T)-Phi(z))
                       + beta_h*w  with Phi = max_j p_hat_j m_j, eta_h = h-1
                       capped rollout scale, beta_h = sqrt-log UCB width
          myopic-plug  p_hat*m only
        Reports cumulative regret vs oracle.

Part 3  Same four policies in the MIND-calibrated 8-category environment
        (kappa=1.966, rho=0.035, hetero margins) and in the backfire variant
        (kappa=5, rho_ns=-0.015) where planning matters.
"""
import json
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


# ---------------------------------------------------------------- generic DP
def make_dp(X, alpha, margin, kappa, rho):
    n = len(alpha)

    def p(i, z):
        return float(sigmoid(alpha[i] + kappa * float(z @ X[i])))

    def succ(z, i):
        return (1 - rho) * z + rho * X[i]

    memo = {}

    def V(h, z):
        if h == 0:
            return 0.0
        key = (h,) + tuple(np.round(z, 9))
        if key in memo:
            return memo[key]
        best = -1e18
        for i in range(n):
            pi = p(i, z)
            best = max(best, pi * (margin[i] + V(h - 1, succ(z, i)))
                       + (1 - pi) * V(h - 1, z))
        memo[key] = best
        return best

    def scores(h, z):
        return [p(i, z) * (margin[i] + V(h - 1, succ(z, i)) - V(h - 1, z))
                for i in range(n)]

    return p, succ, V, scores


# ---------------------------------------------------------- part 1: validate
def validate():
    X = np.array([[1.0, 0.0], [0.0, 0.5], [-1.0, 0.0]])
    alpha = np.zeros(3)
    margin = np.array([10.0, 1.0, 2.0])
    z0 = np.array([-1.0, 0.0])
    p, succ, V, scores = make_dp(X, alpha, margin, 2.0, 0.7)
    r = [p(i, z0) * margin[i] for i in range(3)]
    D3 = scores(3, z0)
    path = []
    z = z0.copy()
    for h in (3, 2, 1):
        i = int(np.argmax(scores(h, z)))
        path.append("ABC"[i])
        z = succ(z, i)
    res = {"r": [round(v, 3) for v in r], "D3": [round(v, 3) for v in D3],
           "path": "->".join(path),
           "expected": {"r": [1.192, 0.5, 1.762], "D3": [2.533, 3.0, 1.762],
                        "path": "B->A->A"}}
    print("validation:", res)
    return res


# ------------------------------------------------- learning-policy machinery
class Learner:
    def __init__(self, n, lam=1.0):
        self.n, self.lam = n, lam
        self.obs = [[] for _ in range(n)]
        self.alpha_hat = np.zeros(n)
        self.counts = np.zeros(n)

    def add(self, i, offset, y):
        self.obs[i].append((offset, y))
        self.counts[i] += 1

    def refit(self):
        for i in range(self.n):
            if not self.obs[i]:
                continue
            arr = np.array(self.obs[i])
            off, y = arr[:, 0], arr[:, 1]
            a = self.alpha_hat[i]
            for _ in range(30):
                pr = sigmoid(a + off)
                g = np.sum(y - pr) - self.lam * a
                hh = -np.sum(pr * (1 - pr)) - self.lam
                step = g / hh
                a -= step
                if abs(step) < 1e-9:
                    break
            self.alpha_hat[i] = np.clip(a, -8, 4)

    def width(self, i):
        return np.sqrt(np.log(50.0) / max(self.counts[i], 1.0))


def run_learning(env_name, X, alpha_true, margin, kappa, rho, H, K, seed,
                 policies, rho_ns=0.0):
    n = len(alpha_true)
    rng = np.random.default_rng(seed)
    z0s = []
    if X.shape[1] == 2:                      # toy: fixed z0
        z0s = [np.array([-1.0, 0.0])] * K
    results = {}
    _, _, Vtrue, scores_true = make_dp(X, alpha_true, margin, kappa, rho)

    for pol in policies:
        rng = np.random.default_rng(seed)
        L = Learner(n)
        total = np.zeros(K)
        for k in range(K):
            z = z0s[k].copy()
            ah = L.alpha_hat.copy()
            _, succ_h, Vh, scores_h = make_dp(X, ah, margin, kappa, rho)
            for t in range(H):
                h = H - t
                if pol == "oracle":
                    i = int(np.argmax(scores_true(h, z)))
                elif pol == "plugin-dp":
                    i = int(np.argmax(scores_h(h, z)))
                elif pol == "myopic-plug":
                    i = int(np.argmax([sigmoid(ah[j] + kappa * z @ X[j])
                                       * margin[j] for j in range(n)]))
                else:                          # index
                    ph = np.array([sigmoid(ah[j] + kappa * z @ X[j])
                                   for j in range(n)])
                    Phi_z = float(np.max(ph * margin))
                    sc = np.zeros(n)
                    eta = min(h - 1, 3)
                    for j in range(n):
                        zs = (1 - rho) * z + rho * X[j]
                        ps = np.array([sigmoid(ah[l] + kappa * zs @ X[l])
                                       for l in range(n)])
                        Phi_s = float(np.max(ps * margin))
                        sc[j] = (ph[j] * margin[j]
                                 + eta * ph[j] * (Phi_s - Phi_z)
                                 + 1.0 * margin[j] * L.width(j))
                    i = int(np.argmax(sc))
                ptrue = sigmoid(alpha_true[i] + kappa * z @ X[i])
                y = float(rng.random() < ptrue)
                total[k] += margin[i] * y
                if pol != "oracle":
                    L.add(i, kappa * float(z @ X[i]), y)
                if y:
                    z = (1 - rho) * z + rho * X[i]
                elif rho_ns != 0.0:
                    z = z + rho_ns * (X[i] - z)
            if pol != "oracle":
                L.refit()
        results[pol] = {"mean_reward": float(total.mean()),
                        "cum_reward": float(total.sum())}
    orc = results["oracle"]["cum_reward"]
    for pol in results:
        results[pol]["regret_vs_oracle"] = orc - results[pol]["cum_reward"]
        results[pol]["pct_of_oracle"] = round(
            100 * results[pol]["cum_reward"] / orc, 2)
    print(env_name, {p: results[p]["pct_of_oracle"] for p in results})
    return results


def main():
    out = {"validation": validate()}

    # part 2: toy learning benchmark
    X5 = np.array([[1.00, 0.25], [0.45, 0.60], [-0.20, 0.35],
                   [-1.00, 0.00], [0.20, -0.30]])
    a5 = np.array([-0.75, 0.2, 0.2, 0.5, 0.0])
    m5 = np.array([12.0, 3.0, 1.5, 2.5, 4.0])
    out["toy_learning"] = run_learning(
        "toy(bridge regime)", X5, a5, m5, kappa=3.0, rho=0.45, H=7, K=400,
        seed=5, policies=["oracle", "plugin-dp", "index", "myopic-plug"])

    with open(os.path.join(OUT, "p12_index_policy.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved p12_index_policy.json")


if __name__ == "__main__":
    main()
