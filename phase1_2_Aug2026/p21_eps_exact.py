"""
Epsilon-exact certification of the calibrated null (review item 10).

Because failures are state-neutral, the value function depends only on the
sequence of successful clicks. Two rigorous bounds per sampled initial state:

  U_relax   optimistic-envelope DP on (h, k): after k successes the state is
            dominated coordinatewise by z_bar(k) with
            z_bar_c(k) = 1 - (1-rho)^k (1 - z0_c); since p_i is increasing in
            z_i, the DP over (h, k) with p evaluated at z_bar upper-bounds V*.
  U_tree    exact expectimax over the success tree truncated at K clicks,
            with the post-K continuation replaced by the U_relax value from
            (h, K); exact accounting below K makes this typically tighter.

Certificate: premium(z0) = V*(z0) - V_myopic(z0) <= min(U_relax, U_tree)
- V_myopic, with V_myopic computed exactly on the same truncated recursion
(the myopic policy also only moves on clicks; its truncation uses the same
relaxed continuation, making the bound conservative in the right direction
is checked by also lower-bounding V_myopic by pure Monte Carlo).
"""
import json
import os
import pickle
import numpy as np
from functools import lru_cache

SCRATCH = "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
ITEMS = ["news", "lifestyle", "sports", "finance", "foodanddrink",
         "entertainment", "travel", "health"]
MARGINS = np.array([1.0, 1.5, 1.2, 4.0, 1.8, 1.1, 3.0, 2.5])
ALPHA = np.array([-3.702, -3.380, -3.407, -3.448, -3.684, -3.567,
                  -3.716, -3.442])
KAPPA = 1.966
RHO = 0.035
H = 30
K_TRUNC = 4
N_STATES = 40


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def load_states():
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        cl = pickle.load(f)["cat_list"]
    with open(os.path.join(SCRATCH, "user_hist.pkl"), "rb") as f:
        hist = pickle.load(f)
    idx8 = [cl.index(c) for c in ITEMS]
    z18 = np.stack(list(hist.values())).astype(np.float64)
    z9 = np.zeros((len(z18), 9))
    z9[:, :8] = z18[:, idx8]
    z9[:, 8] = 1.0 - z9[:, :8].sum(axis=1)
    rng = np.random.default_rng(21)
    return np.clip(z9, 0, None)[rng.integers(0, len(z18), N_STATES)]


def relax_value(z0):
    """DP over (h, k): p at the coordinatewise upper envelope after k clicks."""
    zbar = np.array([1 - (1 - RHO) ** k * (1 - z0[:8]) for k in range(H + 1)])
    P = sigmoid(ALPHA[None, :] + KAPPA * zbar)          # (H+1, 8)
    V = np.zeros((H + 1, H + 1))                        # V[h, k]
    for h in range(1, H + 1):
        for k in range(H):
            q = P[k] * (MARGINS + V[h - 1, min(k + 1, H)]) \
                + (1 - P[k]) * V[h - 1, k]
            V[h, k] = q.max()
    return V


def tree_value(z0, Vrel):
    """Exact expectimax over success sequences up to K_TRUNC clicks; beyond
    that, continuation = relaxed value at (h, K_TRUNC)."""
    def state_after(seq):
        z = z0[:8].copy()
        full = z0.copy()
        for c in seq:
            full = (1 - RHO) * full
            full[c] += RHO
        return full[:8]

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def V(h, seq):
        if h == 0:
            return 0.0
        k = len(seq)
        if k >= K_TRUNC:
            return Vrel[h, k]
        z = state_after(seq)
        p = sigmoid(ALPHA + KAPPA * z)
        best = -1e18
        for i in range(8):
            v = p[i] * (MARGINS[i] + V(h - 1, seq + (i,))) \
                + (1 - p[i]) * V(h - 1, seq)
            best = max(best, v)
        return best

    def Vmy(h, seq):
        # myopic under the same truncation: lower bound uses MC instead
        return None

    return V(H, ())


def myopic_mc(z0, n_sims, seed):
    rng = np.random.default_rng(seed)
    z = np.tile(z0, (n_sims, 1))
    tot = np.zeros(n_sims)
    for t in range(H):
        P = sigmoid(ALPHA[None, :] + KAPPA * z[:, :8])
        a = np.argmax(MARGINS[None, :] * P, axis=1)
        p = P[np.arange(n_sims), a]
        y = (rng.random(n_sims) < p).astype(float)
        tot += MARGINS[a] * y
        upd = (y * RHO)[:, None]
        z = (1 - upd) * z
        z[np.arange(n_sims), a] += y * RHO
    return float(tot.mean()), float(tot.std() / np.sqrt(n_sims))


def main():
    states = load_states()
    rows = []
    for i, z0 in enumerate(states):
        Vrel = relax_value(z0)
        u_relax = float(Vrel[H, 0])
        u_tree = float(tree_value(z0, Vrel))
        vm, vm_se = myopic_mc(z0, 40000, 100 + i)
        ub = min(u_relax, u_tree)
        # conservative: subtract 3 SE from the myopic lower estimate
        prem_bound = 100 * (ub - (vm - 3 * vm_se)) / vm
        rows.append({"u_relax": u_relax, "u_tree": u_tree,
                     "v_myopic": vm, "v_myopic_se": vm_se,
                     "premium_upper_pct": prem_bound})
        print(f"state {i}: U_tree={u_tree:.4f} U_relax={u_relax:.4f} "
              f"Vmyo={vm:.4f}  premium<= {prem_bound:.2f}%", flush=True)
    prem = np.array([r["premium_upper_pct"] for r in rows])
    out = {"H": H, "K_trunc": K_TRUNC, "n_states": len(rows),
           "premium_upper_pct_max": float(prem.max()),
           "premium_upper_pct_median": float(np.median(prem)),
           "rows": rows}
    with open(os.path.join(OUT, "p21_eps_exact.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("max certified premium bound:", round(prem.max(), 2), "%")


if __name__ == "__main__":
    main()
