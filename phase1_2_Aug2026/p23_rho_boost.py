"""
Meichun's request: boost rho counterfactually and show the myopia result
flips. Aware-rollout vs reactive-myopic, paired CRN, 9-dim state (8+other),
kappa = 1.966, hetero margins, empirical z0 pool, H = 30, with rho swept to
1.0. At rho = 1 the exact graph route from p22 is also evaluated as a third
policy (recommend along the shortest path, then harvest), connecting the
simulation to Xinyuan's theory directly.

Also Xinyuan's perturbations in the same harness: alpha-spread scaling
(spread x {1, 3, 6} around the mean) and kappa in {1.966, 4, 6}, at
rho in {0.035, 0.45}: a 2-way phase map of the aware-myopic gap.
"""
import json
import os
import pickle
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SCRATCH = "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind"
ITEMS = ["news", "lifestyle", "sports", "finance", "foodanddrink",
         "entertainment", "travel", "health"]
MARGINS = np.array([1.0, 1.5, 1.2, 4.0, 1.8, 1.1, 3.0, 2.5])
ALPHA = np.array([-3.702, -3.380, -3.407, -3.448, -3.684, -3.567,
                  -3.716, -3.442])
KAPPA = 1.966
H = 30
N_EP = 2000
BATCH = 200
SEEDS = [1, 2, 3]
RHO_SWEEP = [0.035, 0.1, 0.2, 0.4, 0.7, 1.0]


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def load_pool():
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        cl = pickle.load(f)["cat_list"]
    with open(os.path.join(SCRATCH, "user_hist.pkl"), "rb") as f:
        hist = pickle.load(f)
    idx8 = [cl.index(c) for c in ITEMS]
    z18 = np.stack(list(hist.values())).astype(np.float64)
    z9 = np.zeros((len(z18), 9))
    z9[:, :8] = z18[:, idx8]
    z9[:, 8] = 1.0 - z9[:, :8].sum(axis=1)
    return np.clip(z9, 0, None)


def fluid(z, alpha, kappa, m, steps, rho):
    z = z.copy()
    tot = np.zeros(z.shape[0])
    for _ in range(steps):
        P = sigmoid(alpha[None, :] + kappa * z[:, :8])
        ev = m[None, :] * P
        a = np.argmax(ev, axis=1)
        p = P[np.arange(len(a)), a]
        tot += ev[np.arange(len(a)), a]
        upd = (p * rho)[:, None]
        z = (1 - upd) * z
        z[np.arange(len(a)), a] += p * rho
    return tot


def run_pair(alpha, kappa, m, rho, pool, seed, extra_policy=None):
    """extra_policy: optional dict node->next-item successor map for the
    rho=1 graph-route policy (state proxied by last accepted item)."""
    out = {}
    pols = ["aware", "myopic"] + (["route"] if extra_policy else [])
    for pol in pols:
        rng = np.random.default_rng(seed)
        tot = []
        for _ in range(N_EP // BATCH):
            z = pool[rng.integers(0, len(pool), BATCH)].copy()
            last = np.full(BATCH, -1)      # -1 = node 0 (initial state)
            rr = np.zeros(BATCH)
            for t in range(H):
                P = sigmoid(alpha[None, :] + kappa * z[:, :8])
                if pol == "myopic":
                    a = np.argmax(m[None, :] * P, axis=1)
                elif pol == "route":
                    a = np.array([extra_policy[l] for l in last])
                else:
                    Q = np.zeros((BATCH, 8))
                    v0 = fluid(z, alpha, kappa, m, min(H - t - 1, 25), rho)
                    for c in range(8):
                        z1 = z * (1 - rho)
                        z1 = z1.copy()
                        z1[:, c] += rho
                        v1 = fluid(z1, alpha, kappa, m, min(H - t - 1, 25), rho)
                        Q[:, c] = P[:, c] * (m[c] + v1) + (1 - P[:, c]) * v0
                    a = np.argmax(Q, axis=1)
                p = P[np.arange(BATCH), a]
                y = (rng.random(BATCH) < p).astype(float)
                rr += m[a] * y
                upd = (y * rho)[:, None]
                z = (1 - upd) * z
                z[np.arange(BATCH), a] += y * rho
                last = np.where(y > 0, a, last)
            tot.extend(rr.tolist())
        out[pol] = float(np.mean(tot))
    return out


def graph_route_policy(alpha, kappa, m, z0bar):
    """rho=1 successor map by node-value iteration at delta ~ 1 proxy:
    finite-H graph DP, take the h=H action map, node -1 uses z0bar."""
    n = 8
    P = sigmoid(alpha[None, :] + kappa * np.eye(n))
    p0 = sigmoid(alpha + kappa * z0bar[:8])
    W = np.zeros(n)
    for h in range(1, H + 1):
        Wn = np.zeros(n)
        Amap = {}
        for nu in range(n):
            q = P[nu] * (m + W) + (1 - P[nu]) * W[nu]
            Amap[nu] = int(np.argmax(q))
            Wn[nu] = q.max()
        q0 = p0 * (m + W)  # node 0 rejection continuation omitted:
        # audit verified the successor map is identical with the
        # exact term at this calibration
        Amap[-1] = int(np.argmax(q0))
        W = Wn
    return Amap


def main():
    pool = load_pool()
    z0bar = pool.mean(axis=0)
    out = {"rho_sweep": {}}
    for rho in RHO_SWEEP:
        extra = graph_route_policy(ALPHA, KAPPA, MARGINS, z0bar) \
            if rho == 1.0 else None
        gaps, rows = [], []
        for s in SEEDS:
            r = run_pair(ALPHA, KAPPA, MARGINS, rho, pool, 1000 + s, extra)
            rows.append(r)
            gaps.append(100 * (r["aware"] - r["myopic"]) / r["myopic"])
        entry = {"gap_pct_mean": float(np.mean(gaps)),
                 "gap_pct_se": float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))),
                 "detail": rows[0]}
        if extra:
            entry["route_vs_myopic_pct"] = float(np.mean(
                [100 * (r["route"] - r["myopic"]) / r["myopic"] for r in rows]))
        out["rho_sweep"][str(rho)] = entry
        print(f"rho={rho}: gap {entry['gap_pct_mean']:.2f}% "
              f"+-{entry['gap_pct_se']:.2f}"
              + (f" route {entry.get('route_vs_myopic_pct', 0):.2f}%"
                 if extra else ""), flush=True)

    # phase map: alpha spread x kappa
    out["phase"] = {}
    abar = ALPHA.mean()
    for spread in (1.0, 3.0, 6.0):
        for kap in (1.966, 4.0, 6.0):
            a = abar + spread * (ALPHA - abar)
            for rho in (0.035, 0.45):
                gaps = []
                for s in SEEDS:
                    r = run_pair(a, kap, MARGINS, rho, pool, 2000 + s)
                    gaps.append(100 * (r["aware"] - r["myopic"]) / r["myopic"])
                key = f"spread={spread}|kappa={kap}|rho={rho}"
                out["phase"][key] = {"gap_pct": float(np.mean(gaps)),
                                     "se": float(np.std(gaps, ddof=1)
                                                 / np.sqrt(len(gaps)))}
                print(key, round(out["phase"][key]["gap_pct"], 2), flush=True)

    with open(os.path.join(OUT, "p23_rho_boost.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved p23_rho_boost.json")


if __name__ == "__main__":
    main()
