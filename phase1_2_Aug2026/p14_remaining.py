"""
Remaining meeting items, three experiments:

  infinite   discounted long-horizon comparison (beta = 0.97, H = 150 as the
             truncation) at calibrated parameters, aware rollout vs myopic
  target     manufactured-target sweep in the calibrated environment: one
             category (finance) gets intercept shift da in {0,-1,-2,-3} and
             margin multiplier mm in {1,2,4,8}; paired gap at (kappa_hat,
             rho_hat) and at rho = 0.15
  showcase   exact-DP sensitivity grids in Xinyuan's 5-item toy: premium and
             bridge-phase length over kappa in {1,2,3,4,5} x mT in {4,8,12}
             and kappa x alphaT in {-2,-1,-0.75,0}
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
RHO = 0.035


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


def fluid_disc(z, alpha, m, steps, beta, rho):
    z = z.copy()
    tot = np.zeros(z.shape[0])
    disc = 1.0
    for _ in range(steps):
        P = sigmoid(alpha[None, :] + KAPPA * z[:, :8])
        ev = m[None, :] * P
        a = np.argmax(ev, axis=1)
        p = P[np.arange(len(a)), a]
        tot += disc * ev[np.arange(len(a)), a]
        upd = (p * rho)[:, None]
        z = (1 - upd) * z
        z[np.arange(len(a)), a] += p * rho
        disc *= beta
    return tot


def paired_gap(alpha, m, rho, pool, seed, n_ep=1500, H=30, beta=1.0,
               lookahead=True):
    revs = {}
    batch = 150
    for pol in ("aware", "myopic"):
        rng = np.random.default_rng(seed)
        tot = []
        for _ in range(n_ep // batch):
            z = pool[rng.integers(0, len(pool), batch)].copy()
            rr = np.zeros(batch)
            disc = 1.0
            for t in range(H):
                P = sigmoid(alpha[None, :] + KAPPA * z[:, :8])
                if pol == "aware" and lookahead and t < H - 1:
                    Q = np.zeros((batch, 8))
                    steps = min(H - t - 1, 40)
                    v0 = fluid_disc(z, alpha, m, steps, beta, rho)
                    for a in range(8):
                        z1 = z * (1 - rho)
                        z1 = z1.copy()
                        z1[:, a] += rho
                        v1 = fluid_disc(z1, alpha, m, steps, beta, rho)
                        Q[:, a] = P[:, a] * (m[a] + beta * v1) \
                            + (1 - P[:, a]) * beta * v0
                    a = np.argmax(Q, axis=1)
                else:
                    a = np.argmax(m[None, :] * P, axis=1)
                p = P[np.arange(batch), a]
                y = (rng.random(batch) < p).astype(float)
                rr += disc * m[a] * y
                upd = (y * rho)[:, None]
                z = (1 - upd) * z
                z[np.arange(batch), a] += y * rho
                disc *= beta
            tot.extend(rr.tolist())
        revs[pol] = float(np.mean(tot))
    return 100 * (revs["aware"] - revs["myopic"]) / revs["myopic"], revs


def showcase_grid():
    X = np.array([[1.00, 0.25], [0.45, 0.60], [-0.20, 0.35],
                  [-1.00, 0.00], [0.20, -0.30]])
    A0 = np.array([-0.75, 0.2, 0.2, 0.5, 0.0])
    M0 = np.array([12.0, 3.0, 1.5, 2.5, 4.0])
    Z0 = np.array([-1.0, 0.0])
    H = 7
    RHO_T = 0.45

    def solve(kap, alpha, m):
        memo = {}

        def p(i, z):
            return float(sigmoid(alpha[i] + kap * (z @ X[i])))

        def succ(z, i):
            return (1 - RHO_T) * z + RHO_T * X[i]

        def V(h, z):
            if h == 0:
                return 0.0
            key = (h,) + tuple(np.round(z, 9))
            if key in memo:
                return memo[key]
            best = max(p(i, z) * (m[i] + V(h - 1, succ(z, i)))
                       + (1 - p(i, z)) * V(h - 1, z) for i in range(5))
            memo[key] = best
            return best

        def myo(z):
            return int(np.argmax([p(i, z) * m[i] for i in range(5)]))

        memo_m = {}

        def Vm(h, z):
            if h == 0:
                return 0.0
            key = (h,) + tuple(np.round(z, 9))
            if key in memo_m:
                return memo_m[key]
            i = myo(z)
            v = p(i, z) * (m[i] + Vm(h - 1, succ(z, i))) \
                + (1 - p(i, z)) * Vm(h - 1, z)
            memo_m[key] = v
            return v

        z = Z0.copy()
        bridge_len = 0
        for h in range(H, 0, -1):
            scores = [p(i, z) * (m[i] + V(h - 1, succ(z, i)) - V(h - 1, z))
                      for i in range(5)]
            i = int(np.argmax(scores))
            if i != 0:
                bridge_len += 1
            else:
                break
            z = succ(z, i)
        vstar, vmyo = V(H, Z0), Vm(H, Z0)
        return 100 * (vstar - vmyo) / vmyo, bridge_len

    grid_m = {}
    for kap in (1.0, 2.0, 3.0, 4.0, 5.0):
        for mT in (4.0, 8.0, 12.0):
            m = M0.copy(); m[0] = mT
            prem, bl = solve(kap, A0, m)
            grid_m[f"kappa={kap}|mT={mT}"] = {"premium_pct": round(prem, 2),
                                              "bridge_phase": bl}
    grid_a = {}
    for kap in (1.0, 2.0, 3.0, 4.0, 5.0):
        for aT in (-2.0, -1.0, -0.75, 0.0):
            a = A0.copy(); a[0] = aT
            prem, bl = solve(kap, a, M0)
            grid_a[f"kappa={kap}|alphaT={aT}"] = {"premium_pct": round(prem, 2),
                                                  "bridge_phase": bl}
    return {"margin_grid": grid_m, "alpha_grid": grid_a}


def main():
    pool = load_pool()
    out = {}

    # ---- infinite horizon (discounted, truncated at H=150)
    gaps = []
    for s in (1, 2, 3):
        g, revs = paired_gap(ALPHA, MARGINS, RHO, pool, 3000 + s,
                             n_ep=900, H=150, beta=0.97)
        gaps.append(g)
    out["infinite_horizon"] = {"beta": 0.97, "H_trunc": 150,
                               "gap_pct_mean": float(np.mean(gaps)),
                               "gap_pct_se": float(np.std(gaps, ddof=1)
                                                   / np.sqrt(len(gaps)))}
    print("infinite:", out["infinite_horizon"], flush=True)

    # ---- manufactured target sweep (finance = index 3)
    tg = {}
    for da in (0.0, -1.0, -2.0, -3.0):
        for mm in (1.0, 2.0, 4.0, 8.0):
            a = ALPHA.copy(); a[3] += da
            m = MARGINS.copy(); m[3] *= mm
            g35, _ = paired_gap(a, m, RHO, pool, 41, n_ep=1200, H=30)
            g15, _ = paired_gap(a, m, 0.15, pool, 41, n_ep=1200, H=30)
            tg[f"da={da}|mm={mm}"] = {"gap_rho_hat": round(g35, 2),
                                      "gap_rho_015": round(g15, 2)}
            print(f"target da={da} mm={mm}: {g35:.2f}% / {g15:.2f}%",
                  flush=True)
    out["manufactured_target"] = tg

    # ---- showcase sensitivity grids
    out["showcase"] = showcase_grid()
    print("showcase grids done", flush=True)

    with open(os.path.join(OUT, "p14_remaining.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved p14_remaining.json")


if __name__ == "__main__":
    main()
