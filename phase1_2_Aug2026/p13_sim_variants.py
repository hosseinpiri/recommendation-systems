"""
Model-variant simulations in the MIND-calibrated 8+other environment
(kappa = 1.966 from the joint 9-universe estimation, hetero margins,
empirical z0 pool aggregated to 9 dims). Three variants from the Aug 5
meeting, each aware-rollout vs reactive-myopic under CRN:

  terminal  reward is collected only in the last L periods (early periods
            are pure cultivation opportunities)
  distance  acceptance p = sigma(alpha_c - lambda ||z - e_c||^2), intercepts
            recentered so category-level mean click rates match baseline
            (Xinyuan's Section 6.4 form; creates a reachability wall)
  mnl       the platform recommends a slate of 3 categories; the user picks
            via MNL with an outside option calibrated to the baseline click
            rate; the state moves toward the chosen category
"""
import itertools
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
H = 30
N_EP = 2000
BATCH = 200
SEEDS = [1, 2, 3]


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


def probs_linear(z):
    return sigmoid(ALPHA[None, :] + KAPPA * z[:, :8])


def step_state(z, a, y, rho=RHO):
    U = len(a)
    upd = (y * rho)[:, None]
    z = (1 - upd) * z
    z[np.arange(U), a] += upd[:, 0] if upd.ndim > 1 else y * rho
    return z


def fluid(z, m, steps, pfun, rho=RHO):
    z = z.copy()
    tot = np.zeros(z.shape[0])
    for _ in range(steps):
        P = pfun(z)
        ev = m[None, :] * P
        a = np.argmax(ev, axis=1)
        p = P[np.arange(len(a)), a]
        tot += ev[np.arange(len(a)), a]
        upd = (p * rho)[:, None]
        z = (1 - upd) * z
        z[np.arange(len(a)), a] += (p * rho)
    return tot


def aware_action(z, m, h_left, pfun, reward_mask=None, rho=RHO):
    """Depth-1 lookahead + fluid rollout; reward_mask (len H) gates margins
    by period for the terminal-payoff variant."""
    U = z.shape[0]
    P = pfun(z)
    Q = np.zeros((U, 8))
    m_now = m if reward_mask is None or reward_mask[0] else np.zeros(8)
    for a in range(8):
        z1 = z * (1 - rho)
        z1 = z1.copy()
        z1[:, a] += rho
        if reward_mask is None:
            v1 = fluid(z1, m, h_left - 1, pfun, rho)
            v0 = fluid(z, m, h_left - 1, pfun, rho)
        else:
            v1 = fluid_masked(z1, m, reward_mask[1:], pfun, rho)
            v0 = fluid_masked(z, m, reward_mask[1:], pfun, rho)
        Q[:, a] = P[:, a] * (m_now[a] + v1) + (1 - P[:, a]) * v0
    return np.argmax(Q, axis=1)


def fluid_masked(z, m, mask, pfun, rho=RHO):
    z = z.copy()
    tot = np.zeros(z.shape[0])
    for t in range(len(mask)):
        P = pfun(z)
        mm = m if mask[t] else np.zeros(8)
        ev_dir = m[None, :] * P
        a = np.argmax(ev_dir, axis=1)          # myopic-toward-value rollout
        p = P[np.arange(len(a)), a]
        if mask[t]:
            tot += ev_dir[np.arange(len(a)), a]
        upd = (p * rho)[:, None]
        z = (1 - upd) * z
        z[np.arange(len(a)), a] += (p * rho)
    return tot


def run_variant(name, seed):
    rng = np.random.default_rng(seed)
    pool = load_pool()
    res = {}
    if name == "terminal":
        L = 5
        mask = [t >= H - L for t in range(H)]
        rev = {}
        for pol in ("aware", "myopic"):
            r = np.random.default_rng(seed)
            tot = []
            for _ in range(N_EP // BATCH):
                z = pool[r.integers(0, len(pool), BATCH)].copy()
                rr = np.zeros(BATCH)
                for t in range(H):
                    P = probs_linear(z)
                    if pol == "aware":
                        a = aware_action(z, MARGINS, H - t, probs_linear,
                                         reward_mask=mask[t:])
                    else:
                        a = np.argmax(MARGINS[None, :] * P, axis=1)
                    p = P[np.arange(BATCH), a]
                    y = (r.random(BATCH) < p).astype(float)
                    if mask[t]:
                        rr += MARGINS[a] * y
                    upd = (y * RHO)[:, None]
                    z = (1 - upd) * z
                    z[np.arange(BATCH), a] += y * RHO
                tot.extend(rr.tolist())
            rev[pol] = float(np.mean(tot))
        res = {"aware": rev["aware"], "myopic": rev["myopic"],
               "gap_pct": 100 * (rev["aware"] - rev["myopic"]) / rev["myopic"]}
    elif name.startswith("distance"):
        lam = float(name.split("_")[1])

        def recentered_alpha():
            a = ALPHA.copy()
            for c in range(8):
                target = sigmoid(ALPHA[c] + KAPPA * pool[:, c]).mean()
                d2 = (1 - pool[:, c]) ** 2 + (pool ** 2).sum(1) - pool[:, c] ** 2
                lo, hi = -10, 40
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if sigmoid(mid - lam * d2).mean() > target:
                        hi = mid
                    else:
                        lo = mid
                a[c] = 0.5 * (lo + hi)
            return a

        AD = recentered_alpha()

        def pfun(z):
            d2 = ((z[:, None, :8] - np.eye(9)[None, :8, :8]) ** 2).sum(-1) \
                 + (z[:, None, 8] ** 2)
            return sigmoid(AD[None, :] - lam * d2)

        rev = {}
        for pol in ("aware", "myopic"):
            r = np.random.default_rng(seed)
            tot = []
            for _ in range(N_EP // BATCH):
                z = pool[r.integers(0, len(pool), BATCH)].copy()
                rr = np.zeros(BATCH)
                for t in range(H):
                    P = pfun(z)
                    if pol == "aware":
                        a = aware_action(z, MARGINS, H - t, pfun)
                    else:
                        a = np.argmax(MARGINS[None, :] * P, axis=1)
                    p = P[np.arange(BATCH), a]
                    y = (r.random(BATCH) < p).astype(float)
                    rr += MARGINS[a] * y
                    upd = (y * RHO)[:, None]
                    z = (1 - upd) * z
                    z[np.arange(BATCH), a] += y * RHO
                tot.extend(rr.tolist())
            rev[pol] = float(np.mean(tot))
        res = {"aware": rev["aware"], "myopic": rev["myopic"],
               "gap_pct": 100 * (rev["aware"] - rev["myopic"]) / rev["myopic"]}
    elif name == "mnl":
        SLATES = list(itertools.combinations(range(8), 3))
        # outside option delta calibrated so mean P(click) ~ baseline mean p
        base_p = float(probs_linear(pool).mean())

        def mnl_probs(z, slate):
            u = ALPHA[list(slate)][None, :] + KAPPA * z[:, list(slate)]
            eu = np.exp(u)
            denom = np.exp(DELTA) + eu.sum(axis=1, keepdims=True)
            return eu / denom          # (U, 3) choice probs

        # calibrate DELTA on the pool with a representative myopic slate
        DELTA = 0.0
        for _ in range(40):
            probs = []
            for slate in SLATES[:8]:
                probs.append(mnl_probs(pool[:2000], slate).sum(axis=1).mean())
            cur = float(np.mean(probs))
            DELTA += np.log(cur / base_p) if cur > 0 else 0.1
        rev = {}
        for pol in ("aware", "myopic"):
            r = np.random.default_rng(seed)
            tot = []
            for _ in range(N_EP // BATCH):
                z = pool[r.integers(0, len(pool), BATCH)].copy()
                rr = np.zeros(BATCH)
                for t in range(H):
                    # slate value per user: expected revenue now (+ fluid for aware)
                    best = np.zeros(BATCH, dtype=int)
                    bestv = np.full(BATCH, -1e18)
                    for si, slate in enumerate(SLATES):
                        cp = mnl_probs(z, slate)
                        ev = (cp * MARGINS[list(slate)][None, :]).sum(axis=1)
                        if pol == "aware" and t < H - 1:
                            zm = z.copy()
                            move = (cp[:, :, None]
                                    * (np.eye(9)[list(slate)][None, :, :] - z[:, None, :]))
                            zm = z + RHO * move.sum(axis=1)
                            ev = ev + fluid(np.clip(zm, 0, None), MARGINS,
                                            min(H - t - 1, 10), probs_linear)
                        upd_mask = ev > bestv
                        best[upd_mask] = si
                        bestv[upd_mask] = ev[upd_mask]
                    # realize choice
                    uu = r.random(BATCH)
                    for si in np.unique(best):
                        rows = np.where(best == si)[0]
                        cp = mnl_probs(z[rows], SLATES[si])
                        cum = np.cumsum(cp, axis=1)
                        chosen = (uu[rows][:, None] < cum).argmax(axis=1)
                        nochoice = uu[rows] >= cum[:, -1]
                        for k, row in enumerate(rows):
                            if nochoice[k]:
                                continue
                            c = SLATES[si][chosen[k]]
                            rr[row] += MARGINS[c]
                            z[row] = (1 - RHO) * z[row]
                            z[row, c] += RHO
                    tot_step = None
                tot.extend(rr.tolist())
            rev[pol] = float(np.mean(tot))
        res = {"aware": rev["aware"], "myopic": rev["myopic"],
               "gap_pct": 100 * (rev["aware"] - rev["myopic"]) / rev["myopic"],
               "delta": float(DELTA)}
    return res


def main():
    out = {}
    for name in ("terminal", "distance_5", "distance_15", "mnl"):
        per_seed = [run_variant(name, 1000 + s) for s in SEEDS]
        gaps = [r["gap_pct"] for r in per_seed]
        out[name] = {"gap_pct_mean": float(np.mean(gaps)),
                     "gap_pct_se": float(np.std(gaps, ddof=1) / np.sqrt(len(gaps))),
                     "detail": per_seed[0]}
        print(name, "gap", round(out[name]["gap_pct_mean"], 2), "+-",
              round(out[name]["gap_pct_se"], 2), flush=True)
    with open(os.path.join(OUT, "p13_sim_variants.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved p13_sim_variants.json")


if __name__ == "__main__":
    main()
