"""
Simulation of the transition the data actually prefer.

Environment (category one-hot, 8 largest categories, heterogeneous margins,
alpha and kappa calibrated):
  click:    z <- (1 - rho_t) z + rho_t e_a,   rho_t = 1 / (n0 + N_t),
            N_t = clicks so far in the episode, n0 = user tenure (prior mass)
  nonclick: z <- (1 - rho_ns) z + rho_ns e_a  with rho_ns <= 0 (backfire),
            clipped to the simplex per step.

Backfire breaks state-neutrality of failures; tenure concentrates updating
among new users. Scenarios cross rho_ns in {0, -0.002, -0.015} with user
tenure (new n0=5, tenured n0=40, empirical mix) at the calibrated kappa, and
repeat the mix at kappa=5 where the reachability gate is close. Policies:
cultivation-aware rollout (plans through both mechanisms, tracking expected
click counts in the fluid rollout) vs reactive myopic. Common random numbers.
Includes an exact expectimax-DP validation on a small instance, feasible
because backfire trees are shallow (n=3, H=7).
"""
import json
import os
import pickle
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SCRATCH = os.environ.get(
    "SCRATCH",
    "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind",
)
ITEMS = ["news", "lifestyle", "sports", "finance", "foodanddrink",
         "entertainment", "travel", "health"]
MARGINS = {"news": 1.0, "sports": 1.2, "entertainment": 1.1, "lifestyle": 1.5,
           "foodanddrink": 1.8, "health": 2.5, "travel": 3.0, "finance": 4.0}
H = 30
N_EP = 3000
BATCH = 150
SEEDS = [1, 2, 3]


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


class Env:
    def __init__(self, alpha, kappa, rho_ns, n0, z0, rng):
        self.alpha, self.kappa, self.rho_ns = alpha, kappa, rho_ns
        self.z = z0.copy()
        self.n = n0.astype(np.float64).copy()   # effective click mass
        self.rng = rng

    def step(self, a):
        U = self.z.shape[0]
        p = sigmoid(self.alpha[a] + self.kappa * self.z[np.arange(U), a])
        yv = (self.rng.random(U) < p).astype(np.float64)
        rho_t = 1.0 / (self.n + 1.0)
        # click branch
        upd = yv * rho_t
        self.z = (1 - upd[:, None]) * self.z
        self.z[np.arange(U), a] += upd
        self.n += yv
        # nonclick backfire branch
        if self.rho_ns != 0.0:
            nb = (1 - yv) * self.rho_ns
            self.z = (1 - nb[:, None]) * self.z
            self.z[np.arange(U), a] += nb
            np.clip(self.z, 0.0, None, out=self.z)
            self.z /= self.z.sum(axis=1, keepdims=True)
        return yv


def myopic(z, alpha, kappa, m):
    return np.argmax(m[None, :] * sigmoid(alpha[None, :] + kappa * z), axis=1)


def fluid_value(z, n, alpha, kappa, rho_ns, m, steps):
    z = z.copy()
    n = n.copy()
    total = np.zeros(z.shape[0])
    U = z.shape[0]
    for _ in range(steps):
        P = sigmoid(alpha[None, :] + kappa * z)
        ev = m[None, :] * P
        a = np.argmax(ev, axis=1)
        p = P[np.arange(U), a]
        total += ev[np.arange(U), a]
        rho_t = 1.0 / (n + 1.0)
        upd = (p * rho_t)[:, None]
        nb = ((1 - p) * rho_ns)[:, None]
        z = (1 - upd - nb) * z
        z[np.arange(U), a] += (upd + nb)[:, 0]
        np.clip(z, 0.0, None, out=z)
        z /= z.sum(axis=1, keepdims=True)
        n = n + p
    return total


def plan_aware(z, n, alpha, kappa, rho_ns, m, h_left, depth=2):
    U, nn = z.shape
    if h_left <= 1:
        return myopic(z, alpha, kappa, m)
    depth = min(depth, h_left - 1)
    rho_t = 1.0 / (n + 1.0)

    def succ(w, a, rt):
        w1 = (1 - rt[:, None]) * w
        w1[:, a] += rt
        return w1

    def fail(w, a):
        if rho_ns == 0.0:
            return w
        w1 = (1 - rho_ns) * w
        w1[:, a] += rho_ns
        np.clip(w1, 0.0, None, out=w1)
        return w1 / w1.sum(axis=1, keepdims=True)

    def v0(w, nv, h):
        if h <= 0:
            return np.zeros(w.shape[0])
        return fluid_value(w, nv, alpha, kappa, rho_ns, m, h)

    def v1(w, nv, h):
        if h <= 0:
            return np.zeros(w.shape[0])
        P = sigmoid(alpha[None, :] + kappa * w)
        rt = 1.0 / (nv + 1.0)
        best = np.full(w.shape[0], -np.inf)
        for b in range(nn):
            vs = v0(succ(w, b, rt), nv + 1, h - 1)
            vf = v0(fail(w, b), nv, h - 1)
            np.maximum(best, P[:, b] * (m[b] + vs) + (1 - P[:, b]) * vf, out=best)
        return best

    vfun = v0 if depth <= 1 else v1
    Q = np.zeros((U, nn))
    P = sigmoid(alpha[None, :] + kappa * z)
    for a in range(nn):
        vs = vfun(succ(z, a, rho_t), n + 1, h_left - 1)
        vf = vfun(fail(z, a), n, h_left - 1)
        Q[:, a] = P[:, a] * (m[a] + vs) + (1 - P[:, a]) * vf
    return np.argmax(Q, axis=1)


def run(policy, alpha, kappa, rho_ns, m, pool, n0pool, seed):
    rng = np.random.default_rng(seed)
    rev, bridge, recs = [], 0, 0
    for _ in range(N_EP // BATCH):
        pick = rng.integers(0, pool.shape[0], BATCH)
        env = Env(alpha, kappa, rho_ns, n0pool[pick], pool[pick].copy(), rng)
        r = np.zeros(BATCH)
        for t in range(H):
            if policy == "aware":
                a = plan_aware(env.z, env.n, alpha, kappa, rho_ns, m, H - t)
            else:
                a = myopic(env.z, alpha, kappa, m)
            amy = myopic(env.z, alpha, kappa, m)
            bridge += int(np.sum(a != amy))
            recs += BATCH
            r += m[a] * env.step(a)
        rev.extend(r.tolist())
    return float(np.mean(rev)), bridge / recs


def exact_dp(z0, n0, alpha, kappa, rho_ns, m, h):
    nn = len(alpha)

    def rec(z, n, hh):
        if hh == 0:
            return 0.0
        best = -1e18
        for a in range(nn):
            p = float(sigmoid(alpha[a] + kappa * z[a]))
            rt = 1.0 / (n + 1.0)
            zs = (1 - rt) * np.array(z); zs[a] += rt
            zf = np.array(z)
            if rho_ns != 0.0:
                zf = (1 - rho_ns) * zf; zf[a] += rho_ns
                zf = np.clip(zf, 0, None); zf /= zf.sum()
            v = p * (m[a] + rec(tuple(zs), n + 1, hh - 1)) \
                + (1 - p) * rec(tuple(zf), n, hh - 1)
            best = max(best, v)
        return best

    return rec(tuple(z0), n0, h)


def mc_value(z0, n0, alpha, kappa, rho_ns, m, h, policy, n_sims, seed):
    rng = np.random.default_rng(seed)
    z = np.tile(z0, (n_sims, 1))
    env = Env(alpha, kappa, rho_ns, np.full(n_sims, float(n0)), z, rng)
    tot = np.zeros(n_sims)
    for t in range(h):
        if policy == "aware":
            a = plan_aware(env.z, env.n, alpha, kappa, rho_ns, m, h - t)
        else:
            a = myopic(env.z, alpha, kappa, m)
        tot += m[a] * env.step(a)
    return float(tot.mean()), float(tot.std() / np.sqrt(n_sims))


def main():
    with open(os.path.join(OUT, "p1a_consumption.json")) as f:
        p1a = json.load(f)
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        cat_list = pickle.load(f)["cat_list"]
    with open(os.path.join(SCRATCH, "user_hist.pkl"), "rb") as f:
        hist = pickle.load(f)
    with open(os.path.join(SCRATCH, "train_clicks.pkl"), "rb") as f:
        tc = pickle.load(f)

    alpha = np.array([p1a["alpha_by_category"][c]["alpha"] for c in ITEMS])
    kappa = p1a["kappa"]
    m = np.array([MARGINS[c] for c in ITEMS])
    idx8 = [cat_list.index(c) for c in ITEMS]
    users = sorted(hist)
    pool_full = np.stack([hist[u] for u in users]).astype(np.float64)[:, idx8]
    mass = pool_full.sum(axis=1)
    ok = mass > 0.5
    pool = pool_full[ok] / pool_full[ok].sum(axis=1, keepdims=True)
    # empirical tenure: history length (clicks), capped
    hlen = np.array([len(tc.get(u, [])) + 20 for u in users])[ok].astype(float)
    hlen = np.clip(hlen, 5, 100)
    print(f"pool {pool.shape}, tenure median {np.median(hlen):.0f}")

    # ---- exact-DP validation with backfire (n=3, H=7)
    a3 = np.array([-2.0, -3.0, -1.5]); m3 = np.array([1.0, 4.0, 1.5])
    z3 = np.array([0.6, 0.3, 0.1])
    val = {}
    for rns in [0.0, -0.05]:
        dp = exact_dp(z3, 10.0, a3, 4.0, rns, m3, 7)
        aw = mc_value(z3, 10.0, a3, 4.0, rns, m3, 7, "aware", 20000, 99)
        my = mc_value(z3, 10.0, a3, 4.0, rns, m3, 7, "myopic", 20000, 99)
        val[f"rho_ns={rns}"] = {"exact_dp": dp, "aware_mc": aw[0],
                                "aware_se": aw[1], "myopic_mc": my[0]}
        print(f"validate rho_ns={rns}: DP={dp:.4f} aware={aw[0]:.4f}"
              f"+-{aw[1]:.4f} myopic={my[0]:.4f}", flush=True)

    tenures = {"new_n0_5": np.full(len(pool), 5.0),
               "tenured_n0_40": np.full(len(pool), 40.0),
               "empirical_mix": hlen}
    results = {}
    for kap, tag in [(kappa, "kappa_hat"), (5.0, "kappa_5")]:
        for tname, n0pool in tenures.items():
            if tag == "kappa_5" and tname != "empirical_mix":
                continue
            for rns in [0.0, -0.002, -0.015]:
                key = f"{tag}|{tname}|rho_ns={rns}"
                aw = [run("aware", alpha, kap, rns, m, pool, n0pool, 7000 + s)
                      for s in SEEDS]
                my = [run("myopic", alpha, kap, rns, m, pool, n0pool, 7000 + s)
                      for s in SEEDS]
                ra = np.array([x[0] for x in aw]); rm = np.array([x[0] for x in my])
                results[key] = {
                    "aware_rev": float(ra.mean()), "myopic_rev": float(rm.mean()),
                    "gap_pct": float(100 * (ra.mean() - rm.mean()) / rm.mean()),
                    "per_seed_gap_pct": (100 * (ra - rm) / rm).tolist(),
                    "aware_bridge_share": float(np.mean([x[1] for x in aw])),
                }
                print(f"{key}: gap {results[key]['gap_pct']:.2f}% "
                      f"bridge {results[key]['aware_bridge_share']:.3f}", flush=True)

    out = {"items": ITEMS, "margins": MARGINS, "H": H, "N_EP": N_EP,
           "seeds": SEEDS, "kappa_hat": kappa, "validation": val,
           "results": results}
    with open(os.path.join(OUT, "p8_sim_transition.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved p8_sim_transition.json")


if __name__ == "__main__":
    main()
