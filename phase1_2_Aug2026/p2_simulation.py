"""
Phase 2: MIND-calibrated simulation of cultivation-aware vs cultivation-blind policies.

Environment (the group's model, Jiangze May / Meichun July, finite horizon):
  n items = top categories, x_i = e_i, uniform margins m = 1
  P(click | z, i) = sigma(alpha_i + kappa z_i)
  on click: z <- (1 - rho) z + rho e_i ; no click: z unchanged
  alpha, kappa calibrated from Phase 1a; rho swept over a grid containing rho_hat (1b);
  z0 drawn from the empirical MIND distribution of user history shares (renormalized).

Policies:
  oracle-aware : knows (alpha, kappa, rho); depth-D lookahead + fluid myopic rollout
  oracle-blind : knows (alpha, kappa); plans as if rho = 0  =>  myopic argmax
  learn-aware  : knows (kappa, rho); learns alpha by per-item offset-logistic MLE,
                 batched episodes, plug-in into the same planner
  learn-blind  : same learner, myopic planner

Outputs: per-(rho, policy) mean clicks/episode with CIs, regret vs oracle-aware,
bridge share (recommendations that are not myopically optimal), planner validation
against exact DP on a small instance, all to output/p2_simulation.json + figures.
"""
import json
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SCRATCH = os.environ.get(
    "SCRATCH",
    "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind",
)
os.makedirs(OUT, exist_ok=True)

H = 30                 # periods per episode (recommendation opportunities per user)
N_EP = 3000            # episodes (users) per policy per rho
BATCH = 150            # episodes simulated in parallel; learners re-fit between batches
DEPTH = 2              # lookahead depth of the aware planner
N_ITEMS = 8
SEEDS = [1, 2, 3, 4, 5]
RHO_SWEEP = None       # set in main() to include rho_hat


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


# ---------------------------------------------------------------- environment
class Env:
    def __init__(self, alpha, kappa, rho, z0, rng):
        self.alpha, self.kappa, self.rho = alpha, kappa, rho
        self.z = z0.copy()          # (U, n)
        self.rng = rng

    def step(self, a):
        U = self.z.shape[0]
        p = sigmoid(self.alpha[a] + self.kappa * self.z[np.arange(U), a])
        y = (self.rng.random(U) < p).astype(np.float64)
        if self.rho > 0:
            upd = y[:, None] * self.rho
            self.z = (1 - upd) * self.z
            self.z[np.arange(U), a] += upd[:, 0]
        return y


# ---------------------------------------------------------------- planners
def myopic_action(z, alpha, kappa, m):
    return np.argmax(m[None, :] * sigmoid(alpha[None, :] + kappa * z), axis=1)


def fluid_rollout_value(z, alpha, kappa, rho, steps, m):
    """Deterministic mean-field rollout of the myopic policy; returns (U,) value."""
    z = z.copy()
    U, n = z.shape
    total = np.zeros(U)
    for _ in range(steps):
        ev = m[None, :] * sigmoid(alpha[None, :] + kappa * z)
        a = np.argmax(ev, axis=1)
        p = sigmoid(alpha[None, :] + kappa * z)[np.arange(U), a]
        total += ev[np.arange(U), a]
        if rho > 0:
            upd = (p * rho)[:, None]
            z = (1 - upd) * z
            z[np.arange(U), a] += upd[:, 0]
    return total


def plan_aware(z, alpha, kappa, rho, h_left, m, depth=DEPTH):
    """Depth-`depth` expectimax over actions with fluid myopic rollout at leaves,
    memoizing the shared no-click continuation once per level.
    Vectorized over users. Returns (U,) actions."""
    U, n = z.shape
    if h_left <= 1 or rho == 0.0:
        return myopic_action(z, alpha, kappa, m)
    depth = min(depth, h_left - 1)

    def v0(w, h):  # fluid myopic continuation value
        if h <= 0:
            return np.zeros(w.shape[0])
        return fluid_rollout_value(w, alpha, kappa, rho, h, m)

    def v1(w, h):  # exact one-step lookahead, fluid continuation
        if h <= 0:
            return np.zeros(w.shape[0])
        vstay = v0(w, h - 1)
        best = np.full(w.shape[0], -np.inf)
        for b in range(n):
            p = sigmoid(alpha[b] + kappa * w[:, b])
            w1 = w * (1 - rho)
            w1[:, b] += rho
            np.maximum(best, p * (m[b] + v0(w1, h - 1)) + (1 - p) * vstay, out=best)
        return best

    vfun = v0 if depth <= 1 else v1
    vstay = vfun(z, h_left - 1)
    Q = np.zeros((U, n))
    for a in range(n):
        p = sigmoid(alpha[a] + kappa * z[:, a])
        z1 = z * (1 - rho)
        z1[:, a] += rho
        Q[:, a] = p * (m[a] + vfun(z1, h_left - 1)) + (1 - p) * vstay
    return np.argmax(Q, axis=1)


# ---------------------------------------------------------------- learner
class AlphaLearner:
    """Per-item MLE of alpha_i in sigma(alpha_i + offset) with L2 reg toward 0."""
    def __init__(self, n, lam=1.0):
        self.n = n
        self.lam = lam
        self.obs = [[] for _ in range(n)]   # (offset, y)
        self.alpha_hat = np.zeros(n)

    def add(self, a, offset, y):
        for j in range(len(a)):
            self.obs[a[j]].append((offset[j], y[j]))

    def refit(self):
        for i in range(self.n):
            if not self.obs[i]:
                continue
            arr = np.array(self.obs[i])
            off, y = arr[:, 0], arr[:, 1]
            al = self.alpha_hat[i]
            for _ in range(25):  # Newton
                p = sigmoid(al + off)
                g = np.sum(y - p) - self.lam * al
                h = -np.sum(p * (1 - p)) - self.lam
                step = g / h
                al -= step
                if abs(step) < 1e-8:
                    break
            self.alpha_hat[i] = np.clip(al, -8.0, 2.0)


# ---------------------------------------------------------------- simulation
def run_policy(policy, alpha, kappa, rho, z0_pool, seed, m, H_ep=None):
    rng = np.random.default_rng(seed)
    learner = AlphaLearner(len(alpha)) if policy.startswith("learn") else None
    clicks_ep = []
    rev_ep = []
    bridge = 0
    total_recs = 0
    H_ep = H if H_ep is None else H_ep
    n_batches = N_EP // BATCH
    for b in range(n_batches):
        idx = rng.integers(0, z0_pool.shape[0], BATCH)
        z0 = z0_pool[idx].copy()
        env = Env(alpha, kappa, rho, z0, rng)
        a_hat = learner.alpha_hat if learner else alpha
        ep_clicks = np.zeros(BATCH)
        ep_rev = np.zeros(BATCH)
        for t in range(H_ep):
            if policy.endswith("aware"):
                a = plan_aware(env.z, a_hat, kappa, rho, H_ep - t, m)
            else:
                a = myopic_action(env.z, a_hat, kappa, m)
            if policy.startswith("oracle"):
                amy = myopic_action(env.z, alpha, kappa, m)
                bridge += int(np.sum(a != amy))
                total_recs += BATCH
            offset = kappa * env.z[np.arange(BATCH), a]
            y = env.step(a)
            ep_clicks += y
            ep_rev += m[a] * y
            if learner:
                learner.add(a, offset, y)
        if learner:
            learner.refit()
        clicks_ep.extend(ep_clicks.tolist())
        rev_ep.extend(ep_rev.tolist())
    res = {"mean_clicks": float(np.mean(clicks_ep)),
           "se_clicks": float(np.std(clicks_ep) / np.sqrt(len(clicks_ep))),
           "mean_revenue": float(np.mean(rev_ep)),
           "se_revenue": float(np.std(rev_ep) / np.sqrt(len(rev_ep)))}
    if total_recs:
        res["bridge_share"] = bridge / total_recs
    if learner:
        res["alpha_hat_final"] = learner.alpha_hat.tolist()
    return res


# ------------------------------------------------------- planner validation
def exact_dp_value(z0, alpha, kappa, rho, h, m):
    """Exact value by recursion over the reachable tree (small n, h only)."""
    from functools import lru_cache
    n = len(alpha)

    def rec(z, hh):
        if hh == 0:
            return 0.0
        best = -1.0
        for a in range(n):
            p = sigmoid(alpha[a] + kappa * z[a])
            z1 = tuple((1 - rho) * np.array(z) + rho * np.eye(n)[a])
            v = p * (m[a] + rec(z1, hh - 1)) + (1 - p) * rec(z, hh - 1)
            best = max(best, v)
        return best

    return rec(tuple(z0), h)


def mc_policy_value(z0, alpha, kappa, rho, h, planner, n_sims, seed, m):
    rng = np.random.default_rng(seed)
    z0m = np.tile(z0, (n_sims, 1))
    env = Env(alpha, kappa, rho, z0m, rng)
    tot = np.zeros(n_sims)
    for t in range(h):
        if planner == "aware":
            a = plan_aware(env.z, alpha, kappa, rho, h - t, m)
        else:
            a = myopic_action(env.z, alpha, kappa, m)
        tot += m[a] * env.step(a)
    return float(np.mean(tot)), float(np.std(tot) / np.sqrt(n_sims))


# Illustrative per-click margins by vertical (display-ad RPM ratios; news = 1).
# Chosen for the heterogeneous-margin scenario: high-value verticals (finance,
# travel, health, autos) monetize clicks better than general news.
MARGIN_MAP = {
    "news": 1.0, "sports": 1.2, "entertainment": 1.1, "lifestyle": 1.5,
    "foodanddrink": 1.8, "health": 2.5, "travel": 3.0, "finance": 4.0,
    "video": 1.0, "tv": 1.1, "music": 1.1, "movies": 1.2, "weather": 1.0,
    "autos": 3.5,
}


def load_calibration():
    with open(os.path.join(OUT, "p1a_consumption.json")) as f:
        p1a = json.load(f)
    with open(os.path.join(OUT, "p1b_transition.json")) as f:
        p1b = json.load(f)
    import pickle
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        cat_list = pickle.load(f)["cat_list"]
    kappa = p1a["kappa"]
    rho_hat = p1b["rho_hat"]
    top = sorted(p1a["alpha_by_category"].items(),
                 key=lambda kv: -kv[1]["n_fit"])[:N_ITEMS]
    item_names = [k for k, _ in top]
    alpha = np.array([v["alpha"] for _, v in top])
    pool_full = np.load(os.path.join(SCRATCH, "z0_pool.npy"))
    cat_idx = [cat_list.index(c) for c in item_names]
    pool = pool_full[:, cat_idx]
    ok = pool.sum(axis=1) > 0.5   # users mostly within the top categories
    pool = pool[ok] / pool[ok].sum(axis=1, keepdims=True)
    return item_names, alpha, kappa, rho_hat, pool


def sweep(policies, rhos, alpha, kappa, pool, m, H_ep, seeds, metric="mean_revenue"):
    results = {}
    for rho in rhos:
        results[str(rho)] = {}
        for pol in policies:
            runs = [run_policy(pol, alpha, kappa, rho, pool,
                               1000 * s + int(rho * 100), m, H_ep)
                    for s in seeds]
            mv = np.array([r[metric] for r in runs])
            entry = {metric: float(mv.mean()),
                     "se_over_seeds": float(mv.std(ddof=1) / np.sqrt(len(seeds))),
                     "per_seed": mv.tolist(),
                     "mean_clicks": float(np.mean([r["mean_clicks"] for r in runs]))}
            if "bridge_share" in runs[0]:
                entry["bridge_share"] = float(np.mean([r["bridge_share"] for r in runs]))
            results[str(rho)][pol] = entry
            print(f"H={H_ep} rho={rho} {pol}: {entry[metric]:.3f} "
                  f"+- {entry['se_over_seeds']:.3f}", flush=True)
    return results


def main():
    import sys
    scenario = sys.argv[1] if len(sys.argv) > 1 else "uniform"
    item_names, alpha, kappa, rho_hat, pool = load_calibration()
    print("items:", item_names)
    print("alpha:", np.round(alpha, 3), "kappa:", round(kappa, 3),
          "rho_hat:", rho_hat, "scenario:", scenario)
    print("z0 pool:", pool.shape)

    global RHO_SWEEP
    RHO_SWEEP = sorted(set([0.0, 0.05, rho_hat, 0.15, 0.25, 0.35]))
    policies = ["oracle-aware", "oracle-blind", "learn-aware", "learn-blind"]

    if scenario == "uniform":
        m = np.ones(len(alpha))
        # planner validation on a small instance
        a3 = np.array([-2.0, -3.0, -1.5])
        z3 = np.array([0.6, 0.3, 0.1])
        m3 = np.ones(3)
        val = {}
        for rho in [0.1, 0.3]:
            dp = exact_dp_value(z3, a3, 4.0, rho, 6, m3)
            aw, aw_se = mc_policy_value(z3, a3, 4.0, rho, 6, "aware", 20000, 99, m3)
            my, my_se = mc_policy_value(z3, a3, 4.0, rho, 6, "myopic", 20000, 99, m3)
            val[f"rho={rho}"] = {"exact_dp": dp, "planner_mc": aw,
                                 "planner_mc_se": aw_se, "myopic_mc": my,
                                 "myopic_mc_se": my_se}
        results = sweep(policies, RHO_SWEEP, alpha, kappa, pool, m, H, SEEDS,
                        metric="mean_clicks")
        out_name = "p2_simulation.json"
        cfg_m = "uniform m=1"
    elif scenario == "hetero":
        m = np.array([MARGIN_MAP[c] for c in item_names])
        m3h = np.array([1.0, 4.0, 1.5])
        a3 = np.array([-2.0, -3.0, -1.5])
        z3 = np.array([0.6, 0.3, 0.1])
        val = {}
        for rho in [0.1, 0.3]:
            dp = exact_dp_value(z3, a3, 4.0, rho, 6, m3h)
            aw, aw_se = mc_policy_value(z3, a3, 4.0, rho, 6, "aware", 20000, 99, m3h)
            my, my_se = mc_policy_value(z3, a3, 4.0, rho, 6, "myopic", 20000, 99, m3h)
            val[f"rho={rho}"] = {"exact_dp": dp, "planner_mc": aw,
                                 "planner_mc_se": aw_se, "myopic_mc": my,
                                 "myopic_mc_se": my_se}
            print(f"validation rho={rho}: DP={dp:.4f} planner={aw:.4f}"
                  f"+-{aw_se:.4f} myopic={my:.4f}+-{my_se:.4f}")
        results = sweep(policies, RHO_SWEEP, alpha, kappa, pool, m, H, SEEDS)
        out_name = "p2_simulation_hetero.json"
        cfg_m = {c: MARGIN_MAP[c] for c in item_names}
    elif scenario == "engagement":
        # counterfactual high-engagement platform: same kappa, rho, margins,
        # intercepts shifted so mean consumption probability ~0.3 (vs ~0.04)
        m = np.array([MARGIN_MAP[c] for c in item_names])
        alpha = alpha + 3.0
        val = {}
        results = sweep(policies, RHO_SWEEP, alpha, kappa, pool, m, H, SEEDS)
        out_name = "p2_simulation_engagement.json"
        cfg_m = {c: MARGIN_MAP[c] for c in item_names}
    elif scenario == "kappasweep":
        # where does cultivation start to pay? sweep alignment sensitivity
        # kappa at heterogeneous margins, MIND alpha, oracle policies only
        m = np.array([MARGIN_MAP[c] for c in item_names])
        val = {}
        results = {}
        for kap in [kappa, 3.0, 5.0, 7.0]:
            results[f"kappa={round(kap, 3)}"] = sweep(
                ["oracle-aware", "oracle-blind"], [rho_hat, 0.15],
                alpha, kap, pool, m, H, SEEDS)
        out_name = "p2_kappasweep.json"
        cfg_m = {c: MARGIN_MAP[c] for c in item_names}
    elif scenario == "hsweep":
        m = np.array([MARGIN_MAP[c] for c in item_names])
        global N_EP, BATCH
        N_EP, BATCH = 1500, 150
        val = {}
        results = {}
        for H_ep in [30, 60, 120]:
            results[f"H={H_ep}"] = sweep(
                ["oracle-aware", "oracle-blind"], [rho_hat, 0.15],
                alpha, kappa, pool, m, H_ep, SEEDS[:3])
        out_name = "p2_hsweep.json"
        cfg_m = {c: MARGIN_MAP[c] for c in item_names}
    else:
        raise SystemExit(f"unknown scenario {scenario}")

    out = {
        "config": {"H": H, "N_EP": N_EP, "BATCH": BATCH, "DEPTH": DEPTH,
                   "N_ITEMS": N_ITEMS, "SEEDS": SEEDS, "items": item_names,
                   "alpha": alpha.tolist(), "kappa": kappa, "rho_hat": rho_hat,
                   "rho_sweep": RHO_SWEEP, "margins": cfg_m,
                   "scenario": scenario},
        "planner_validation": val,
        "results": results,
        "true_alpha": alpha.tolist(),
    }
    with open(os.path.join(OUT, out_name), "w") as f:
        json.dump(out, f, indent=2)
    print("saved", out_name)


if __name__ == "__main__":
    main()
