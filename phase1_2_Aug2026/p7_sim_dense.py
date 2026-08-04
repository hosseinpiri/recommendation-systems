"""
Dense-geometry simulation: the bridge test.

Items are the 8 largest categories represented by their unit-normalized LSA
centroids, so the feature Gram matrix has off-diagonals of 0.44-0.89 and
cross-item bridges are geometrically possible. State z in R^50, transition
z+ = (1-rho) z + rho x_a on click, p = sigma(alpha_c + kappa_d z.x_c) with
(alpha, kappa_d) from the dense binary logit (model B). Initial states are
empirical dense history vectors. Margins = heterogeneous ad-value map.

Reports: aware-vs-myopic gap over rho, bridge share, and the exact
two-period bridge diagnostic on empirical states (share of states where the
optimal two-period first action is a persuasive bridge), computed under both
dense and one-hot geometry for contrast.
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
RHO_SWEEP = [0.0, 0.035, 0.15, 0.35]


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


class DenseEnv:
    def __init__(self, alpha, kappa, rho, X, z0, rng):
        self.alpha, self.kappa, self.rho, self.X = alpha, kappa, rho, X
        self.z = z0.copy()
        self.rng = rng

    def probs(self):
        return sigmoid(self.alpha[None, :] + self.kappa * (self.z @ self.X.T))

    def step(self, a):
        U = self.z.shape[0]
        p = self.probs()[np.arange(U), a]
        yv = (self.rng.random(U) < p).astype(np.float64)
        if self.rho > 0:
            upd = (yv * self.rho)[:, None]
            self.z = (1 - upd) * self.z + upd * self.X[a]
        return yv


def myopic(z, alpha, kappa, X, m):
    return np.argmax(m[None, :] * sigmoid(alpha[None, :] + kappa * (z @ X.T)),
                     axis=1)


def fluid_value(z, alpha, kappa, rho, X, m, steps):
    z = z.copy()
    U = z.shape[0]
    total = np.zeros(U)
    for _ in range(steps):
        P = sigmoid(alpha[None, :] + kappa * (z @ X.T))
        ev = m[None, :] * P
        a = np.argmax(ev, axis=1)
        p = P[np.arange(U), a]
        total += ev[np.arange(U), a]
        if rho > 0:
            upd = (p * rho)[:, None]
            z = (1 - upd) * z + upd * X[a]
    return total


def plan_aware(z, alpha, kappa, rho, X, m, h_left, depth=2):
    U, n = z.shape[0], len(alpha)
    if h_left <= 1 or rho == 0.0:
        return myopic(z, alpha, kappa, X, m)
    depth = min(depth, h_left - 1)

    def v0(w, h):
        if h <= 0:
            return np.zeros(w.shape[0])
        return fluid_value(w, alpha, kappa, rho, X, m, h)

    def v1(w, h):
        if h <= 0:
            return np.zeros(w.shape[0])
        vstay = v0(w, h - 1)
        best = np.full(w.shape[0], -np.inf)
        P = sigmoid(alpha[None, :] + kappa * (w @ X.T))
        for b in range(n):
            w1 = (1 - rho) * w + rho * X[b]
            np.maximum(best, P[:, b] * (m[b] + v0(w1, h - 1))
                       + (1 - P[:, b]) * vstay, out=best)
        return best

    vfun = v0 if depth <= 1 else v1
    vstay = vfun(z, h_left - 1)
    Q = np.zeros((U, n))
    P = sigmoid(alpha[None, :] + kappa * (z @ X.T))
    for a in range(n):
        z1 = (1 - rho) * z + rho * X[a]
        Q[:, a] = P[:, a] * (m[a] + vfun(z1, h_left - 1)) + (1 - P[:, a]) * vstay
    return np.argmax(Q, axis=1)


def run(policy, alpha, kappa, rho, X, m, pool, seed):
    rng = np.random.default_rng(seed)
    rev, bridge, recs = [], 0, 0
    for _ in range(N_EP // BATCH):
        z0 = pool[rng.integers(0, pool.shape[0], BATCH)].copy()
        env = DenseEnv(alpha, kappa, rho, X, z0, rng)
        r = np.zeros(BATCH)
        for t in range(H):
            if policy == "aware":
                a = plan_aware(env.z, alpha, kappa, rho, X, m, H - t)
            else:
                a = myopic(env.z, alpha, kappa, X, m)
            amy = myopic(env.z, alpha, kappa, X, m)
            bridge += int(np.sum(a != amy))
            recs += BATCH
            r += m[a] * env.step(a)
        rev.extend(r.tolist())
    return float(np.mean(rev)), bridge / recs


def two_period_bridge_share(pool, alpha, kappa, X, m, rho):
    """Exact h=2 diagnostic: share of states whose optimal first action is a
    persuasive bridge (not myopically optimal, and the follow-up target after
    its success differs from the bridge item)."""
    P = sigmoid(alpha[None, :] + kappa * (pool @ X.T))
    r = m[None, :] * P
    rbar = r.max(axis=1)
    n = len(alpha)
    Psi = np.zeros_like(r)
    tgt = np.zeros((pool.shape[0], n), dtype=int)
    for i in range(n):
        z1 = (1 - rho) * pool + rho * X[i]
        r1 = m[None, :] * sigmoid(alpha[None, :] + kappa * (z1 @ X.T))
        Psi[:, i] = r[:, i] + P[:, i] * (r1.max(axis=1) - rbar)
        tgt[:, i] = np.argmax(r1, axis=1)
    first = np.argmax(Psi, axis=1)
    my = np.argmax(r, axis=1)
    rows = np.arange(pool.shape[0])
    bridge = (first != my) & (tgt[rows, first] != first)
    return float(bridge.mean())


def main():
    with open(os.path.join(OUT, "p5_slate_dense.json")) as f:
        est = json.load(f)
    with open(os.path.join(OUT, "p1a_consumption.json")) as f:
        p1a = json.load(f)
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        cat_list = pickle.load(f)["cat_list"]

    kappa_d = est["B_dense"]["kappa"]
    cent = np.load(os.path.join(SCRATCH, "cat_centroids.npy"))
    X = cent[[cat_list.index(c) for c in ITEMS]].astype(np.float64)

    # dense-model intercepts: refit quickly is overkill; use p1a alphas shifted
    # so that mean predicted p at empirical states matches the dense model.
    # Cleaner: alpha_dense_c = logit-intercept from model B is not saved per
    # category, so recompute alpha_c = p1a alpha + kappa_cat*mean(cat_align)
    # - kappa_d*mean(dense_align) per category is fragile; instead calibrate
    # each alpha_c so the average click prob at empirical states equals the
    # p1a-implied average for that category.
    with open(os.path.join(SCRATCH, "dense_rho_pieces.pkl"), "rb") as f:
        prof = pickle.load(f)
    pool = np.stack([v["z0"] for v in prof.values()]).astype(np.float64)
    alpha_cat = np.array([p1a["alpha_by_category"][c]["alpha"] for c in ITEMS])
    kappa_cat = p1a["kappa"]
    # category-space mean alignment for these users
    with open(os.path.join(SCRATCH, "user_hist.pkl"), "rb") as f:
        hist = pickle.load(f)
    users = sorted(prof)
    zc = np.stack([hist[u] for u in users if u in hist]).astype(np.float64)
    zc8 = zc[:, [cat_list.index(c) for c in ITEMS]]
    target_mean_logit = alpha_cat[None, :] + kappa_cat * zc8[:len(pool)]
    dense_align = pool @ X.T
    alpha = (target_mean_logit.mean(axis=0) - kappa_d * dense_align.mean(axis=0))
    m = np.array([MARGINS[c] for c in ITEMS])
    print("kappa_dense", round(kappa_d, 3))
    print("alpha_dense", np.round(alpha, 3))
    print("dense_align mean/sd", round(dense_align.mean(), 3),
          round(dense_align.std(), 3))

    results = {}
    for rho in RHO_SWEEP:
        aware = [run("aware", alpha, kappa_d, rho, X, m, pool, 1000 * s + int(rho * 100))
                 for s in SEEDS]
        myop = [run("myopic", alpha, kappa_d, rho, X, m, pool, 1000 * s + int(rho * 100))
                for s in SEEDS]
        ra = np.array([a[0] for a in aware])
        rm = np.array([a[0] for a in myop])
        results[str(rho)] = {
            "aware_rev": float(ra.mean()), "myopic_rev": float(rm.mean()),
            "gap_pct": float(100 * (ra.mean() - rm.mean()) / rm.mean()),
            "aware_bridge_share": float(np.mean([a[1] for a in aware])),
            "per_seed_gap_pct": (100 * (ra - rm) / rm).tolist(),
            "bridge2_share_dense": two_period_bridge_share(
                pool, alpha, kappa_d, X, m, rho) if rho > 0 else 0.0,
        }
        print(f"rho={rho}: aware {ra.mean():.3f} myopic {rm.mean():.3f} "
              f"gap {results[str(rho)]['gap_pct']:.2f}% "
              f"bridge2 {results[str(rho)]['bridge2_share_dense']:.3f}", flush=True)

    # one-hot contrast for the two-period diagnostic (category geometry)
    onehot_share = {}
    zc8n = zc8 / np.clip(zc8.sum(axis=1, keepdims=True), 1e-9, None)
    for rho in [0.035, 0.15, 0.35]:
        onehot_share[str(rho)] = two_period_bridge_share(
            zc8n, alpha_cat, kappa_cat, np.eye(len(ITEMS)), m, rho)
    print("one-hot two-period bridge shares:", onehot_share)

    out = {"items": ITEMS, "kappa_dense": kappa_d, "alpha_dense": alpha.tolist(),
           "margins": MARGINS, "H": H, "N_EP": N_EP, "seeds": SEEDS,
           "results": results, "bridge2_share_onehot": onehot_share,
           "gram_offdiag_mean": float((X @ X.T)[~np.eye(len(ITEMS), dtype=bool)].mean())}
    with open(os.path.join(OUT, "p7_sim_dense.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved p7_sim_dense.json")


if __name__ == "__main__":
    main()
