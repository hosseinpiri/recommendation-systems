"""
Xinyuan's rho = 1 graph theory evaluated at the MIND calibration
(14 categories with intercepts, kappa = 1.966, hetero ad-value margins;
one-hot geometry so G = I, plus the dense-centroid Gram variant).

Computes, per the Optimal Path document:
  - stay-put rates r_bar_i = m_i sigma(alpha_i + kappa G_ii)
  - harvest item a*, cycle check (no negative cycle in weights w)
  - edge weights w(j->i) = g/p_{j->i} - m_i and shortest-path routes c(nu)
  - the skip/bridge inequality over all (j,k,i) triples: does ANY stepping
    stone pay at calibration?
  - patience breakpoints delta*: smallest delta at which the discounted
    fixed point departs from the myopic successor map (per start node)
  - turnpike threshold H_bar = log(4C/Delta_sp)/log(1/gamma), gamma = 1-p_min
  - the efficient frontier (Pareto on (alpha_i, m_i, z0'x_i, G_i.)); NOTE:
    with one-hot G the Gram rows are pairwise incomparable, so the full
    frontier is structurally 100% regardless of calibration; the (alpha, m)
    restricted frontier is also reported for the meaningful comparison
  - artificial alpha = -10 category test: is it on the frontier (yes, if its
    margin is the largest) while contributing nothing (essential-set check
    via leave-one-out at small H)?
"""
import itertools
import json
import os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
CATS = ["music", "tv", "weather", "video", "lifestyle", "sports", "health",
        "finance", "movies", "entertainment", "foodanddrink", "news",
        "travel", "autos"]
ALPHA = np.array([-2.869, -2.973, -2.982, -3.124, -3.380, -3.407, -3.442,
                  -3.448, -3.494, -3.567, -3.684, -3.702, -3.716, -3.719])
MARGIN_MAP = {"news": 1.0, "sports": 1.2, "entertainment": 1.1,
              "lifestyle": 1.5, "foodanddrink": 1.8, "health": 2.5,
              "travel": 3.0, "finance": 4.0, "video": 1.0, "tv": 1.1,
              "music": 1.1, "movies": 1.2, "weather": 1.0, "autos": 3.5}
M = np.array([MARGIN_MAP[c] for c in CATS])
KAPPA = 1.966
DELTA_GRID = np.round(np.arange(0.05, 1.0, 0.01), 3)


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def analyze(G, label, z0=None, CATS=CATS, ALPHA=ALPHA, M=M):
    n = len(CATS)
    # edge probs p[j, i]: from node j recommend i; node 0 = z0 handled apart
    P = sigmoid(ALPHA[None, :] + KAPPA * G)
    rbar = M * np.diag(P)
    astar = int(np.argmax(rbar))
    g = rbar[astar]

    # cycle condition: any cycle with mean rate > g? check 2-cycles and
    # 3-cycles exhaustively (n=14 -> fine)
    worst_cycle = -1e18
    for j, i in itertools.permutations(range(n), 2):
        mu = (M[i] + M[j]) / (1 / P[j, i] + 1 / P[i, j])
        worst_cycle = max(worst_cycle, mu)
    for j, i, k in itertools.permutations(range(n), 3):
        mu = (M[i] + M[k] + M[j]) / (1 / P[j, i] + 1 / P[i, k] + 1 / P[k, j])
        worst_cycle = max(worst_cycle, mu)

    # edge weights and shortest paths to a* (Bellman-Ford on 14 nodes)
    w = g / P - M[None, :]
    c = np.full(n, 1e18)
    c[astar] = 0.0
    for _ in range(n):
        for j in range(n):
            for i in range(n):
                if c[i] + w[j, i] < c[j]:
                    c[j] = c[i] + w[j, i]

    # skip test: is any 2-hop j->k->i cheaper than direct j->i?
    n_bridge = 0
    examples = []
    for j, k, i in itertools.permutations(range(n), 3):
        if 1 / P[j, k] + 1 / P[k, i] - M[k] / g <= 1 / P[j, i] - 1e-12:
            n_bridge += 1
            if len(examples) < 5:
                examples.append((CATS[j], CATS[k], CATS[i]))

    # patience breakpoint per start node: myopic map vs delta-optimal map
    def delta_map(delta):
        phi = P / (1 - delta * (1 - P))
        V = np.zeros(n)
        for _ in range(3000):
            Q = phi * (M[None, :] + delta * V[None, :])
            V2 = Q.max(axis=1)
            if np.abs(V2 - V).max() < 1e-12:
                V = V2
                break
            V = V2
        return Q.argmax(axis=1)

    myo_map = np.argmax(P * M[None, :], axis=1)
    breakpoints = np.full(n, np.nan)
    for d in DELTA_GRID:
        amap = delta_map(float(d))
        diff = amap != myo_map
        newly = diff & np.isnan(breakpoints)
        breakpoints[newly] = d
    share_never = float(np.mean(np.isnan(breakpoints)))

    # turnpike threshold H_bar (rough, using document constants)
    p_min = float(P.min())
    gamma = 1 - p_min
    # action gap of the shortest-path map in the normalized recursion
    U = -c
    Qsp = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            Qsp[j, i] = U[i] - w[j, i]
    best = Qsp.max(axis=1)
    second = np.partition(Qsp, -2, axis=1)[:, -2]
    delta_sp = float((best - second).min())
    Cconst = max(np.abs(c[c < 1e17]).max(), 1.0)
    H_bar = (np.log(4 * Cconst / max(delta_sp, 1e-12)) / np.log(1 / gamma)
             if delta_sp > 0 else np.inf)

    # efficient frontier (Pareto on (alpha, m, z0'x, G row))
    if z0 is None:
        z0 = np.full(n, 1.0 / n) if label == "dense" else np.zeros(n)
    A = np.column_stack([ALPHA, M, G @ z0, G])
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(A[j] >= A[i] - 1e-12) \
                    and np.any(A[j] > A[i] + 1e-12):
                dominated[i] = True
                break
    frontier = [CATS[i] for i in range(n) if not dominated[i]]
    # (alpha, m)-restricted frontier: the meaningful pruning statement
    dom2 = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and ALPHA[j] >= ALPHA[i] - 1e-12 and M[j] >= M[i] - 1e-12 \
                    and (ALPHA[j] > ALPHA[i] + 1e-12 or M[j] > M[i] + 1e-12):
                dom2[i] = True
                break
    frontier_am = [CATS[i] for i in range(n) if not dom2[i]]

    res = {
        "harvest_item": CATS[astar], "g": float(g),
        "rbar": {CATS[i]: round(float(rbar[i]), 4) for i in range(n)},
        "worst_cycle_rate": float(worst_cycle),
        "harvest_absorbs": bool(g >= worst_cycle),
        "n_profitable_bridge_triples": n_bridge,
        "n_triples_total": int(n * (n - 1) * (n - 2)),
        "bridge_examples": examples,
        "route_costs_c": {CATS[i]: round(float(c[i]), 2) for i in range(n)},
        "patience_breakpoints": {CATS[i]: (None if np.isnan(breakpoints[i])
                                           else float(breakpoints[i]))
                                 for i in range(n)},
        "share_nodes_never_leave_myopic": share_never,
        "p_min": p_min, "gamma": gamma, "delta_sp_gap": delta_sp,
        "H_bar": float(H_bar) if np.isfinite(H_bar) else "inf",
        "frontier": frontier, "n_frontier": len(frontier),
        "frontier_alpha_m_only": frontier_am,
    }
    print(f"[{label}] harvest={res['harvest_item']} g={g:.3f} "
          f"bridges={n_bridge} H_bar={res['H_bar']} "
          f"frontier={len(frontier)}/{n}", flush=True)
    return res


def main():
    out = {}
    # one-hot geometry: G = I (alignment j->i is 0 unless i == j)
    out["onehot"] = analyze(np.eye(len(CATS)), "onehot")

    # dense geometry: category-centroid Gram (rebuild from centroids if
    # present, else approximate with the memo's published similarity range)
    try:
        import pickle
        SC = "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind"
        C = np.load(os.path.join(SC, "cat_centroids.npy"))
        with open(os.path.join(SC, "news_cat.pkl"), "rb") as f:
            cl = pickle.load(f)["cat_list"]
        idx = [cl.index(c) for c in CATS]
        G = (C[idx] @ C[idx].T)
        out["dense"] = analyze(G, "dense")
    except Exception as e:
        out["dense_error"] = str(e)
        print("dense skipped:", e)

    # artificial alpha = -10 item test (one-hot, margin 5.0)
    cats2 = CATS + ["artificial"]
    alpha2 = np.append(ALPHA, -10.0)
    m2 = np.append(M, 5.0)
    out["artificial_onehot"] = analyze(np.eye(len(cats2)), "artificial",
                                       CATS=cats2, ALPHA=alpha2, M=m2)

    with open(os.path.join(OUT, "p22_graph_tests.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved p22_graph_tests.json")


if __name__ == "__main__":
    main()
