"""
Python port of Xinyuan's dynamic_recommendation_general_case_v2.m
(5 items, 2D features, exact DP on the reachable tree), extended with the
MIND-estimated frictions:

  backfire : rejection moves taste AWAY from the recommended feature,
             z- = z + rho_ns (x_i - z) with rho_ns < 0  (his header notes the
             Bellman then needs the extra rejection-transition term; included)
  tenure   : rho_t = 1 / (n0 + N_t) declining click step

Scenarios: baseline linear pull (his), spillover (his matrix), baseline +
backfire at the MIND-estimated per-impression scale and at 10x, tenure
variant, and MIND-calibrated (kappa=1.966, rho=0.035) versions of each.
For each: the optimal success-path action sequence and the aware-vs-myopic
value gap at z0 (exact DP vs exact myopic-policy evaluation).
"""
import json
import os
import numpy as np
from functools import lru_cache

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

ITEMS = ["T", "B2", "B1", "L", "D"]
X = np.array([[1.00, 0.25], [0.45, 0.60], [-0.20, 0.35],
              [-1.00, 0.00], [0.20, -0.30]])
ALPHA = np.array([-0.75, 0.2, 0.2, 0.5, 0.0])
MARGIN = np.array([12.0, 3.0, 1.5, 2.5, 4.0])
Z0 = np.array([-1.0, 0.0])
MSPILL = np.array([[1.00, 0.25], [0.10, 0.85]])
H = 7


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def make_model(kappa, rho, mode="linear", rho_ns=0.0, tenure_n0=None):
    def p(i, z):
        return float(sigmoid(ALPHA[i] + kappa * (z @ X[i])))

    def succ(z, i, n):
        r = rho if tenure_n0 is None else 1.0 / (tenure_n0 + n + 1.0)
        if mode == "spillover":
            return z + r * (MSPILL @ (X[i] - z))
        return (1 - r) * z + r * X[i]

    def fail(z, i):
        if rho_ns == 0.0:
            return z
        return z + rho_ns * (X[i] - z)

    return p, succ, fail


def exact_values(p, succ, fail, h, z0):
    """Exact DP with rejection transitions; returns (V*, V_myopic, path)."""
    memo = {}

    def V(h, z, n):
        if h == 0:
            return 0.0
        key = (h, round(z[0], 10), round(z[1], 10), n)
        if key in memo:
            return memo[key]
        best = -1e18
        for i in range(5):
            pi = p(i, z)
            v = pi * (MARGIN[i] + V(h - 1, succ(z, i, n), n + 1)) \
                + (1 - pi) * V(h - 1, fail(z, i), n)
            best = max(best, v)
        memo[key] = best
        return best

    memo_m = {}

    def myo_action(z):
        return int(np.argmax([p(i, z) * MARGIN[i] for i in range(5)]))

    def Vm(h, z, n):
        if h == 0:
            return 0.0
        key = (h, round(z[0], 10), round(z[1], 10), n)
        if key in memo_m:
            return memo_m[key]
        i = myo_action(z)
        pi = p(i, z)
        v = pi * (MARGIN[i] + Vm(h - 1, succ(z, i, n), n + 1)) \
            + (1 - pi) * Vm(h - 1, fail(z, i), n)
        memo_m[key] = v
        return v

    # optimal success-path actions
    path_actions = []
    z, n = z0.copy(), 0
    for h_left in range(h, 0, -1):
        scores = []
        for i in range(5):
            pi = p(i, z)
            scores.append(pi * (MARGIN[i] + V(h_left - 1, succ(z, i, n), n + 1))
                          + (1 - pi) * V(h_left - 1, fail(z, i), n))
        istar = int(np.argmax(scores))
        path_actions.append(ITEMS[istar])
        z = succ(z, istar, n)
        n += 1
    return V(h, z0, 0), Vm(h, z0, 0), path_actions


def main():
    scenarios = {
        "xy_baseline": dict(kappa=3.0, rho=0.45, mode="linear"),
        "xy_spillover": dict(kappa=3.0, rho=0.45, mode="spillover"),
        "xy_backfire_mind": dict(kappa=3.0, rho=0.45, mode="linear",
                                 rho_ns=-0.002),
        "xy_backfire_10x": dict(kappa=3.0, rho=0.45, mode="linear",
                                rho_ns=-0.02),
        "xy_tenure": dict(kappa=3.0, rho=0.45, mode="linear", tenure_n0=2),
        "mind_baseline": dict(kappa=1.966, rho=0.035, mode="linear"),
        "mind_backfire": dict(kappa=1.966, rho=0.035, mode="linear",
                              rho_ns=-0.002),
        "mind_backfire_strong": dict(kappa=1.966, rho=0.035, mode="linear",
                                     rho_ns=-0.02),
    }
    out = {}
    for name, kw in scenarios.items():
        p, succ, fail = make_model(**kw)
        v, vm, actions = exact_values(p, succ, fail, H, Z0)
        out[name] = {"V_opt": round(v, 4), "V_myopic": round(vm, 4),
                     "premium_pct": round(100 * (v - vm) / vm, 2),
                     "success_path": actions, **{k: str(v2) for k, v2 in kw.items()}}
        print(f"{name:22s} V*={v:8.3f} Vmyo={vm:8.3f} "
              f"premium={100*(v-vm)/vm:6.2f}%  path={'-'.join(actions)}")
    with open(os.path.join(OUT, "p10_xinyuan_toy.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
