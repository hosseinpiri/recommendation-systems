"""Simulation core for cluster cells: category one-hot environment with
margins, optional tenure-dependent rho and backfire, CRN paired policies,
and iso-engagement intercept recentering for kappa sweeps."""
import os
import numpy as np

OUT = os.path.expanduser("~/mind_calib/out")
PREP = os.path.join(OUT, "prep")

ITEMS8 = ["news", "lifestyle", "sports", "finance", "foodanddrink",
          "entertainment", "travel", "health"]
MARGIN_MAP = {"news": 1.0, "sports": 1.2, "entertainment": 1.1,
              "lifestyle": 1.5, "foodanddrink": 1.8, "health": 2.5,
              "travel": 3.0, "finance": 4.0}
# category intercepts from the (to-be-updated) joint estimation; defaults are
# the original p1a values so cells can run before the fit merge completes.
ALPHA8 = np.array([-3.702, -3.380, -3.407, -3.448, -3.684, -3.567,
                   -3.716, -3.442])


def sigmoid(u):
    return 1.0 / (1.0 + np.exp(-u))


def load_pool(universe=9):
    z9 = np.load(os.path.join(PREP, "z0_pool9.npy")).astype(np.float64)
    ten = np.load(os.path.join(PREP, "tenure_pool.npy")).astype(np.float64)
    return z9, np.clip(ten, 5, 100)


def align_cols(z):
    """First 8 columns of the 9-dim state are the item alignments."""
    return z[:, :8]


def recenter_alpha(alpha, kappa_new, kappa_base, pool, match="category"):
    """Iso-engagement recentering: shift each intercept so the pool-average
    click probability under kappa_new equals the baseline under kappa_base."""
    a = align_cols(pool)
    out = alpha.copy()
    for c in range(len(alpha)):
        target = sigmoid(alpha[c] + kappa_base * a[:, c]).mean()
        lo, hi = alpha[c] - 6, alpha[c] + 6
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            v = sigmoid(mid + kappa_new * a[:, c]).mean()
            if v > target:
                hi = mid
            else:
                lo = mid
        out[c] = 0.5 * (lo + hi)
    return out


class Env:
    """9-dim state (8 items + other). Recommending item a in [0,8)."""

    def __init__(self, alpha, kappa, z0, tenure, rng,
                 rho_const=None, rho_ns=0.0):
        self.alpha, self.kappa = alpha, kappa
        self.z = z0.copy()
        self.n = tenure.copy()
        self.rng = rng
        self.rho_const = rho_const
        self.rho_ns = rho_ns

    def probs(self):
        return sigmoid(self.alpha[None, :] + self.kappa * align_cols(self.z))

    def step(self, a):
        U = self.z.shape[0]
        p = self.probs()[np.arange(U), a]
        y = (self.rng.random(U) < p).astype(np.float64)
        rho_t = (np.full(U, self.rho_const) if self.rho_const is not None
                 else 1.0 / (self.n + 1.0))
        upd = y * rho_t
        self.z = (1 - upd[:, None]) * self.z
        self.z[np.arange(U), a] += upd
        self.n += y
        if self.rho_ns != 0.0:
            nb = (1 - y) * self.rho_ns
            self.z = (1 - nb[:, None]) * self.z
            self.z[np.arange(U), a] += nb
            np.clip(self.z, 0.0, None, out=self.z)
            self.z /= self.z.sum(axis=1, keepdims=True)
        return y


def myopic(z, alpha, kappa, m):
    return np.argmax(m[None, :] * sigmoid(alpha[None, :]
                                          + kappa * align_cols(z)), axis=1)


def fluid_value(z, n, alpha, kappa, m, steps, rho_const, rho_ns):
    z = z.copy(); n = n.copy()
    U = z.shape[0]
    total = np.zeros(U)
    for _ in range(steps):
        P = sigmoid(alpha[None, :] + kappa * align_cols(z))
        ev = m[None, :] * P
        a = np.argmax(ev, axis=1)
        p = P[np.arange(U), a]
        total += ev[np.arange(U), a]
        rho_t = (np.full(U, rho_const) if rho_const is not None
                 else 1.0 / (n + 1.0))
        upd = (p * rho_t)[:, None]
        nb = ((1 - p) * rho_ns)[:, None]
        z = (1 - upd - nb) * z
        z[np.arange(U), a] += (upd + nb)[:, 0]
        np.clip(z, 0.0, None, out=z)
        z /= z.sum(axis=1, keepdims=True)
        n = n + p
    return total


def plan_aware(z, n, alpha, kappa, m, h_left, rho_const, rho_ns, depth=2):
    U = z.shape[0]
    nn = len(alpha)
    if h_left <= 1 or (rho_const == 0.0 and rho_ns == 0.0):
        return myopic(z, alpha, kappa, m)
    depth = min(depth, h_left - 1)
    rho_t = (np.full(U, rho_const) if rho_const is not None
             else 1.0 / (n + 1.0))

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
        return fluid_value(w, nv, alpha, kappa, m, h, rho_const, rho_ns)

    def v1(w, nv, h):
        if h <= 0:
            return np.zeros(w.shape[0])
        P = sigmoid(alpha[None, :] + kappa * align_cols(w))
        rt = (np.full(w.shape[0], rho_const) if rho_const is not None
              else 1.0 / (nv + 1.0))
        best = np.full(w.shape[0], -np.inf)
        for b in range(nn):
            vs = v0(succ(w, b, rt), nv + 1, h - 1)
            vf = v0(fail(w, b), nv, h - 1)
            np.maximum(best, P[:, b] * (m[b] + vs) + (1 - P[:, b]) * vf,
                       out=best)
        return best

    vfun = v0 if depth <= 1 else v1
    Q = np.zeros((U, nn))
    P = sigmoid(alpha[None, :] + kappa * align_cols(z))
    for a in range(nn):
        vs = vfun(succ(z, a, rho_t), n + 1, h_left - 1)
        vf = vfun(fail(z, a), n, h_left - 1)
        Q[:, a] = P[:, a] * (m[a] + vs) + (1 - P[:, a]) * vf
    return np.argmax(Q, axis=1)


def run_paired(alpha, kappa, m, pool, tenure, seed, n_ep, H,
               rho_const, rho_ns, tenure_mode="const"):
    """Aware and myopic under identical draws; returns per-user paired stats."""
    diffs, revs_a, revs_m = [], [], []
    batch = 150
    for pol in ("aware", "myopic"):
        rng = np.random.default_rng(seed)
        revs = []
        for _ in range(n_ep // batch):
            pick = rng.integers(0, pool.shape[0], batch)
            ten = (tenure[pick] if tenure_mode == "empirical"
                   else np.full(batch, 1e9 if rho_const is not None else 20.0))
            env = Env(alpha, kappa, pool[pick].copy(), ten, rng,
                      rho_const=rho_const, rho_ns=rho_ns)
            r = np.zeros(batch)
            for t in range(H):
                if pol == "aware":
                    a = plan_aware(env.z, env.n, alpha, kappa, m, H - t,
                                   rho_const, rho_ns)
                else:
                    a = myopic(env.z, alpha, kappa, m)
                r += m[a] * env.step(a)
            revs.extend(r.tolist())
        if pol == "aware":
            revs_a = np.array(revs)
        else:
            revs_m = np.array(revs)
    D = revs_a - revs_m
    return {"aware": float(revs_a.mean()), "myopic": float(revs_m.mean()),
            "d_mean": float(D.mean()), "d_sd": float(D.std(ddof=1)),
            "n": int(len(D))}
