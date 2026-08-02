"""
Phase 1b: structural estimation of the taste transition

    click:             z <- (1 - rho) z + rho e_{cat}         (model's assumption)
    shown-not-clicked: z <- z + rho_ns (e_{cat} - z)          (should be 0 under the model)

Identification is temporal, not mechanical:
  z0        = history distribution (clicks BEFORE the log window),
  z_T(rho)  = z0 updated through the user's TRAIN-period clicks (Nov 9-14),
  fit/eval  = logit of DEV-period impressions (Nov 15) with alignment z_T(rho)[cat].

For each rho on a grid we rebuild z_T(rho), refit (alpha, kappa) by ML on the dev rows,
and record the maximized log likelihood: a profile likelihood over rho. rho = 0 is the
nested null (tastes never move). 95% CI by likelihood-ratio inversion (chi2_1).

Pass 2: coarse 2-D grid over (rho, rho_ns) on a user subsample, testing
no-movement-without-consumption (and the sign of any shown-not-clicked effect).
Pass 3: nonlinearity check, constant-rho update vs diminishing-weight running mean.
"""
import json
import os
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

SCRATCH = os.environ.get(
    "SCRATCH",
    "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind",
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT, exist_ok=True)

MAX_USERS = 120_000
MAX_USERS_2D = 40_000
SEED = 11
RHO_GRID = np.round(np.arange(0.0, 0.42, 0.02), 3)
RHO_NS_GRID = [-0.04, -0.02, -0.01, 0.0, 0.01, 0.02, 0.04]
RHO_COARSE = [0.0, 0.1, 0.2, 0.3]
N0_GRID = [2, 5, 10, 20, 40, 80]


def load():
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        cat_list = pickle.load(f)["cat_list"]
    with open(os.path.join(SCRATCH, "user_hist.pkl"), "rb") as f:
        hist = pickle.load(f)
    with open(os.path.join(SCRATCH, "train_clicks.pkl"), "rb") as f:
        clicks = pickle.load(f)
    with open(os.path.join(SCRATCH, "train_shown.pkl"), "rb") as f:
        shown = pickle.load(f)
    with open(os.path.join(SCRATCH, "dev_rows.pkl"), "rb") as f:
        dev = pickle.load(f)
    users = sorted(dev.keys())
    rng = np.random.default_rng(SEED)
    if len(users) > MAX_USERS:
        users = sorted(rng.choice(users, MAX_USERS, replace=False).tolist())
    return cat_list, hist, clicks, shown, dev, users


def stack_events(users, clicks, shown):
    ev = {}
    for u in users:
        e = [(t, c, 1) for (t, c) in clicks.get(u, [])]
        e += [(t, c, 0) for (t, c) in shown.get(u, [])]
        e.sort()
        ev[u] = e
    return ev


def z_final(z0, events, rho, rho_ns):
    z = z0.astype(np.float64).copy()
    for _, c, isclick in events:
        if isclick:
            if rho != 0.0:
                z *= (1.0 - rho)
                z[c] += rho
        elif rho_ns != 0.0:
            z *= (1.0 - rho_ns)
            z[c] += rho_ns
    if rho_ns != 0.0:  # negative rho_ns can leave the simplex; clip + renorm
        np.clip(z, 0.0, None, out=z)
        s = z.sum()
        if s > 0:
            z /= s
    return z


class Design:
    """Fixed dev rows for a user set; only the alignment column changes with rho."""

    def __init__(self, users, hist, ev, dev, keep_cats, remap):
        self.users, self.hist, self.ev = users, hist, ev
        cats, ys, uidx = [], [], []
        for k, u in enumerate(users):
            cs, yy = dev[u]
            m = np.isin(cs, keep_cats)
            cs, yy = cs[m].astype(int), yy[m].astype(int)
            cats.append(cs)
            ys.append(yy)
            uidx.append(np.full(len(cs), k))
        self.cats = np.concatenate(cats)
        self.y = np.concatenate(ys)
        self.uidx = np.concatenate(uidx)
        R = len(self.cats)
        cols = np.array([remap[c] for c in self.cats])
        self.onehot = sp.csr_matrix(
            (np.ones(R), (np.arange(R), cols)), shape=(R, len(remap)))
        print(f"design: {len(users)} users, {R} dev rows")

    def aligns(self, rho, rho_ns):
        zmat = np.stack([z_final(self.hist[u], self.ev[u], rho, rho_ns)
                         for u in self.users])
        return zmat[self.uidx, self.cats]

    def aligns_runmean(self, n0):
        def zf(u):
            cnt = self.hist[u].astype(np.float64) * n0
            n = float(n0)
            for _, c, isclick in self.ev[u]:
                if isclick:
                    cnt[c] += 1.0
                    n += 1.0
            return cnt / n
        zmat = np.stack([zf(u) for u in self.users])
        return zmat[self.uidx, self.cats]

    def maxll(self, align_col):
        X = sp.hstack([self.onehot, sp.csr_matrix(align_col[:, None])], format="csr")
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000,
                                 fit_intercept=False, tol=1e-7)
        clf.fit(X, self.y)
        u = X @ clf.coef_[0]
        p = 1.0 / (1.0 + np.exp(-u))
        eps = 1e-12
        ll = float(np.sum(self.y * np.log(p + eps) + (1 - self.y) * np.log(1 - p + eps)))
        return ll, float(clf.coef_[0][-1])


def main():
    cat_list, hist, clicks, shown, dev, users = load()
    ev = stack_events(users, clicks, shown)
    n_clicks = sum(len(clicks.get(u, [])) for u in users)
    print(f"{len(users)} users, {n_clicks} train clicks")

    all_cats = np.concatenate([dev[u][0] for u in users])
    cnt = np.bincount(all_cats, minlength=len(cat_list))
    keep_cats = np.where(cnt >= 500)[0]
    remap = {int(c): j for j, c in enumerate(keep_cats)}
    D = Design(users, hist, ev, dev, keep_cats, remap)

    # ---- 1-D profile over rho (rho_ns = 0)
    profile, kappas = [], []
    for rho in RHO_GRID:
        ll, kap = D.maxll(D.aligns(float(rho), 0.0))
        profile.append(ll)
        kappas.append(kap)
        print(f"rho={rho:.2f}  logL={ll:.1f}  kappa={kap:.3f}", flush=True)
    profile = np.array(profile)
    i_hat = int(np.argmax(profile))
    rho_hat = float(RHO_GRID[i_hat])
    ll_hat, ll0 = profile[i_hat], profile[0]
    lr_rho = 2 * (ll_hat - ll0)
    ci = RHO_GRID[profile >= ll_hat - 1.92]
    rho_ci = [float(ci.min()), float(ci.max())]

    # kappa SE at rho_hat via statsmodels on the same rows
    import statsmodels.api as sm
    al = D.aligns(rho_hat, 0.0)
    X = sp.hstack([D.onehot, sp.csr_matrix(al[:, None])], format="csr").toarray()
    res = sm.Logit(D.y, X).fit(disp=0, maxiter=200)
    kappa_dev, kappa_dev_se = float(res.params[-1]), float(res.bse[-1])

    # ---- 2-D coarse profile on a subsample
    rng = np.random.default_rng(SEED + 1)
    users2 = sorted(rng.choice(users, MAX_USERS_2D, replace=False).tolist())
    D2 = Design(users2, hist, ev, dev, keep_cats, remap)
    grid2 = np.zeros((len(RHO_COARSE), len(RHO_NS_GRID)))
    for a, rho in enumerate(RHO_COARSE):
        for b, rns in enumerate(RHO_NS_GRID):
            grid2[a, b], _ = D2.maxll(D2.aligns(float(rho), float(rns)))
        print(f"2D row rho={rho:.2f} best rho_ns="
              f"{RHO_NS_GRID[int(np.argmax(grid2[a]))]}", flush=True)
    a_hat, b_hat = np.unravel_index(np.argmax(grid2), grid2.shape)
    lr_ns = 2 * (grid2[a_hat, b_hat] - grid2[a_hat, RHO_NS_GRID.index(0.0)])

    # ---- nonlinearity: running-mean alternative (uses same D, full sample)
    lls_mean = []
    for n0 in N0_GRID:
        ll, _ = D.maxll(D.aligns_runmean(float(n0)))
        lls_mean.append(ll)
        print(f"running-mean n0={n0}  logL={ll:.1f}", flush=True)

    out = {
        "n_users": len(users),
        "n_dev_rows": int(len(D.y)),
        "n_train_clicks": int(n_clicks),
        "rho_grid": RHO_GRID.tolist(),
        "profile_loglik": profile.tolist(),
        "profile_kappa": kappas,
        "rho_hat": rho_hat,
        "rho_ci95_profileLR": rho_ci,
        "LR_rho_vs_zero": float(lr_rho),
        "kappa_at_rho_hat_dev": kappa_dev,
        "kappa_at_rho_hat_dev_se": kappa_dev_se,
        "grid2_rho": RHO_COARSE,
        "grid2_rho_ns": RHO_NS_GRID,
        "grid2_loglik": grid2.tolist(),
        "rho_ns_hat": float(RHO_NS_GRID[b_hat]),
        "rho_at_2d_max": float(RHO_COARSE[a_hat]),
        "LR_rho_ns_vs_zero": float(lr_ns),
        "runmean_n0_grid": N0_GRID,
        "runmean_loglik": lls_mean,
        "best_runmean_vs_best_const_rho": float(max(lls_mean) - ll_hat),
    }
    with open(os.path.join(OUT, "p1b_transition.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if not isinstance(v, list)}, indent=2))


if __name__ == "__main__":
    main()
