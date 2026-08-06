"""Shared fitting utilities for profile-grid and bootstrap cells."""
import os
import pickle
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

OUT = os.path.expanduser("~/mind_calib/out")
PREP = os.path.join(OUT, "prep")


def load_prep():
    with open(os.path.join(PREP, "news_cat.pkl"), "rb") as f:
        nc = pickle.load(f)
    with open(os.path.join(PREP, "states.pkl"), "rb") as f:
        states = pickle.load(f)
    with open(os.path.join(PREP, "train_clicks.pkl"), "rb") as f:
        clicks = pickle.load(f)
    with open(os.path.join(PREP, "dev_rows.pkl"), "rb") as f:
        dev = pickle.load(f)
    return nc, states, clicks, dev


def cat18_to9(nc):
    m = np.full(18, 8, dtype=int)
    for j, c in enumerate(nc["cat_list"]):
        if c in nc["items8"]:
            m[j] = nc["items8"].index(c)
    return m


def user_sample(dev, cap, seed):
    users = sorted(dev.keys())
    if cap and len(users) > cap:
        rng = np.random.default_rng(seed)
        users = sorted(rng.choice(users, cap, replace=False).tolist())
    return users


def z_rolled(states, clicks, users, rho, universe, map9):
    """Final state per user after constant-rho updates through train clicks."""
    dim = 18 if universe == 18 else 9
    Z = np.zeros((len(users), dim))
    for k, u in enumerate(users):
        z = (states[u][0] if universe == 18 else states[u][1]).astype(np.float64).copy()
        if rho > 0:
            for _, a in clicks.get(u, []):
                c = a if universe == 18 else map9[a]
                z *= (1 - rho)
                z[c] += rho
        Z[k] = z
    return Z


def build_rows(dev, users, universe, map9):
    cats, ys, uidx = [], [], []
    for k, u in enumerate(users):
        cs, yy = dev[u]
        cats.append(cs.astype(int))
        ys.append(yy.astype(int))
        uidx.append(np.full(len(cs), k))
    cats = np.concatenate(cats)
    ys = np.concatenate(ys)
    uidx = np.concatenate(uidx)
    ecat = cats if universe == 18 else map9[cats]
    # intercept categories: those with enough rows
    cnt = np.bincount(ecat, minlength=18 if universe == 18 else 9)
    keepc = np.where(cnt >= 500)[0]
    m = np.isin(ecat, keepc)
    return cats[m], ecat[m], ys[m], uidx[m], keepc


def max_loglik(ecat, ys, aligns, keepc, kappa_fixed=None):
    remap = {int(c): j for j, c in enumerate(keepc)}
    R = len(ecat)
    onehot = sp.csr_matrix((np.ones(R), (np.arange(R),
                            [remap[c] for c in ecat])), shape=(R, len(keepc)))
    if kappa_fixed is None:
        X = sp.hstack([onehot, sp.csr_matrix(aligns[:, None])], format="csr")
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000,
                                 fit_intercept=False, tol=1e-7)
        clf.fit(X, ys)
        u = X @ clf.coef_[0]
        kappa = float(clf.coef_[0][-1])
    else:
        # alpha-only fit with kappa*align as offset (IRLS on sparse one-hot:
        # separable per category -> closed-form-ish via per-category Newton)
        kappa = float(kappa_fixed)
        off = kappa * aligns
        alpha = np.zeros(len(keepc))
        idx = np.array([remap[c] for c in ecat])
        for _ in range(60):
            p = 1 / (1 + np.exp(-(alpha[idx] + off)))
            g = np.bincount(idx, weights=ys - p, minlength=len(keepc))
            h = np.bincount(idx, weights=p * (1 - p), minlength=len(keepc))
            step = g / np.maximum(h, 1e-9)
            alpha += step
            if np.abs(step).max() < 1e-9:
                break
        u = alpha[idx] + off
    p = 1 / (1 + np.exp(-u))
    eps = 1e-12
    ll = float(np.sum(ys * np.log(p + eps) + (1 - ys) * np.log(1 - p + eps)))
    return ll, kappa
