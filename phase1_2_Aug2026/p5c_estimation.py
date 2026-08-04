"""
Dense-feature and slate-aware click models on the sampled impressions.

Models (same rows throughout, category effects for categories with >= 200 rows):
  A  binary logit: cat effects + kappa_cat * cat_align        (baseline replica)
  B  binary logit: cat effects + kappa_d * dense_align
  C  binary logit: B + position-bucket dummies
  D  slate conditional logit with outside option, single-click impressions:
     P(click k | S) = exp(u_k) / (1 + sum_j exp(u_j)),
     u_i = alpha_cat(i) + kappa * align_i + gamma_posbucket(i)
     (fit twice: category alignment and dense alignment)

Alignment scales differ (cat_align in [0,1], dense_align roughly [-1,1]), so
each kappa is also reported per standard deviation of its regressor.
"""
import json
import os
import numpy as np
from scipy.optimize import minimize
import statsmodels.api as sm

SCRATCH = os.environ.get(
    "SCRATCH",
    "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind",
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
FIT_N = 2_500_000
SEED = 21
POS_EDGES = [0, 1, 2, 3, 4, 5, 10, 20, 30]   # buckets: [0],[1],[2],[3],[4],[5-9],[10-19],[20+]


def pos_bucket(pos):
    return np.searchsorted(POS_EDGES, pos, side="right") - 1


def binary_fit(X, y, label):
    res = sm.Logit(y, X).fit(disp=0, maxiter=200)
    print(f"{label}: fitted, llf/n = {res.llf/len(y):.5f}")
    return res


def main():
    d = np.load(os.path.join(SCRATCH, "slate_rows.npz"))
    cat, pos, ca, da, y = (d["cat"].astype(int), d["pos"].astype(int),
                           d["cat_align"].astype(float),
                           d["dense_align"].astype(float), d["y"].astype(int))
    impr, nclick = d["impr"].astype(int), d["nclick"].astype(int)
    n = len(y)
    print(f"{n} rows, {impr.max()+1} impressions, click rate {y.mean():.4f}")

    counts = np.bincount(cat)
    keep = np.where(counts >= 200)[0]
    remap = {c: j for j, c in enumerate(keep)}
    kmask = np.isin(cat, keep)

    rng = np.random.default_rng(SEED)
    idx_all = np.where(kmask)[0]
    fit_idx = rng.choice(idx_all, min(FIT_N, len(idx_all)), replace=False)

    nc = len(keep)
    npb = len(POS_EDGES) - 1
    pb = pos_bucket(pos)

    def design(idx, align, with_pos):
        cols = nc + 1 + (npb - 1 if with_pos else 0)
        X = np.zeros((len(idx), cols))
        X[np.arange(len(idx)), [remap[c] for c in cat[idx]]] = 1.0
        X[:, nc] = align[idx]
        if with_pos:
            for r, b in enumerate(pb[idx]):
                if b > 0:
                    X[r, nc + b] = 1.0   # bucket 0 is the base
        return X

    results = {}
    sd_ca, sd_da = float(ca[fit_idx].std()), float(da[fit_idx].std())
    yf = y[fit_idx]

    resA = binary_fit(design(fit_idx, ca, False), yf, "A cat_align")
    resB = binary_fit(design(fit_idx, da, False), yf, "B dense_align")
    resC = binary_fit(design(fit_idx, da, True), yf, "C dense+pos")
    resCa = binary_fit(design(fit_idx, ca, True), yf, "C' cat+pos")
    for name, res, sd in [("A_cat", resA, sd_ca), ("B_dense", resB, sd_da),
                          ("C_dense_pos", resC, sd_da), ("C_cat_pos", resCa, sd_ca)]:
        k, se = float(res.params[nc]), float(res.bse[nc])
        results[name] = {"kappa": k, "se": se, "kappa_per_sd": k * sd,
                         "llf_per_row": float(res.llf / len(yf))}
    results["C_dense_pos"]["pos_effects"] = {
        f"bucket{b}": float(resC.params[nc + b]) for b in range(1, npb)}
    results["align_sd"] = {"cat": sd_ca, "dense": sd_da}

    # ---- D: conditional logit on single-click impressions
    one = nclick == 1
    use = one & kmask
    # keep only impressions fully inside the kept-category mask
    bad_impr = np.unique(impr[~kmask])
    use &= ~np.isin(impr, bad_impr)
    ui, ustart = np.unique(impr[use], return_index=True)
    print(f"conditional logit: {len(ui)} single-click impressions, "
          f"{int(use.sum())} rows")

    idxD = np.where(use)[0]
    catD = np.array([remap[c] for c in cat[idxD]])
    pbD = pb[idxD]
    yD = y[idxD]
    imprD = impr[idxD]
    # contiguous group offsets
    _, starts = np.unique(imprD, return_index=True)
    starts = np.sort(starts)

    def make_nll(align):
        aD = align[idxD]

        def nll_grad(theta):
            alpha = theta[:nc]
            kap = theta[nc]
            gam = np.concatenate([[0.0], theta[nc + 1: nc + npb]])
            u = alpha[catD] + kap * aD + gam[pbD]
            eu = np.exp(u - 0)          # outside option utility 0
            denom_g = np.add.reduceat(eu, starts)
            denom = 1.0 + denom_g
            # log-lik: sum over impressions [u_click - log denom]
            ll = float(u[yD == 1].sum() - np.log(denom).sum())
            # gradients
            w = eu / np.repeat(denom, np.diff(np.append(starts, len(eu))))
            resid = yD - w                     # d ll / d u_i
            g = np.zeros_like(theta)
            np.add.at(g, catD, resid)
            g[nc] = float((resid * aD).sum())
            gpos = np.zeros(npb)
            np.add.at(gpos, pbD, resid)
            g[nc + 1: nc + npb] = gpos[1:]
            return -ll, -g

        return nll_grad

    for tag, align, sd in [("D_slate_cat", ca, sd_ca), ("D_slate_dense", da, sd_da)]:
        f = make_nll(align)
        x0 = np.zeros(nc + npb)
        x0[:nc] = -3.0
        opt = minimize(f, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500})
        kap = float(opt.x[nc])
        # numerical Hessian for kappa SE (central differences on gradient)
        eps = 1e-4
        e = np.zeros_like(opt.x); e[nc] = eps
        gp = f(opt.x + e)[1][nc]
        gm = f(opt.x - e)[1][nc]
        h = (gp - gm) / (2 * eps)
        se = float(1.0 / np.sqrt(max(h, 1e-12)))
        results[tag] = {"kappa": kap, "se_approx": se, "kappa_per_sd": kap * sd,
                        "converged": bool(opt.success), "nll": float(opt.fun),
                        "n_impressions": int(len(starts))}
        print(f"{tag}: kappa={kap:.3f} (se~{se:.3f}), per-sd {kap*sd:.3f}")

    results["n_fit_binary"] = int(len(fit_idx))
    results["share_single_click"] = float(one.mean())
    with open(os.path.join(OUT, "p5_slate_dense.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print("saved p5_slate_dense.json")


if __name__ == "__main__":
    main()
