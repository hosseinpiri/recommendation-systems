"""
Xinyuan's 'are we forcing the data to fit our model' concern: a
semi-synthetic recovery test of the estimation stack. Take the REAL design
(the sampled impression-item rows: category, alignment), regenerate the
click labels from known synthetic truths, run the exact production
estimation (logit with category effects + kappa * alignment), and check
whether the pipeline recovers spread intercepts and large kappa, or whether
it flattens them toward the MIND-like near-uniform pattern.

Truths tested: (alpha spread x1 as estimated | x3 wide | x6 very wide) x
(kappa 1.966 | 4 | 6). If recovery is faithful everywhere, the flat MIND
estimates reflect the data, not the pipeline.
"""
import json
import os
import numpy as np
import statsmodels.api as sm

SCRATCH = "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
N_ROWS = 2_000_000
SEED = 20260820


def main():
    d = np.load(os.path.join(SCRATCH, "impr_1a.npz"))
    cat, align = d["cat"].astype(int), d["align"].astype(float)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(cat), min(N_ROWS, len(cat)), replace=False)
    cat, align = cat[idx], align[idx]
    counts = np.bincount(cat, minlength=18)
    keep = np.where(counts >= 200)[0]
    m = np.isin(cat, keep)
    cat, align = cat[m], align[m]
    remap = {c: j for j, c in enumerate(keep)}
    K = len(keep)
    X = np.zeros((len(cat), K + 1))
    X[np.arange(len(cat)), [remap[c] for c in cat]] = 1.0
    X[:, K] = align

    base_alpha = np.full(18, -3.4)
    est_alpha = np.array([-3.702, -3.380, -3.407, -3.448, -3.684, -3.567,
                          -3.716, -3.719, -3.494, -3.124, -2.869, -2.973,
                          -2.982, -3.442, -3.4, -3.4, -3.4, -3.4])
    abar = est_alpha.mean()
    out = {}
    for spread in (1.0, 3.0, 6.0):
        for kap in (1.966, 4.0, 6.0):
            atrue = abar + spread * (est_alpha - abar)
            # recentre so overall click rate stays plausible
            # recentre so the overall click rate matches MIND (~4%)
            atrue2 = atrue.copy()
            for _ in range(40):
                u = atrue2[cat] + kap * align
                cur = (1 / (1 + np.exp(-u))).mean()
                atrue2 = atrue2 + np.log(0.0407 / cur)
                if abs(cur - 0.0407) < 1e-5:
                    break
            u = atrue2[cat] + kap * align
            p = 1 / (1 + np.exp(-u))
            y = (rng.random(len(p)) < p).astype(int)
            res = sm.Logit(y, X).fit(disp=0, maxiter=300)
            ahat = res.params[:K]
            khat = float(res.params[K])
            true_sub = atrue2[keep]
            spread_true = float(true_sub.max() - true_sub.min())
            spread_hat = float(ahat.max() - ahat.min())
            corr = float(np.corrcoef(true_sub, ahat)[0, 1])
            key = f"spread={spread}|kappa={kap}"
            out[key] = {"kappa_true": kap, "kappa_hat": khat,
                        "alpha_spread_true": spread_true,
                        "alpha_spread_hat": spread_hat,
                        "alpha_corr": corr,
                        "click_rate": float(y.mean())}
            print(f"{key}: kappa {kap} -> {khat:.3f}; spread "
                  f"{spread_true:.2f} -> {spread_hat:.2f}; corr {corr:.4f}",
                  flush=True)
    with open(os.path.join(OUT, "p24_modelfit_test.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved p24_modelfit_test.json")


if __name__ == "__main__":
    main()
