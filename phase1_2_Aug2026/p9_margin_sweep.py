"""
Margin-space sweep: is there ANY margin configuration under which cultivation-
aware planning beats reactive myopia at calibrated (alpha, kappa, rho)?

Draws random log-normal margin vectors at three dispersion levels plus
adversarial constructions (margins rank-reversed against alpha and against
popularity), normalizes each to mean 1, and computes the aware-minus-myopic
revenue gap under common random numbers (category one-hot geometry, 8 items,
H = 30). The sweep runs at rho = 0.15 (where steering is most plausible);
the top-gap vectors are re-run at rho_hat.
"""
import json
import os
import numpy as np

import p2_simulation as S

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
N_VEC = 90
SEEDS = [11, 12]
RHO_MAIN = 0.15
S.N_EP, S.BATCH = 1500, 150


def gap(alpha, kappa, rho, pool, m):
    gaps = []
    for s in SEEDS:
        a = S.run_policy("oracle-aware", alpha, kappa, rho, pool, 5000 + s, m)
        b = S.run_policy("oracle-blind", alpha, kappa, rho, pool, 5000 + s, m)
        gaps.append(100 * (a["mean_revenue"] - b["mean_revenue"])
                    / b["mean_revenue"])
    return float(np.mean(gaps)), float(np.std(gaps))


def main():
    item_names, alpha, kappa, rho_hat, pool = S.load_calibration()
    popularity = pool.mean(axis=0)
    rng = np.random.default_rng(99)

    vectors = []
    for sig in (0.3, 0.6, 1.0):
        for _ in range(N_VEC // 3):
            g = rng.standard_normal(len(alpha))
            m = np.exp(sig * g)
            vectors.append(m / m.mean())
    # adversarial: reward the categories users are least drawn to
    for base in (alpha, popularity):
        order = np.argsort(base)          # ascending attractiveness
        m = np.ones(len(alpha))
        m[order] = np.linspace(4.0, 0.5, len(alpha))   # worst gets most margin
        vectors.append(m / m.mean())
    # margins proportional to inverse popularity
    m = 1.0 / np.clip(popularity, 1e-3, None)
    vectors.append(m / m.mean())

    rows = []
    for i, m in enumerate(vectors):
        g, gse = gap(alpha, kappa, RHO_MAIN, pool, m)
        rows.append({
            "gap_pct": g, "gap_se": gse,
            "dispersion": float(np.std(np.log(m))),
            "corr_alpha": float(np.corrcoef(m, alpha)[0, 1]),
            "corr_pop": float(np.corrcoef(m, popularity)[0, 1]),
            "margins": np.round(m, 3).tolist(),
        })
        if i % 10 == 0:
            print(f"{i}/{len(vectors)}: gap {g:.2f}% "
                  f"(disp {rows[-1]['dispersion']:.2f})", flush=True)

    gaps = np.array([r["gap_pct"] for r in rows])
    top = np.argsort(gaps)[-10:][::-1]
    for j in top:
        g35, _ = gap(alpha, kappa, rho_hat, pool, np.array(rows[j]["margins"]))
        rows[j]["gap_pct_at_rho_hat"] = g35

    out = {
        "n_vectors": len(vectors), "rho_main": RHO_MAIN, "rho_hat": rho_hat,
        "H": S.H, "n_ep": S.N_EP, "seeds": SEEDS,
        "max_gap_pct": float(gaps.max()), "median_gap_pct": float(np.median(gaps)),
        "p90_gap_pct": float(np.percentile(gaps, 90)),
        "corr_gap_vs_corr_alpha": float(np.corrcoef(
            gaps, [r["corr_alpha"] for r in rows])[0, 1]),
        "corr_gap_vs_dispersion": float(np.corrcoef(
            gaps, [r["dispersion"] for r in rows])[0, 1]),
        "rows": rows,
    }
    with open(os.path.join(OUT, "p9_margin_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"max gap {out['max_gap_pct']:.2f}%, median {out['median_gap_pct']:.2f}%, "
          f"p90 {out['p90_gap_pct']:.2f}%")


if __name__ == "__main__":
    main()
