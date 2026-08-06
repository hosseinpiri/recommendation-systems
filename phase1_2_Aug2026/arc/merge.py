"""Merge all cell outputs into four family summaries + one master JSON."""
import glob
import json
import os
import numpy as np

OUT = os.path.expanduser("~/mind_calib/out")


def load_dir(sub, pattern):
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, sub, pattern))):
        with open(p) as f:
            rows.append(json.load(f))
    return rows


def main():
    master = {}

    # ---- profile fits
    fits = load_dir("fitcells", "cell_*.json")
    prof = {}
    for r in fits:
        key = f"u{r['universe']}|{r['kmode']}"
        prof.setdefault(key, []).append((r["rho"], r["loglik"], r["kappa"]))
    profiles = {}
    for key, pts in prof.items():
        pts.sort()
        rhos = [p[0] for p in pts]
        lls = [p[1] for p in pts]
        i = int(np.argmax(lls))
        inside = [rhos[j] for j in range(len(rhos)) if lls[j] >= lls[i] - 1.92]
        profiles[key] = {"rho": rhos, "loglik": lls,
                         "kappa": [p[2] for p in pts],
                         "rho_hat": rhos[i],
                         "LR_vs_zero": 2 * (lls[i] - lls[0]),
                         "ci_grid": [min(inside), max(inside)],
                         "n_cells": len(pts)}
    master["profiles"] = profiles

    # ---- bootstrap
    boots = load_dir("bootcells", "rep_*.json")
    if boots:
        rh = np.array([b["rho_hat"] for b in boots])
        lr = np.array([b["LR_vs_zero"] for b in boots])
        master["bootstrap"] = {
            "n_reps": len(boots),
            "rho_hat_mean": float(rh.mean()),
            "rho_hat_pct": {p: float(np.percentile(rh, p))
                            for p in (2.5, 25, 50, 75, 97.5)},
            "share_reps_LR_above_3.84": float((lr > 3.84).mean()),
        }

    # ---- simulations
    sims = load_dir("simcells", "cell_*.json")
    master["sim"] = sims
    if sims:
        mains = [s for s in sims if s["family"] == "main"]
        if mains:
            hi = [s["gap_pct_ci95_hi"] for s in mains]
            master["equivalence"] = {
                "max_gap_pct_upper95_main": float(max(hi)),
                "n_main_cells": len(mains)}

    # ---- margin sweep
    chunks = load_dir("margincells", "chunk_*.json")
    rows = [r for ch in chunks for r in ch]
    if rows:
        gaps = np.array([r["gap_pct"] for r in rows])
        order = np.argsort(gaps)[::-1]
        master["margins"] = {
            "n_vectors": len(rows),
            "max_gap_pct": float(gaps.max()),
            "p99": float(np.percentile(gaps, 99)),
            "p90": float(np.percentile(gaps, 90)),
            "median": float(np.median(gaps)),
            "top20": [rows[i] for i in order[:20]],
        }

    with open(os.path.join(OUT, "master_results.json"), "w") as f:
        json.dump(master, f, indent=2)
    print("merged:", {k: (len(v) if isinstance(v, list) else "ok")
                      for k, v in master.items()})


if __name__ == "__main__":
    main()
