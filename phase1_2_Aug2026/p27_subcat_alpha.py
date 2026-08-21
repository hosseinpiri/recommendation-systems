"""
Why are the category intercepts flat? Test the aggregation explanation:
fit the click model with SUBCATEGORY intercepts (one per subcategory with
enough data) + kappa * top-level alignment, and compare the spread of
subcategory intercepts within and between top-level categories. If
subcategory alphas are widely spread and category alphas are their
near-equal averages, the flatness is an aggregation artifact.
"""
import json
import os
import pickle
import random
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

BASE = "/Users/piri/Desktop/Recommendation Systems/Mind-Data-Large"
SCRATCH = "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SAMPLE_P = 0.05
MIN_HIST = 5
MIN_SUB_ROWS = 2000
SEED = 20260821


def main():
    # subcategory map
    sub_of = {}
    cat_of = {}
    for split in ("MINDlarge_train", "MINDlarge_dev"):
        with open(os.path.join(BASE, split, "news.tsv"), encoding="utf-8") as f:
            for line in f:
                p = line.split("\t")
                sub_of[p[0]] = p[2]
                cat_of[p[0]] = p[1]
    subs = sorted(set(sub_of.values()))
    sidx = {s: i for i, s in enumerate(subs)}
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        nc = pickle.load(f)
    nid18 = nc["nid18"] if "nid18" in nc else nc["nid_cat"]
    cat_list = nc["cat_list"]

    rng = random.Random(SEED)
    hist_cache = {}
    rows_s, rows_c, rows_a, rows_y = [], [], [], []
    with open(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
              encoding="utf-8") as f:
        for ln, line in enumerate(f):
            _, user, tstr, hist, impr = line.rstrip("\n").split("\t")
            if rng.random() >= SAMPLE_P:
                continue
            if user not in hist_cache:
                cnt = np.zeros(18)
                for h in (hist.split() if hist else []):
                    c = nid18.get(h)
                    if c is not None:
                        cnt[c] += 1
                n = cnt.sum()
                hist_cache[user] = cnt / n if n >= MIN_HIST else None
            z = hist_cache[user]
            if z is None:
                continue
            for tok in impr.split():
                nid, lab = tok.rsplit("-", 1)
                c = nid18.get(nid)
                s = sidx.get(sub_of.get(nid))
                if c is None or s is None:
                    continue
                rows_s.append(s)
                rows_c.append(c)
                rows_a.append(float(z[c]))
                rows_y.append(int(lab))
            if ln % 400000 == 0:
                print(f"line {ln}: {len(rows_y)} rows", flush=True)

    S = np.array(rows_s)
    C = np.array(rows_c)
    A = np.array(rows_a)
    Y = np.array(rows_y)
    cnt = np.bincount(S, minlength=len(subs))
    keep = np.where(cnt >= MIN_SUB_ROWS)[0]
    m = np.isin(S, keep)
    S, C, A, Y = S[m], C[m], A[m], Y[m]
    remap = {int(s): j for j, s in enumerate(keep)}
    R = len(Y)
    X = sp.hstack([
        sp.csr_matrix((np.ones(R), (np.arange(R), [remap[s] for s in S])),
                      shape=(R, len(keep))),
        sp.csr_matrix(A[:, None])], format="csr")
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=3000,
                             fit_intercept=False, tol=1e-7)
    clf.fit(X, Y)
    asub = clf.coef_[0][:len(keep)]
    kappa = float(clf.coef_[0][-1])

    sub_names = [subs[s] for s in keep]
    sub_cat = []
    for s in keep:
        nids = [n for n, ss in sub_of.items() if sidx.get(ss) == s]
        sub_cat.append(cat_of[nids[0]] if nids else "?")
    # spread decomposition
    total_spread = float(asub.max() - asub.min())
    p595 = float(np.percentile(asub, 95) - np.percentile(asub, 5))
    within = {}
    cat_means = {}
    for cat in sorted(set(sub_cat)):
        vals = asub[[i for i, cc in enumerate(sub_cat) if cc == cat]]
        if len(vals) >= 3:
            within[cat] = {"n_subcats": len(vals),
                           "spread": float(vals.max() - vals.min()),
                           "sd": float(vals.std())}
        cat_means[cat] = float(vals.mean())
    between_spread = float(max(cat_means.values()) - min(cat_means.values()))
    out = {"n_rows": int(R), "n_subcats": len(keep), "kappa": kappa,
           "subcat_alpha_total_spread": total_spread,
           "subcat_alpha_p5_p95": p595,
           "between_category_mean_spread": between_spread,
           "within_category": within,
           "category_means": cat_means,
           "top10": sorted(zip(sub_names, np.round(asub, 3).tolist()),
                           key=lambda t: -t[1])[:10],
           "bottom10": sorted(zip(sub_names, np.round(asub, 3).tolist()),
                              key=lambda t: t[1])[:10]}
    with open(os.path.join(OUT, "p27_subcat_alpha.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"subcats {len(keep)}, kappa {kappa:.3f}, total spread "
          f"{total_spread:.2f}, p5-p95 {p595:.2f}, between-cat "
          f"{between_spread:.2f}")


if __name__ == "__main__":
    main()
