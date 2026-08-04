"""
Dense article features for the bridge-geometry analysis.

TF-IDF on title + abstract over the union of train and dev news, TruncatedSVD
to d = 50, unit-normalized rows. Saves the vectors, the news-id index, and the
cosine (Gram) matrix of the 18 category centroids.
"""
import json
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

BASE = "/Users/piri/Desktop/Recommendation Systems/Mind-Data-Large"
SCRATCH = os.environ.get(
    "SCRATCH",
    "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind",
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
D = 50
SEED = 3


def main():
    text = {}
    cat = {}
    for split in ("MINDlarge_train", "MINDlarge_dev"):
        with open(os.path.join(BASE, split, "news.tsv"), encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                nid = p[0]
                if nid not in text:
                    cat[nid] = p[1]
                    text[nid] = (p[3] + " " + (p[4] or "")).strip()
    ids = sorted(text)
    print(f"{len(ids)} unique articles")

    vec = TfidfVectorizer(max_features=60000, stop_words="english",
                          sublinear_tf=True, min_df=3)
    X = vec.fit_transform(text[n] for n in ids)
    svd = TruncatedSVD(n_components=D, random_state=SEED)
    E = svd.fit_transform(X)
    norms = np.linalg.norm(E, axis=1)
    zero = norms < 1e-10
    E[~zero] /= norms[~zero, None]
    print(f"SVD explained variance: {svd.explained_variance_ratio_.sum():.3f}; "
          f"{int(zero.sum())} zero-norm articles")

    idx = {n: i for i, n in enumerate(ids)}
    np.save(os.path.join(SCRATCH, "news_vec.npy"), E.astype(np.float32))
    with open(os.path.join(SCRATCH, "news_vec_idx.pkl"), "wb") as f:
        pickle.dump({"idx": idx, "cat": cat}, f)

    # category centroids (unit-normalized) and their Gram matrix
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        cat_list = pickle.load(f)["cat_list"]
    cent = np.zeros((len(cat_list), D))
    for j, c in enumerate(cat_list):
        rows = [idx[n] for n in ids if cat[n] == c and not zero[idx[n]]]
        if rows:
            m = E[rows].mean(axis=0)
            cent[j] = m / (np.linalg.norm(m) + 1e-12)
    np.save(os.path.join(SCRATCH, "cat_centroids.npy"), cent.astype(np.float32))
    G = cent @ cent.T
    sim_items = ["news", "lifestyle", "sports", "finance", "foodanddrink",
                 "entertainment", "travel", "health"]
    sub = [cat_list.index(c) for c in sim_items]
    Gs = G[np.ix_(sub, sub)]
    off = Gs[~np.eye(len(sub), dtype=bool)]
    out = {
        "d": D,
        "n_articles": len(ids),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "sim_items": sim_items,
        "gram_sim_items": np.round(Gs, 3).tolist(),
        "offdiag_mean": float(off.mean()),
        "offdiag_min": float(off.min()),
        "offdiag_max": float(off.max()),
    }
    with open(os.path.join(OUT, "p5_embeddings.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("centroid Gram off-diagonal: mean", round(out["offdiag_mean"], 3),
          "range", round(out["offdiag_min"], 3), "to", round(out["offdiag_max"], 3))


if __name__ == "__main__":
    main()
