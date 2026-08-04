"""
Impression-level prep for the dense-feature and slate-aware click models.

Pass over MINDlarge train behaviors, sampling whole impressions. Per item row:
category, position within the impression, category alignment z_cat[cat], dense
alignment z_dense . x, click label, impression index, impression click count.
z_cat and z_dense are built from the pre-window history field only.

Also stores, for a 30k-user subsample with dev rows, the pieces needed for a
coarse dense-rho profile: history dense vector, train-window clicked article
vector indices (time-sorted), and dev impression item vector indices + labels
+ categories.
"""
import os
import pickle
import random
import numpy as np

BASE = "/Users/piri/Desktop/Recommendation Systems/Mind-Data-Large"
SCRATCH = os.environ.get(
    "SCRATCH",
    "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind",
)
IMPR_SAMPLE_P = 0.06
MIN_HIST = 5
RHO_USERS = 30000
SEED = 20260804

rng = random.Random(SEED)


def parse_time(s):
    date, clock, ampm = s.split(" ")
    _, day, _ = date.split("/")
    hh, mm, ss = clock.split(":")
    h = int(hh) % 12
    if ampm == "PM":
        h += 12
    return int(day) * 86400 + h * 3600 + int(mm) * 60 + int(ss)


def main():
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        nc = pickle.load(f)
    nid_cat, cat_list = nc["nid_cat"], nc["cat_list"]
    K = len(cat_list)
    E = np.load(os.path.join(SCRATCH, "news_vec.npy"))
    with open(os.path.join(SCRATCH, "news_vec_idx.pkl"), "rb") as f:
        vidx = pickle.load(f)["idx"]

    z_cache = {}

    def user_states(hist_ids):
        cnt = np.zeros(K)
        vecs = []
        for h in hist_ids:
            c = nid_cat.get(h)
            if c is not None:
                cnt[c] += 1
            j = vidx.get(h)
            if j is not None:
                vecs.append(E[j])
        n = cnt.sum()
        if n < MIN_HIST:
            return None
        zc = cnt / n
        zd = np.mean(vecs, axis=0) if vecs else np.zeros(E.shape[1])
        nz = np.linalg.norm(zd)
        if nz > 1e-9:
            zd = zd / nz
        return zc, zd.astype(np.float32)

    rows = {k: [] for k in ("cat", "pos", "cat_align", "dense_align", "y",
                            "impr", "nclick", "slate_len")}
    n_impr = 0
    with open(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
              encoding="utf-8") as f:
        for ln, line in enumerate(f):
            _, user, tstr, hist, impr = line.rstrip("\n").split("\t")
            if rng.random() >= IMPR_SAMPLE_P:
                continue
            if user not in z_cache:
                z_cache[user] = user_states(hist.split() if hist else [])
            st = z_cache[user]
            if st is None:
                continue
            zc, zd = st
            toks = impr.split()
            items = []
            for pos, tok in enumerate(toks):
                nid, lab = tok.rsplit("-", 1)
                c = nid_cat.get(nid)
                j = vidx.get(nid)
                if c is None or j is None:
                    continue
                items.append((c, pos, float(zc[c]), float(zd @ E[j]), int(lab)))
            if not items:
                continue
            nclick = sum(it[4] for it in items)
            for c, pos, ca, da, y in items:
                rows["cat"].append(c)
                rows["pos"].append(min(pos, 29))
                rows["cat_align"].append(ca)
                rows["dense_align"].append(da)
                rows["y"].append(y)
                rows["impr"].append(n_impr)
                rows["nclick"].append(nclick)
                rows["slate_len"].append(len(items))
            n_impr += 1
            if ln % 400000 == 0:
                print(f"line {ln}: {n_impr} impressions, {len(rows['y'])} rows",
                      flush=True)

    np.savez_compressed(
        os.path.join(SCRATCH, "slate_rows.npz"),
        cat=np.array(rows["cat"], dtype=np.int16),
        pos=np.array(rows["pos"], dtype=np.int16),
        cat_align=np.array(rows["cat_align"], dtype=np.float32),
        dense_align=np.array(rows["dense_align"], dtype=np.float32),
        y=np.array(rows["y"], dtype=np.int8),
        impr=np.array(rows["impr"], dtype=np.int64),
        nclick=np.array(rows["nclick"], dtype=np.int16),
        slate_len=np.array(rows["slate_len"], dtype=np.int16),
    )
    print(f"saved slate_rows: {n_impr} impressions, {len(rows['y'])} rows, "
          f"click rate {np.mean(rows['y']):.4f}")

    # ---- dense-rho profile pieces (users with history + dev rows)
    with open(os.path.join(SCRATCH, "user_hist.pkl"), "rb") as f:
        hist_users = set(pickle.load(f).keys())
    with open(os.path.join(SCRATCH, "dev_rows.pkl"), "rb") as f:
        dev_users = set(pickle.load(f).keys())
    keep = sorted(hist_users & dev_users)
    rng2 = np.random.default_rng(SEED + 1)
    keep = set(rng2.choice(keep, min(RHO_USERS, len(keep)), replace=False).tolist())

    z0d = {}
    train_clicks_v = {u: [] for u in keep}
    with open(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
              encoding="utf-8") as f:
        for line in f:
            _, user, tstr, hist, impr = line.rstrip("\n").split("\t")
            if user not in keep:
                continue
            if user not in z0d:
                st = user_states(hist.split() if hist else [])
                z0d[user] = st[1] if st else None
            t = parse_time(tstr)
            for tok in impr.split():
                nid, lab = tok.rsplit("-", 1)
                if lab == "1":
                    j = vidx.get(nid)
                    if j is not None:
                        train_clicks_v[user].append((t, j))
    dev_v = {}
    with open(os.path.join(BASE, "MINDlarge_dev", "behaviors.tsv"),
              encoding="utf-8") as f:
        for line in f:
            _, user, tstr, hist, impr = line.rstrip("\n").split("\t")
            if user not in keep:
                continue
            for tok in impr.split():
                nid, lab = tok.rsplit("-", 1)
                j = vidx.get(nid)
                c = nid_cat.get(nid)
                if j is None or c is None:
                    continue
                dev_v.setdefault(user, []).append((j, c, int(lab)))
    prof = {u: {"z0": z0d[u],
                "clicks": [j for _, j in sorted(train_clicks_v[u])],
                "dev": dev_v.get(u, [])}
            for u in keep if z0d.get(u) is not None and dev_v.get(u)}
    with open(os.path.join(SCRATCH, "dense_rho_pieces.pkl"), "wb") as f:
        pickle.dump(prof, f, protocol=4)
    print(f"dense-rho pieces: {len(prof)} users, "
          f"{sum(len(v['dev']) for v in prof.values())} dev rows")


if __name__ == "__main__":
    main()
