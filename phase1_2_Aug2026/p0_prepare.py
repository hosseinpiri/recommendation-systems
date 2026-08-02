"""
Phase 1 data preparation.

Parses MINDlarge news + behaviors (train = Nov 9-14, dev = Nov 15) into:
  1. news_cat.pkl        : news_id -> top-level category
  2. user_hist.pkl       : user -> np.array category distribution of pre-period history
                           (from the history field, which lists clicks before the log period)
  3. train_clicks.pkl    : user -> list of (timestamp, cat_idx) clicked during train period
  4. impr_1a.npz         : impression-item sample rows for the consumption model
                           (cat_idx, align = z_hist[cat], label)
  5. dev_rows.pkl        : user -> (cat_idx array, label array) from dev impressions,
                           for the transition profile likelihood
  6. train_shown.pkl     : user -> list of (timestamp, cat_idx) shown-not-clicked (subsampled)
                           for the no-movement-without-consumption test

Intermediates go to SCRATCH; nothing here fits models.
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
os.makedirs(SCRATCH, exist_ok=True)

MIN_HIST = 5          # min history clicks to trust z0
IMPR_SAMPLE_P = 0.15  # fraction of train impressions sampled for the 1a design
SHOWN_SAMPLE_P = 0.10 # fraction of shown-not-clicked kept per user for the rho_ns test
SEED = 20260801

rng = random.Random(SEED)


def build_news_cat():
    cat_of = {}
    cats = set()
    for split in ("MINDlarge_train", "MINDlarge_dev"):
        with open(os.path.join(BASE, split, "news.tsv"), encoding="utf-8") as f:
            for line in f:
                parts = line.split("\t")
                nid, cat = parts[0], parts[1]
                cat_of[nid] = cat
                cats.add(cat)
    cat_list = sorted(cats)
    cat_idx = {c: i for i, c in enumerate(cat_list)}
    nid_cat = {n: cat_idx[c] for n, c in cat_of.items()}
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "wb") as f:
        pickle.dump({"nid_cat": nid_cat, "cat_list": cat_list}, f)
    print(f"news: {len(nid_cat)} ids, {len(cat_list)} categories: {cat_list}")
    return nid_cat, cat_list


def parse_time(s):
    # "11/10/2019 11:30:54 AM" -> sortable key; all Nov 2019, day + seconds suffice
    date, clock, ampm = s.split(" ")
    _, day, _ = date.split("/")
    hh, mm, ss = clock.split(":")
    h = int(hh) % 12
    if ampm == "PM":
        h += 12
    return int(day) * 86400 + h * 3600 + int(mm) * 60 + int(ss)


def main():
    nid_cat, cat_list = build_news_cat()
    K = len(cat_list)

    user_hist_counts = {}
    train_clicks = {}
    train_shown = {}
    rows_cat, rows_align, rows_y = [], [], []

    with open(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"), encoding="utf-8") as f:
        for ln, line in enumerate(f):
            _, user, tstr, hist, impr = line.rstrip("\n").split("\t")
            t = parse_time(tstr)
            hist_ids = hist.split() if hist else []
            if user not in user_hist_counts:
                cnt = np.zeros(K, dtype=np.float64)
                for h in hist_ids:
                    c = nid_cat.get(h)
                    if c is not None:
                        cnt[c] += 1
                user_hist_counts[user] = cnt
            cnt = user_hist_counts[user]
            nh = cnt.sum()
            z = cnt / nh if nh > 0 else None

            sample_this = z is not None and nh >= MIN_HIST and rng.random() < IMPR_SAMPLE_P
            for tok in impr.split():
                nid, lab = tok.rsplit("-", 1)
                c = nid_cat.get(nid)
                if c is None:
                    continue
                y = int(lab)
                if y == 1:
                    train_clicks.setdefault(user, []).append((t, c))
                elif rng.random() < SHOWN_SAMPLE_P:
                    train_shown.setdefault(user, []).append((t, c))
                if sample_this:
                    rows_cat.append(c)
                    rows_align.append(z[c])
                    rows_y.append(y)
            if ln % 200000 == 0:
                print(f"train line {ln}, 1a rows {len(rows_y)}")

    np.savez_compressed(
        os.path.join(SCRATCH, "impr_1a.npz"),
        cat=np.array(rows_cat, dtype=np.int16),
        align=np.array(rows_align, dtype=np.float64),
        y=np.array(rows_y, dtype=np.int8),
    )
    print(f"1a design: {len(rows_y)} rows, click rate {np.mean(rows_y):.4f}")

    dev_rows = {}
    with open(os.path.join(BASE, "MINDlarge_dev", "behaviors.tsv"), encoding="utf-8") as f:
        for line in f:
            _, user, tstr, hist, impr = line.rstrip("\n").split("\t")
            cs, ys = [], []
            for tok in impr.split():
                nid, lab = tok.rsplit("-", 1)
                c = nid_cat.get(nid)
                if c is None:
                    continue
                cs.append(c)
                ys.append(int(lab))
            if cs:
                a, b = dev_rows.setdefault(user, ([], []))
                a.extend(cs)
                b.extend(ys)

    # keep only what the transition step needs: users with history and dev rows
    hist_dist = {}
    for u, cnt in user_hist_counts.items():
        s = cnt.sum()
        if s >= MIN_HIST and u in dev_rows:
            hist_dist[u] = (cnt / s).astype(np.float32)
    dev_rows = {u: (np.array(v[0], dtype=np.int16), np.array(v[1], dtype=np.int8))
                for u, v in dev_rows.items() if u in hist_dist}
    train_clicks = {u: sorted(v) for u, v in train_clicks.items() if u in hist_dist}
    train_shown = {u: sorted(v) for u, v in train_shown.items() if u in hist_dist}

    for name, obj in [
        ("user_hist.pkl", hist_dist),
        ("train_clicks.pkl", train_clicks),
        ("train_shown.pkl", train_shown),
        ("dev_rows.pkl", dev_rows),
    ]:
        with open(os.path.join(SCRATCH, name), "wb") as f:
            pickle.dump(obj, f, protocol=4)

    n_dev_rows = sum(len(v[0]) for v in dev_rows.values())
    print(f"transition sample: {len(hist_dist)} users with history+dev, "
          f"{len(train_clicks)} of them with train clicks, {n_dev_rows} dev item-rows")

    # full-history category distribution across all users, for the simulator's z0 pool
    pool = np.stack([v for v in hist_dist.values()]) if hist_dist else np.zeros((0, K))
    np.save(os.path.join(SCRATCH, "z0_pool.npy"), pool)
    print("saved z0 pool", pool.shape)


if __name__ == "__main__":
    main()
