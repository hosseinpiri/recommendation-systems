"""One-time prep on ARC: parse MIND, build states in BOTH category universes.

Universe 18: the original 18-category simplex.
Universe 9: the 8 simulated categories plus an explicit 'other' bucket, no
renormalization, so kappa/alpha/rho can be re-estimated on exactly the state
definition the simulator uses (audit fix: the 1/q_u amplification).

Outputs (in OUT):
  news_cat.pkl                    cat maps
  states.pkl                      per-user: z18, z9 (history), tenure (hist clicks)
  train_clicks.pkl                per-user time-sorted [(t, cat18)]
  train_shown.pkl                 10% subsample [(t, cat18)]
  dev_rows.pkl                    per-user (cats18 array, labels array)
  impr_rows.npz                   sampled impression-item rows: cat18, align18,
                                  align9, y   (for consumption refits)
  z0_pool18.npy, z0_pool9.npy     empirical initial-state pools
  tenure_pool.npy                 history lengths for the pool users
"""
import os
import pickle
import random
import numpy as np

BASE = os.path.expanduser("~/mind_calib/data")
OUT = os.path.expanduser("~/mind_calib/out/prep")
os.makedirs(OUT, exist_ok=True)

ITEMS8 = ["news", "lifestyle", "sports", "finance", "foodanddrink",
          "entertainment", "travel", "health"]
MIN_HIST = 5
IMPR_SAMPLE_P = 0.15
SHOWN_SAMPLE_P = 0.10
SEED = 20260805
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
    cat_of = {}
    for split in ("MINDlarge_train", "MINDlarge_dev"):
        with open(os.path.join(BASE, split, "news.tsv"), encoding="utf-8") as f:
            for line in f:
                p = line.split("\t")
                cat_of[p[0]] = p[1]
    cat_list = sorted(set(cat_of.values()))
    c18 = {c: i for i, c in enumerate(cat_list)}
    # universe 9: 8 items + other
    c9 = {}
    for c in cat_list:
        c9[c] = ITEMS8.index(c) if c in ITEMS8 else 8
    nid18 = {n: c18[c] for n, c in cat_of.items()}
    nid9 = {n: c9[c] for n, c in cat_of.items()}
    with open(os.path.join(OUT, "news_cat.pkl"), "wb") as f:
        pickle.dump({"cat_list": cat_list, "items8": ITEMS8,
                     "nid18": nid18, "nid9": nid9}, f)

    states = {}
    train_clicks = {}
    train_shown = {}
    rows = {k: [] for k in ("cat18", "align18", "align9", "y")}

    with open(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
              encoding="utf-8") as f:
        for ln, line in enumerate(f):
            _, user, tstr, hist, impr = line.rstrip("\n").split("\t")
            t = parse_time(tstr)
            if user not in states:
                cnt18 = np.zeros(18)
                for h in (hist.split() if hist else []):
                    c = nid18.get(h)
                    if c is not None:
                        cnt18[c] += 1
                n = cnt18.sum()
                if n >= MIN_HIST:
                    z18 = cnt18 / n
                    z9 = np.zeros(9)
                    for c in range(18):
                        z9[c9[cat_list[c]]] += z18[c]
                    states[user] = (z18.astype(np.float32),
                                    z9.astype(np.float32), int(n))
                else:
                    states[user] = None
            st = states[user]
            sample_this = st is not None and rng.random() < IMPR_SAMPLE_P
            for tok in impr.split():
                nid, lab = tok.rsplit("-", 1)
                a = nid18.get(nid)
                if a is None:
                    continue
                y = int(lab)
                if st is not None:
                    if y == 1:
                        train_clicks.setdefault(user, []).append((t, a))
                    elif rng.random() < SHOWN_SAMPLE_P:
                        train_shown.setdefault(user, []).append((t, a))
                if sample_this:
                    rows["cat18"].append(a)
                    rows["align18"].append(float(st[0][a]))
                    rows["align9"].append(float(st[1][nid9[nid]]))
                    rows["y"].append(y)
            if ln % 400000 == 0:
                print(f"train {ln}", flush=True)

    np.savez_compressed(
        os.path.join(OUT, "impr_rows.npz"),
        cat18=np.array(rows["cat18"], dtype=np.int16),
        align18=np.array(rows["align18"], dtype=np.float32),
        align9=np.array(rows["align9"], dtype=np.float32),
        y=np.array(rows["y"], dtype=np.int8),
    )
    print("impr rows:", len(rows["y"]))

    dev_rows = {}
    with open(os.path.join(BASE, "MINDlarge_dev", "behaviors.tsv"),
              encoding="utf-8") as f:
        for line in f:
            _, user, tstr, hist, impr = line.rstrip("\n").split("\t")
            if states.get(user) is None:
                continue
            for tok in impr.split():
                nid, lab = tok.rsplit("-", 1)
                a = nid18.get(nid)
                if a is None:
                    continue
                dev_rows.setdefault(user, ([], []))
                dev_rows[user][0].append(a)
                dev_rows[user][1].append(int(lab))

    keep = {u for u in dev_rows if states.get(u) is not None}
    states = {u: v for u, v in states.items() if v is not None}
    dev_rows = {u: (np.array(v[0], dtype=np.int16),
                    np.array(v[1], dtype=np.int8))
                for u, v in dev_rows.items()}
    train_clicks = {u: sorted(v) for u, v in train_clicks.items() if u in keep}
    train_shown = {u: sorted(v) for u, v in train_shown.items() if u in keep}

    for name, obj in [("states.pkl", states), ("train_clicks.pkl", train_clicks),
                      ("train_shown.pkl", train_shown), ("dev_rows.pkl", dev_rows)]:
        with open(os.path.join(OUT, name), "wb") as f:
            pickle.dump(obj, f, protocol=4)

    pool_users = sorted(keep)
    z18p = np.stack([states[u][0] for u in pool_users])
    z9p = np.stack([states[u][1] for u in pool_users])
    ten = np.array([states[u][2] for u in pool_users])
    np.save(os.path.join(OUT, "z0_pool18.npy"), z18p)
    np.save(os.path.join(OUT, "z0_pool9.npy"), z9p)
    np.save(os.path.join(OUT, "tenure_pool.npy"), ten)
    print(f"prep done: {len(states)} users with state, "
          f"{len(keep)} with dev rows")


if __name__ == "__main__":
    main()
