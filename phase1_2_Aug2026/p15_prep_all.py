"""
One streaming pass over MINDlarge train collecting everything the final
review round needs:

  A. hist_rows.npz      impression-item rows WITH user history length:
                        cat, align, hist_len, y   (3% impression sample,
                        NO minimum-history filter, for spline/interaction
                        fits and calibration diagnostics)
  B. day_rows.pkl       per-user, per-day impression rows {user: {day:
                        (cats, ys)}} for rolling day-pair validation
                        (10% impression sample of users in states_all)
  C. states_all.pkl     per-user raw history category COUNTS (all users,
                        including cold-start, for Dirichlet shrinkage)
  D. shown_full.pkl     ALL shown-not-clicked (t, cat) events for the 30k
                        dense-rho user subsample (full impression scale,
                        for joint backfire estimation)
  E. clicks_by_day.pkl  per-user time-sorted clicks with day retained
                        {user: [(day, t, cat)]} for day-pair state rolls
"""
import os
import pickle
import random
import numpy as np

BASE = "/Users/piri/Desktop/Recommendation Systems/Mind-Data-Large"
SCRATCH = "/private/tmp/claude-503/-Users-piri/428a207e-f2a3-4218-996f-e2751f17b66e/scratchpad/mind"
SEED = 20260811
rng = random.Random(SEED)


def parse_time(s):
    date, clock, ampm = s.split(" ")
    _, day, _ = date.split("/")
    hh, mm, ss = clock.split(":")
    h = int(hh) % 12
    if ampm == "PM":
        h += 12
    return int(day), int(day) * 86400 + h * 3600 + int(mm) * 60 + int(ss)


def main():
    with open(os.path.join(SCRATCH, "news_cat.pkl"), "rb") as f:
        nc = pickle.load(f)
    nid_cat = nc["nid_cat"]

    with open(os.path.join(SCRATCH, "user_hist.pkl"), "rb") as f:
        eligible = sorted(pickle.load(f).keys())
    sub_rng = np.random.default_rng(20260811)
    sub30k = set(sub_rng.choice(eligible, min(30000, len(eligible)),
                                replace=False).tolist())

    hist_counts = {}
    rowsA = {k: [] for k in ("cat", "align", "hist_len", "y")}
    day_rows = {}
    shown_full = {}
    clicks_by_day = {}

    with open(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
              encoding="utf-8") as f:
        for ln, line in enumerate(f):
            _, user, tstr, hist, impr = line.rstrip("\n").split("\t")
            day, t = parse_time(tstr)
            if user not in hist_counts:
                cnt = np.zeros(18, dtype=np.float32)
                for h in (hist.split() if hist else []):
                    c = nid_cat.get(h)
                    if c is not None:
                        cnt[c] += 1
                hist_counts[user] = cnt
            cnt = hist_counts[user]
            n = cnt.sum()
            z = cnt / n if n > 0 else np.full(18, 1.0 / 18, dtype=np.float32)

            sampleA = rng.random() < 0.03
            sampleB = rng.random() < 0.10
            in30k = user in sub30k
            for tok in impr.split():
                nid, lab = tok.rsplit("-", 1)
                c = nid_cat.get(nid)
                if c is None:
                    continue
                y = int(lab)
                if sampleA:
                    rowsA["cat"].append(c)
                    rowsA["align"].append(float(z[c]))
                    rowsA["hist_len"].append(int(n))
                    rowsA["y"].append(y)
                if sampleB:
                    d = day_rows.setdefault(user, {}).setdefault(day, ([], []))
                    d[0].append(c)
                    d[1].append(y)
                if y == 1:
                    clicks_by_day.setdefault(user, []).append((day, t, c))
                elif in30k:
                    shown_full.setdefault(user, []).append((t, c))
            if ln % 400000 == 0:
                print(f"line {ln}: A={len(rowsA['y'])} B={len(day_rows)}",
                      flush=True)

    np.savez_compressed(
        os.path.join(SCRATCH, "hist_rows.npz"),
        cat=np.array(rowsA["cat"], dtype=np.int16),
        align=np.array(rowsA["align"], dtype=np.float32),
        hist_len=np.array(rowsA["hist_len"], dtype=np.int32),
        y=np.array(rowsA["y"], dtype=np.int8))
    for name, obj in [("day_rows.pkl", day_rows),
                      ("states_all.pkl", hist_counts),
                      ("shown_full.pkl", {u: sorted(v) for u, v in shown_full.items()}),
                      ("clicks_by_day.pkl", {u: sorted(v) for u, v in clicks_by_day.items()})]:
        with open(os.path.join(SCRATCH, name), "wb") as fh:
            pickle.dump(obj, fh, protocol=4)
    print(f"A rows {len(rowsA['y'])}; B users {len(day_rows)}; "
          f"C users {len(hist_counts)}; D users {len(shown_full)} "
          f"({sum(len(v) for v in shown_full.values())} shown events)")


if __name__ == "__main__":
    main()
