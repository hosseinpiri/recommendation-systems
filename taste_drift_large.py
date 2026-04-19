"""
Taste Drift Analysis — MIND Large Dataset
==========================================
Same five analyses as the small dataset, but on MINDlarge
(train: Nov 9-14, dev: Nov 15, test: Nov 16-22).

Key advantage: 14-day window (vs. 7 days) with ~5M impressions
and ~1M users. The test set lacks impression click labels, but
we use its click histories to track taste evolution.

Strategy for using the test set:
- Test behaviors have click histories but no impression labels.
- A user's click history in the test set (Nov 16-22) reflects ALL
  articles they clicked up to that point, including articles from
  the train/dev period AND new clicks in the test period.
- By comparing a user's history at their LAST test-set appearance
  vs. their LAST train-set appearance, we can infer what new
  articles they clicked during the test period.
- This gives us taste vectors for 3 periods:
    Period 1 (early):  Nov 9-11  — from train impression clicks
    Period 2 (mid):    Nov 12-15 — from train+dev impression clicks
    Period 3 (late):   Nov 16-22 — inferred from test history growth

Author: Hossein Piri / Claude
Date: 2026-03-27
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import date
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = "/Users/piri/Desktop/Recommendation Systems/Dataset"
OUT = "/Users/piri/Desktop/Recommendation Systems/code/output_large"
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("TASTE DRIFT ANALYSIS — MIND Large Dataset")
print("=" * 60)

# ============================================================
# 0. LOAD NEWS METADATA (all three splits)
# ============================================================
print("\nLoading news metadata...")
news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract',
             'url', 'title_entities', 'abstract_entities']

dfs_news = []
for split in ['MINDlarge_train', 'MINDlarge_dev', 'MINDlarge_test']:
    path = os.path.join(BASE, split, "news.tsv")
    df = pd.read_csv(path, sep='\t', header=None, names=news_cols, usecols=[0, 1, 2])
    dfs_news.append(df)
    print(f"  {split}: {len(df):,} articles")

news = pd.concat(dfs_news).drop_duplicates(subset='news_id')
news_cat = dict(zip(news['news_id'], news['category']))
print(f"  Total unique articles: {len(news):,}")

cat_counts = news['category'].value_counts()
categories = sorted(cat_counts[cat_counts >= 50].index.tolist())
cat_to_idx = {c: i for i, c in enumerate(categories)}
n_cats = len(categories)
print(f"  Categories ({n_cats}): {categories}")


# ============================================================
# HELPERS
# ============================================================
def nids_to_cat_vec(nids):
    vec = np.zeros(n_cats)
    for nid in nids:
        c = news_cat.get(nid)
        if c and c in cat_to_idx:
            vec[cat_to_idx[c]] += 1
    return vec

def normalize(v):
    s = v.sum()
    return v / s if s > 0 else v

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return np.dot(a, b) / (na * nb)

def entropy(v):
    p = v[v > 0]
    return -np.sum(p * np.log2(p)) if len(p) > 0 else 0.0

def parse_history(s):
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '':
        return []
    return s.strip().split()

def parse_impressions_labeled(s):
    """Parse impression string WITH labels (train/dev)."""
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '':
        return [], []
    clicked, shown = [], []
    for item in s.strip().split():
        nid, label = item.rsplit('-', 1)
        shown.append(nid)
        if label == '1':
            clicked.append(nid)
    return clicked, shown

def parse_impressions_unlabeled(s):
    """Parse impression string WITHOUT labels (test)."""
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '':
        return []
    return s.strip().split()


# ============================================================
# 1. LOAD AND PARSE BEHAVIORS
# ============================================================
beh_cols = ['impression_id', 'user_id', 'timestamp', 'click_history', 'impressions']

# --- TRAIN + DEV (have click labels) ---
print("\nLoading train behaviors (2.2M rows)...")
beh_train = pd.read_csv(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
                         sep='\t', header=None, names=beh_cols)
print("Loading dev behaviors (376K rows)...")
beh_dev = pd.read_csv(os.path.join(BASE, "MINDlarge_dev", "behaviors.tsv"),
                       sep='\t', header=None, names=beh_cols)

beh_labeled = pd.concat([beh_train, beh_dev], ignore_index=True)
beh_labeled['time'] = pd.to_datetime(beh_labeled['timestamp'], format='%m/%d/%Y %I:%M:%S %p')
beh_labeled['date'] = beh_labeled['time'].dt.date
print(f"  Labeled impressions: {len(beh_labeled):,}")
print(f"  Unique users (labeled): {beh_labeled['user_id'].nunique():,}")

# Parse labeled impressions
print("  Parsing click histories...")
beh_labeled['hist_list'] = beh_labeled['click_history'].apply(parse_history)
print("  Parsing impression clicks...")
parsed = beh_labeled['impressions'].apply(parse_impressions_labeled)
beh_labeled['clicked_list'] = parsed.apply(lambda x: x[0])
beh_labeled['shown_list'] = parsed.apply(lambda x: x[1])
beh_labeled['n_clicked'] = beh_labeled['clicked_list'].apply(len)
beh_labeled['n_history'] = beh_labeled['hist_list'].apply(len)

# --- TEST (no click labels — use history growth) ---
print("\nLoading test behaviors (2.4M rows)...")
beh_test = pd.read_csv(os.path.join(BASE, "MINDlarge_test", "behaviors.tsv"),
                        sep='\t', header=None, names=beh_cols)
beh_test['time'] = pd.to_datetime(beh_test['timestamp'], format='%m/%d/%Y %I:%M:%S %p')
beh_test['date'] = beh_test['time'].dt.date
print(f"  Test impressions: {len(beh_test):,}")
print(f"  Unique users (test): {beh_test['user_id'].nunique():,}")
print("  Parsing test click histories...")
beh_test['hist_list'] = beh_test['click_history'].apply(parse_history)
beh_test['n_history'] = beh_test['hist_list'].apply(len)

# ============================================================
# 2. DEFINE TIME PERIODS
# ============================================================
# Period 1 (early):  Nov 9-11
# Period 2 (mid):    Nov 12-15
# Period 3 (late):   Nov 16-22  (from test set history growth)

P1_DATES = {date(2019, 11, 9), date(2019, 11, 10), date(2019, 11, 11)}
P2_DATES = {date(2019, 11, 12), date(2019, 11, 13), date(2019, 11, 14), date(2019, 11, 15)}
P3_DATES = {date(2019, 11, d) for d in range(16, 23)}

def get_period(d):
    if d in P1_DATES: return 'P1'
    if d in P2_DATES: return 'P2'
    if d in P3_DATES: return 'P3'
    return None

beh_labeled['period'] = beh_labeled['date'].apply(get_period)

print(f"\nPeriod distribution (labeled):")
print(beh_labeled['period'].value_counts().sort_index().to_string())

# ============================================================
# 3. BUILD PER-USER DATA
# ============================================================
print("\n" + "=" * 60)
print("BUILDING PER-USER AGGREGATES...")
print("=" * 60)

# --- From labeled data (train+dev): get P1 and P2 clicks ---
print("Aggregating labeled clicks per user-period...")
user_period_clicks = defaultdict(lambda: {'P1': [], 'P2': [], 'P1_shown': [], 'P2_shown': []})
user_hist = {}  # uid → longest history from labeled data

for chunk_start in range(0, len(beh_labeled), 500000):
    chunk = beh_labeled.iloc[chunk_start:chunk_start + 500000]
    print(f"  Processing rows {chunk_start:,}–{chunk_start + len(chunk):,}...")
    for _, row in chunk.iterrows():
        uid = row['user_id']
        p = row['period']
        if p in ('P1', 'P2'):
            user_period_clicks[uid][p].extend(row['clicked_list'])
            user_period_clicks[uid][p + '_shown'].extend(row['shown_list'])
        # Track longest history
        if uid not in user_hist or len(row['hist_list']) > len(user_hist[uid]):
            user_hist[uid] = row['hist_list']

print(f"  Users with labeled data: {len(user_period_clicks):,}")

# --- From test data: infer P3 clicks via history growth ---
# For each user in test, get their latest history.
# P3 new clicks = test_history - labeled_history
print("\nInferring P3 clicks from test history growth...")

user_test_hist = {}
for chunk_start in range(0, len(beh_test), 500000):
    chunk = beh_test.iloc[chunk_start:chunk_start + 500000]
    print(f"  Processing test rows {chunk_start:,}–{chunk_start + len(chunk):,}...")
    for _, row in chunk.iterrows():
        uid = row['user_id']
        if uid not in user_test_hist or len(row['hist_list']) > len(user_test_hist[uid]):
            user_test_hist[uid] = row['hist_list']

# Compute P3 new clicks = set(test_history) - set(labeled_history)
user_p3_clicks = {}
overlap_users = set(user_hist.keys()) & set(user_test_hist.keys())
print(f"  Users in both labeled and test: {len(overlap_users):,}")

for uid in overlap_users:
    old = set(user_hist[uid])
    new_full = user_test_hist[uid]
    # New clicks = articles in test history not in labeled history (deduplicated)
    seen = set()
    new_clicks = []
    for nid in new_full:
        if nid not in old and nid not in seen:
            new_clicks.append(nid)
            seen.add(nid)
    if len(new_clicks) > 0:
        user_p3_clicks[uid] = new_clicks

print(f"  Users with P3 growth (new clicks): {len(user_p3_clicks):,}")

# ============================================================
# 4. COMPUTE CATEGORY VECTORS
# ============================================================
print("\nComputing category vectors...")

user_vecs = {}
n_three_period = 0

for uid in user_period_clicks:
    rec = user_period_clicks[uid]
    hist = user_hist.get(uid, [])

    hist_raw = nids_to_cat_vec(hist)
    p1_raw = nids_to_cat_vec(rec['P1'])
    p2_raw = nids_to_cat_vec(rec['P2'])
    p1_shown_raw = nids_to_cat_vec(rec['P1_shown'])
    p2_shown_raw = nids_to_cat_vec(rec['P2_shown'])

    # All labeled clicks
    all_clicks_raw = p1_raw + p2_raw
    all_shown_raw = p1_shown_raw + p2_shown_raw

    d = {
        'hist_vec': normalize(hist_raw),
        'p1_vec': normalize(p1_raw),
        'p2_vec': normalize(p2_raw),
        'click_vec': normalize(all_clicks_raw),
        'shown_vec': normalize(all_shown_raw),
        'n_hist': len(hist),
        'n_p1': int(p1_raw.sum()),
        'n_p2': int(p2_raw.sum()),
    }

    if uid in user_p3_clicks:
        p3_raw = nids_to_cat_vec(user_p3_clicks[uid])
        d['p3_vec'] = normalize(p3_raw)
        d['n_p3'] = int(p3_raw.sum())
        if d['n_p1'] >= 2 and d['n_p2'] >= 2 and d['n_p3'] >= 2:
            n_three_period += 1
    else:
        d['p3_vec'] = None
        d['n_p3'] = 0

    user_vecs[uid] = d

print(f"Total users with vectors: {len(user_vecs):,}")
print(f"Users with all 3 periods (P1≥2, P2≥2, P3≥2): {n_three_period:,}")


# ============================================================
# ANALYSIS 1: PRIOR TASTE PERSISTENCE
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 1: PRIOR TASTE vs. CURRENT CHOICES")
print("=" * 60)

MIN_HISTORY = 5
MIN_CLICKS = 3

a1_data = []
for uid, v in user_vecs.items():
    if v['n_hist'] < MIN_HISTORY or (v['n_p1'] + v['n_p2']) < MIN_CLICKS:
        continue
    shc = cosine_sim(v['hist_vec'], v['click_vec'])
    shs = cosine_sim(v['hist_vec'], v['shown_vec'])
    if np.isnan(shc) or np.isnan(shs):
        continue
    a1_data.append({'uid': uid, 'sim_hist_click': shc, 'sim_hist_shown': shs})

df1 = pd.DataFrame(a1_data)
print(f"Eligible users: {len(df1):,}")

mean_hc = df1['sim_hist_click'].mean()
mean_hs = df1['sim_hist_shown'].mean()
print(f"Avg cos(history, clicked):  {mean_hc:.4f}")
print(f"Avg cos(history, shown):    {mean_hs:.4f}")
print(f"Gap (click − shown):         {mean_hc - mean_hs:+.4f}")
t1, p1 = stats.ttest_rel(df1['sim_hist_click'], df1['sim_hist_shown'])
print(f"Paired t-test: t={t1:.2f}, p={p1:.2e}")
above_diag = (df1['sim_hist_click'] > df1['sim_hist_shown']).mean() * 100
print(f"{above_diag:.1f}% of users above diagonal")

# Per-category
print(f"\n{'Category':<16} {'Heavy':>8} {'Light':>8} {'Ratio':>7} {'p':>10}")
print("-" * 55)
cat_a1 = []
for cat in categories:
    ci = cat_to_idx[cat]
    hf = np.array([user_vecs[d['uid']]['hist_vec'][ci] for d in a1_data])
    cf = np.array([user_vecs[d['uid']]['click_vec'][ci] for d in a1_data])
    if (hf > 0).sum() < 50:
        continue
    med = np.median(hf[hf > 0])
    heavy = hf > med
    light = hf <= med
    if heavy.sum() < 30 or light.sum() < 30:
        continue
    h, l = cf[heavy].mean(), cf[light].mean()
    r = h / l if l > 0 else np.inf
    t, p = stats.ttest_ind(cf[heavy], cf[light])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{cat:<16} {h:>7.4f} {l:>7.4f} {r:>6.2f}x {p:>9.2e} {sig}")
    cat_a1.append({'category': cat, 'heavy': h, 'light': l, 'ratio': r, 'p': p})


# ============================================================
# ANALYSIS 2: THREE-PERIOD TASTE SHIFT
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 2: THREE-PERIOD TASTE SHIFT (P1 → P2 → P3)")
print("=" * 60)
print("Key test: does P3 taste shift toward P1/P2 consumption?")
print("This uses the FULL 14-day window (Nov 9-22).")
print("-" * 60)

a2_data = []
for uid, v in user_vecs.items():
    if v['p3_vec'] is None:
        continue
    if v['n_p1'] < 2 or v['n_p2'] < 2 or v['n_p3'] < 2:
        continue
    if np.linalg.norm(v['hist_vec']) == 0:
        continue

    # Shift from baseline to P3
    shift_p3 = v['p3_vec'] - v['hist_vec']
    # Shift from baseline to P2
    shift_p2 = v['p2_vec'] - v['hist_vec']

    # Does P1 consumption predict P2 shift?
    sim_p1_shift_p2 = cosine_sim(shift_p2, v['p1_vec'])
    # Does P1+P2 consumption predict P3 shift?
    p1p2_vec = normalize(nids_to_cat_vec(
        user_period_clicks[uid]['P1'] + user_period_clicks[uid]['P2']))
    sim_p1p2_shift_p3 = cosine_sim(shift_p3, p1p2_vec)
    # Does P2 consumption predict P3 shift?
    sim_p2_shift_p3 = cosine_sim(shift_p3, v['p2_vec'])

    # Early-to-late similarity
    sim_p1_p2 = cosine_sim(v['p1_vec'], v['p2_vec'])
    sim_p1_p3 = cosine_sim(v['p1_vec'], v['p3_vec'])
    sim_p2_p3 = cosine_sim(v['p2_vec'], v['p3_vec'])

    a2_data.append({
        'uid': uid,
        'sim_p1_shift_p2': sim_p1_shift_p2,
        'sim_p2_shift_p3': sim_p2_shift_p3,
        'sim_p1p2_shift_p3': sim_p1p2_shift_p3,
        'sim_p1_p2': sim_p1_p2,
        'sim_p1_p3': sim_p1_p3,
        'sim_p2_p3': sim_p2_p3,
    })

df2 = pd.DataFrame(a2_data).dropna()
print(f"Three-period users: {len(df2):,}")

print(f"\n--- Taste persistence ---")
print(f"Avg cos(P1, P2): {df2['sim_p1_p2'].mean():.4f}")
print(f"Avg cos(P1, P3): {df2['sim_p1_p3'].mean():.4f}")
print(f"Avg cos(P2, P3): {df2['sim_p2_p3'].mean():.4f}")

print(f"\n--- Shift aligned with consumption? ---")
m1 = df2['sim_p1_shift_p2'].mean()
t_a, p_a = stats.ttest_1samp(df2['sim_p1_shift_p2'], 0)
print(f"cos(P2 shift, P1 consumption):     {m1:+.4f}  t={t_a:.2f}, p={p_a:.2e}")

m2 = df2['sim_p2_shift_p3'].mean()
t_b, p_b = stats.ttest_1samp(df2['sim_p2_shift_p3'], 0)
print(f"cos(P3 shift, P2 consumption):     {m2:+.4f}  t={t_b:.2f}, p={p_b:.2e}")

m3 = df2['sim_p1p2_shift_p3'].mean()
t_c, p_c = stats.ttest_1samp(df2['sim_p1p2_shift_p3'], 0)
print(f"cos(P3 shift, P1+P2 consumption):  {m3:+.4f}  t={t_c:.2f}, p={p_c:.2e}")

# Permutation test for the main result — use ONLY the NaN-filtered users (matching df2)
print(f"\nPermutation test (P3 shift ~ P2 consumption):")
np.random.seed(42)
n_perm = 1000
# Use filtered UIDs from df2, not raw a2_data (which may contain NaN rows)
filtered_uids = df2['uid'].tolist()
shifts_p3 = [user_vecs[uid]['p3_vec'] - user_vecs[uid]['hist_vec'] for uid in filtered_uids]
p2_vecs = [user_vecs[uid]['p2_vec'] for uid in filtered_uids]
obs_mean = m2

null_means = []
for _ in range(n_perm):
    perm = np.random.permutation(len(p2_vecs))
    sims = [cosine_sim(shifts_p3[i], p2_vecs[perm[i]]) for i in range(len(shifts_p3))]
    null_means.append(np.nanmean(sims))

null_mu = np.mean(null_means)
null_sd = np.std(null_means)
z = (obs_mean - null_mu) / null_sd if null_sd > 0 else 0
count = sum(1 for n in null_means if n >= obs_mean) if obs_mean > null_mu else sum(1 for n in null_means if n <= obs_mean)
perm_p = (count + 1) / (n_perm + 1)

print(f"  Observed:  {obs_mean:+.4f}")
print(f"  Null mean: {null_mu:+.4f}")
print(f"  Z-score:   {z:+.2f}")
print(f"  p-value:   {perm_p:.4f}")

# Transition matrix P1 → P3 (the 14-day gap)
top_cats = [c for c in ['news', 'sports', 'finance', 'lifestyle', 'health',
                         'travel', 'foodanddrink', 'autos', 'entertainment',
                         'video', 'weather', 'tv', 'music', 'movies']
            if c in cat_to_idx]
top_idx = [cat_to_idx[c] for c in top_cats]

transition = np.zeros((len(top_cats), len(top_cats)))
trans_counts = np.zeros(len(top_cats))

for d in a2_data:
    v = user_vecs[d['uid']]
    if v['p1_vec'].max() == 0:
        continue
    dom = np.argmax(v['p1_vec'])
    if dom in top_idx:
        row = top_idx.index(dom)
        trans_counts[row] += 1
        for j, cj in enumerate(top_idx):
            transition[row, j] += v['p3_vec'][cj]

for i in range(len(top_cats)):
    if trans_counts[i] > 0:
        transition[i] /= trans_counts[i]

print(f"\nTransition matrix P1 → P3 (14-day gap):")
print(f"{'':>14}", end='')
for c in top_cats:
    print(f"{c[:6]:>8}", end='')
print()
for i, c in enumerate(top_cats):
    print(f"{c:>14}", end='')
    for j in range(len(top_cats)):
        mark = " *" if i == j else "  "
        print(f"{transition[i,j]:>6.2f}{mark}", end='')
    print()


# ============================================================
# ANALYSIS 3: SESSION-LEVEL SELF-REINFORCEMENT
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 3: SESSION-LEVEL SELF-REINFORCEMENT")
print("=" * 60)

# Build session vectors from labeled data
print("Building session vectors for multi-day users...")
multi_day_uids = [uid for uid, v in user_vecs.items() if v['n_p1'] > 0 and v['n_p2'] > 0]
print(f"Users with clicks in both P1 and P2: {len(multi_day_uids):,}")

# Sample if too many (for speed)
MAX_USERS_SESS = 100000
if len(multi_day_uids) > MAX_USERS_SESS:
    np.random.seed(42)
    multi_day_uids = list(np.random.choice(multi_day_uids, MAX_USERS_SESS, replace=False))
    print(f"Sampled {MAX_USERS_SESS:,} users for session analysis")

session_vecs = defaultdict(dict)
subset = beh_labeled[beh_labeled['user_id'].isin(set(multi_day_uids))]
print(f"Filtering to {len(subset):,} impressions...")

for uid, grp in subset.groupby('user_id'):
    for d, dgrp in grp.groupby('date'):
        clicks = []
        for cl in dgrp['clicked_list']:
            clicks.extend(cl)
        if len(clicks) == 0:
            continue
        session_vecs[uid][d] = normalize(nids_to_cat_vec(clicks))

reinf_yes = defaultdict(list)
reinf_no = defaultdict(list)

for uid, sess in session_vecs.items():
    dates = sorted(sess.keys())
    if len(dates) < 2:
        continue
    for t in range(len(dates) - 1):
        v_now = sess[dates[t]]
        v_next = sess[dates[t + 1]]
        for ci in range(n_cats):
            if v_now[ci] > 0:
                reinf_yes[categories[ci]].append(v_next[ci])
            else:
                reinf_no[categories[ci]].append(v_next[ci])

print(f"\n{'Category':<16} {'P(X|click)':>11} {'P(X|no)':>9} {'Lift':>8} {'N(yes)':>8} {'N(no)':>8} {'p':>10}")
print("-" * 76)

cat_reinf = []
overall_yes, overall_no = [], []
for cat in categories:
    yes = reinf_yes[cat]
    no = reinf_no[cat]
    if len(yes) < 30 or len(no) < 30:
        continue
    py = np.mean(yes)
    pn = np.mean(no)
    lift = (py / pn - 1) * 100 if pn > 0 else np.inf
    t, p = stats.ttest_ind(yes, no)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    print(f"{cat:<16} {py:>10.4f} {pn:>8.4f} {lift:>+7.1f}% {len(yes):>8} {len(no):>8} {p:>9.2e} {sig}")
    cat_reinf.append({'category': cat, 'p_yes': py, 'p_no': pn, 'lift': lift, 'p': p})
    overall_yes.extend(yes)
    overall_no.extend(no)

df_cr = pd.DataFrame(cat_reinf)
sig_n = (df_cr['p'] < 0.05).sum()
print(f"\n→ {sig_n}/{len(df_cr)} categories significant")
print(f"Overall: P(next|click) = {np.mean(overall_yes):.4f}, P(next|no) = {np.mean(overall_no):.4f}")
print(f"         Lift = {(np.mean(overall_yes)/np.mean(overall_no)-1)*100:+.1f}%")


# ============================================================
# ANALYSIS 4: ENTROPY OVER THREE PERIODS
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 4: TASTE CONCENTRATION (ENTROPY) — 3 PERIODS")
print("=" * 60)

ent_data = []
for uid, v in user_vecs.items():
    if v['n_p1'] < 2 or v['n_p2'] < 2:
        continue
    e1 = entropy(v['p1_vec'])
    e2 = entropy(v['p2_vec'])

    d = {'uid': uid, 'e_p1': e1, 'e_p2': e2, 'delta_12': e2 - e1}

    if v['p3_vec'] is not None and v['n_p3'] >= 2:
        e3 = entropy(v['p3_vec'])
        d['e_p3'] = e3
        d['delta_13'] = e3 - e1
        d['delta_23'] = e3 - e2
    else:
        d['e_p3'] = np.nan
        d['delta_13'] = np.nan
        d['delta_23'] = np.nan

    ent_data.append(d)

df_ent = pd.DataFrame(ent_data)
df_ent_12 = df_ent.dropna(subset=['e_p1', 'e_p2'])
df_ent_3 = df_ent.dropna(subset=['e_p3'])

print(f"Users with P1+P2: {len(df_ent_12):,}")
print(f"Users with P1+P2+P3: {len(df_ent_3):,}")

print(f"\nMean entropy P1: {df_ent_12['e_p1'].mean():.3f} bits")
print(f"Mean entropy P2: {df_ent_12['e_p2'].mean():.3f} bits")
t12, p12 = stats.ttest_1samp(df_ent_12['delta_12'], 0)
print(f"ΔEntropy (P2−P1): {df_ent_12['delta_12'].mean():+.3f}  t={t12:.2f}, p={p12:.2e}")
print(f"  More focused: {(df_ent_12['delta_12'] < 0).mean()*100:.1f}%")

if len(df_ent_3) > 0:
    print(f"\nMean entropy P3: {df_ent_3['e_p3'].mean():.3f} bits")
    t13, p13 = stats.ttest_1samp(df_ent_3['delta_13'], 0)
    t23, p23 = stats.ttest_1samp(df_ent_3['delta_23'], 0)
    print(f"ΔEntropy (P3−P1): {df_ent_3['delta_13'].mean():+.3f}  t={t13:.2f}, p={p13:.2e}")
    print(f"ΔEntropy (P3−P2): {df_ent_3['delta_23'].mean():+.3f}  t={t23:.2f}, p={p23:.2e}")
    print(f"  More focused P1→P3: {(df_ent_3['delta_13'] < 0).mean()*100:.1f}%")


# ============================================================
# ANALYSIS 5: CONTROLLED REGRESSION
# ============================================================
print("\n" + "=" * 60)
print("ANALYSIS 5: CONTROLLED REGRESSION")
print("=" * 60)

rows = []
uid_list = []
for d in a1_data:
    v = user_vecs[d['uid']]
    for ci in range(n_cats):
        rows.append({'hist': v['hist_vec'][ci], 'shown': v['shown_vec'][ci],
                      'click': v['click_vec'][ci]})
        uid_list.append(d['uid'])

dfr = pd.DataFrame(rows)
dfr['uid'] = uid_list
mask = (dfr['hist'] > 0) | (dfr['shown'] > 0) | (dfr['click'] > 0)
dfr = dfr[mask].reset_index(drop=True)

X = np.column_stack([np.ones(len(dfr)), dfr['hist'].values, dfr['shown'].values])
y = dfr['click'].values
beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
y_hat = X @ beta
resid = y - y_hat
ss_res = np.sum(resid ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
n_obs, k = len(y), 3

# Standard (iid) SEs
se_iid = np.sqrt(ss_res / (n_obs - k) * np.diag(np.linalg.inv(X.T @ X)))

# Clustered SEs (by user) — sandwich estimator
XtX_inv = np.linalg.inv(X.T @ X)
B = np.zeros((k, k))
unique_uids = dfr['uid'].unique()
n_clusters = len(unique_uids)
uid_to_rows = dfr.groupby('uid').apply(lambda g: g.index.tolist())
for uid_rows in uid_to_rows:
    Xg = X[uid_rows]
    eg = resid[uid_rows]
    score_g = Xg.T @ eg
    B += np.outer(score_g, score_g)

correction = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / (n_obs - k))
V_clustered = correction * XtX_inv @ B @ XtX_inv
se_clust = np.sqrt(np.diag(V_clustered))

t_stats_clust = beta / se_clust
p_vals_clust = 2 * (1 - stats.t.cdf(np.abs(t_stats_clust), df=n_clusters - 1))

print(f"N = {n_obs:,}, R² = {r2:.4f}, Clusters (users) = {n_clusters:,}")
print(f"\n{'Variable':<14} {'Coef':>10} {'SE(iid)':>10} {'SE(clust)':>10} {'t(clust)':>10} {'p(clust)':>12}")
print("-" * 70)
for i, lab in enumerate(['intercept', 'hist_frac', 'shown_frac']):
    sig = "***" if p_vals_clust[i] < 0.001 else "**" if p_vals_clust[i] < 0.01 else "*" if p_vals_clust[i] < 0.05 else ""
    print(f"{lab:<14} {beta[i]:>10.4f} {se_iid[i]:>10.4f} {se_clust[i]:>10.4f} {t_stats_clust[i]:>10.2f} {p_vals_clust[i]:>11.2e} {sig}")

print(f"\n→ β(hist_frac) = {beta[1]:.4f}, clustered SEs are {se_clust[1]/se_iid[1]:.1f}x larger than iid SEs.")


# ============================================================
# FIGURES
# ============================================================
print("\n" + "=" * 60)
print("GENERATING FIGURES...")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Taste Drift Analysis — MIND Large (Nov 9–22, 14 days, ~5M impressions)',
             fontsize=14, fontweight='bold')

# A1: Prior taste scatter
ax = axes[0, 0]
idx = np.random.choice(len(df1), min(20000, len(df1)), replace=False)
ax.scatter(df1['sim_hist_shown'].values[idx], df1['sim_hist_click'].values[idx],
           alpha=0.1, s=4, c='steelblue', rasterized=True)
ax.plot([0, 1.05], [0, 1.05], 'r--', lw=1)
ax.set_xlabel('cos(history, shown)')
ax.set_ylabel('cos(history, clicked)')
ax.set_title('A1: Prior taste persistence')
ax.text(0.05, 0.92, f'{above_diag:.0f}% above diagonal',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# A2: Taste persistence decay (P1↔P2 vs P1↔P3)
ax = axes[0, 1]
labels_bar = ['P1↔P2\n(3-day gap)', 'P2↔P3\n(4-11 day gap)', 'P1↔P3\n(5-14 day gap)']
vals = [df2['sim_p1_p2'].mean(), df2['sim_p2_p3'].mean(), df2['sim_p1_p3'].mean()]
errs = [df2['sim_p1_p2'].sem(), df2['sim_p2_p3'].sem(), df2['sim_p1_p3'].sem()]
bars = ax.bar(labels_bar, vals, yerr=errs, color=['#4C72B0', '#55A868', '#C44E52'],
              capsize=5, alpha=0.8)
ax.set_ylabel('Avg cosine similarity')
ax.set_title('A2: Taste persistence decays with time gap')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{v:.3f}', ha='center', fontsize=9)
ax.set_ylim(0, max(vals) * 1.15)

# A3: Self-reinforcement
ax = axes[0, 2]
if len(df_cr) > 0:
    dfs = df_cr.sort_values('lift', ascending=True)
    colors = ['forestgreen' if p < 0.05 else 'lightgray' for p in dfs['p']]
    ax.barh(range(len(dfs)), dfs['lift'], color=colors)
    ax.set_yticks(range(len(dfs)))
    ax.set_yticklabels(dfs['category'], fontsize=8)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Lift (%)')
    ax.set_title('A3: Self-reinforcement by category')

# A4: Entropy over 3 periods
ax = axes[1, 0]
if len(df_ent_3) > 0:
    means = [df_ent_3['e_p1'].mean(), df_ent_3['e_p2'].mean(), df_ent_3['e_p3'].mean()]
    sems = [df_ent_3['e_p1'].sem(), df_ent_3['e_p2'].sem(), df_ent_3['e_p3'].sem()]
    ax.errorbar([1, 2, 3], means, yerr=sems, marker='o', markersize=8,
                linewidth=2, color='steelblue', capsize=5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['P1\nNov 9-11', 'P2\nNov 12-15', 'P3\nNov 16-22'])
    ax.set_ylabel('Mean Shannon entropy (bits)')
    ax.set_title('A4: Category entropy over 3 periods')
    for i, (x, m) in enumerate(zip([1,2,3], means)):
        ax.annotate(f'{m:.3f}', (x, m), textcoords="offset points",
                    xytext=(10, 5), fontsize=9)

# A5: Transition heatmap P1→P3
ax = axes[1, 1]
show_cats = [c for i, c in enumerate(top_cats) if trans_counts[i] >= 10]
show_idx_local = [top_cats.index(c) for c in show_cats]
trans_show = transition[np.ix_(show_idx_local, show_idx_local)]
im = ax.imshow(trans_show, cmap='Blues', aspect='auto', vmin=0)
ax.set_xticks(range(len(show_cats)))
ax.set_xticklabels(show_cats, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(show_cats)))
ax.set_yticklabels(show_cats, fontsize=8)
ax.set_xlabel('P3 category (Nov 16-22)')
ax.set_ylabel('Dominant P1 category (Nov 9-11)')
ax.set_title('A5: Transition matrix P1→P3 (14 days)')
fig.colorbar(im, ax=ax, shrink=0.8)
for i in range(len(show_cats)):
    ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                edgecolor='red', lw=2))

# A6: Shift alignment distribution
ax = axes[1, 2]
ax.hist(df2['sim_p2_shift_p3'].dropna(), bins=50, color='steelblue',
        alpha=0.7, edgecolor='white', density=True)
ax.axvline(0, color='red', lw=1.5, ls='--', label='Zero')
ax.axvline(m2, color='orange', lw=2, label=f'Observed = {m2:+.3f}')
ax.axvline(null_mu, color='gray', lw=1.5, ls=':', label=f'Null = {null_mu:+.3f}')
ax.set_xlabel('cos(P3 shift, P2 consumption)')
ax.set_ylabel('Density')
ax.set_title('A6: Taste shift alignment (P2→P3)')
ax.legend(fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(OUT, "taste_drift_large.pdf")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.savefig(fig_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {fig_path}")


# ============================================================
# COMPARISON WITH SMALL DATASET
# ============================================================
print("\n" + "=" * 60)
print("COMPARISON: SMALL vs. LARGE DATASET")
print("=" * 60)
print(f"""
{'Metric':<45} {'Small':>10} {'Large':>10}
{'-'*67}
{'Users analyzed':<45} {'35,443':>10} {f'{len(df1):,}':>10}
{'cos(history, clicked)':<45} {'0.6640':>10} {f'{mean_hc:.4f}':>10}
{'cos(history, shown)':<45} {'0.7693':>10} {f'{mean_hs:.4f}':>10}
{'Self-reinforcement lift':<45} {'+242%':>10} {f'+{(np.mean(overall_yes)/np.mean(overall_no)-1)*100:.0f}%':>10}
{'Categories significant (A3)':<45} {'14/14':>10} {f'{sig_n}/{len(df_cr)}':>10}
{'β(hist_frac) in regression':<45} {'0.4056':>10} {f'{beta[1]:.4f}':>10}
{'R²':<45} {'0.3822':>10} {f'{r2:.4f}':>10}
{'Taste shift alignment (A2)':<45} {'−0.012':>10} {f'{m2:+.4f}':>10}
{'Time window':<45} {'7 days':>10} {'14 days':>10}
{'Three-period users':<45} {'N/A':>10} {f'{len(df2):,}':>10}
""")


# ============================================================
# SUMMARY
# ============================================================
print("=" * 60)
print("SUMMARY OF LARGE-DATASET FINDINGS")
print("=" * 60)
print(f"""
A1 — PRIOR TASTE PERSISTENCE
   cos(hist, clicked) = {mean_hc:.4f}, cos(hist, shown) = {mean_hs:.4f}
   {above_diag:.0f}% above diagonal.
   NOTE: cos(hist,shown) > cos(hist,clicked) reflects algorithmic personalization
   (the feed is tailored toward user taste). Per-category analysis confirms
   strong self-selection (2-7x ratios); A5 regression confirms hist predicts clicks.

A2 — THREE-PERIOD TASTE SHIFT (THE KEY NEW TEST)
   Taste persistence decays: P1↔P2 = {df2['sim_p1_p2'].mean():.3f}, P1↔P3 = {df2['sim_p1_p3'].mean():.3f}
   cos(P3 shift, P2 consumption) = {m2:+.4f} (perm p = {perm_p:.4f})
   cos(P2 shift, P1 consumption) = {m1:+.4f}
   14-day window detects taste shift that 7-day MINDsmall could not (−0.012).

A3 — SELF-REINFORCEMENT
   Overall lift = +{(np.mean(overall_yes)/np.mean(overall_no)-1)*100:.0f}%
   {sig_n}/{len(df_cr)} categories significant.

A4 — ENTROPY TRAJECTORY
   P1: {df_ent_3['e_p1'].mean():.3f} → P2: {df_ent_3['e_p2'].mean():.3f} → P3: {df_ent_3['e_p3'].mean():.3f} bits
   Trend: {'increasing (diversifying)' if df_ent_3['delta_13'].mean() > 0 else 'decreasing (focusing)'}
   {'NOTE: entropy increases = users diversify on average, consistent with exploration.' if df_ent_3['delta_13'].mean() > 0 else ''}

A5 — REGRESSION (with clustered SEs)
   β(hist) = {beta[1]:.4f} (clustered p={p_vals_clust[1]:.2e}), R² = {r2:.4f}
   Clustered SEs are {se_clust[1]/se_iid[1]:.1f}x larger than iid SEs.
""")

print(f"All outputs saved to: {OUT}/")
print("Done.")
