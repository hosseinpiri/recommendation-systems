"""
Hellinger / Total Variation Robustness for Taste Dynamics
=========================================================
Tests whether taste moves closer to consumed content using distribution-native
distance metrics (Hellinger, Total Variation) rather than cosine alignment.

For each user, computes:
  d(θ_P3, c_P2) - d(θ_P1, c_P2)

If negative on average, taste moved closer to consumed content from P1 to P3.
We test this for four reference vectors: P2 clicks, P2 shown (all),
P2 shown+clicked, P2 shown+NOT clicked.

Author: Hossein Piri / Claude
Date: 2026-04-22
"""

import os, json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
import functools
import warnings
warnings.filterwarnings('ignore')
print = functools.partial(print, flush=True)

BASE = "/Users/piri/Desktop/Recommendation Systems/Mind-Data-Large"
OUT = "/Users/piri/Desktop/Recommendation Systems/code/output_full"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("HELLINGER / TV ROBUSTNESS — Taste Dynamics")
print("=" * 70)

# ============================================================
# 0. LOAD NEWS METADATA
# ============================================================
print("\nLoading news metadata...")
news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract',
             'url', 'title_entities', 'abstract_entities']

dfs_news = []
for split in ['MINDlarge_train', 'MINDlarge_dev', 'MINDlarge_test']:
    path = os.path.join(BASE, split, "news.tsv")
    df = pd.read_csv(path, sep='\t', header=None, names=news_cols, usecols=[0, 1, 2])
    dfs_news.append(df)

news = pd.concat(dfs_news).drop_duplicates(subset='news_id')
news_cat = dict(zip(news['news_id'], news['category']))
print(f"  Articles: {len(news):,}")

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

def hellinger_dist(p, q):
    """Hellinger distance between two probability distributions."""
    return np.sqrt(0.5 * np.sum((np.sqrt(np.maximum(p, 0)) - np.sqrt(np.maximum(q, 0)))**2))

def total_variation(p, q):
    """Total variation distance between two probability distributions."""
    return 0.5 * np.sum(np.abs(p - q))

def kl_divergence(p, q, eps=1e-10):
    """KL divergence D(p||q) with Laplace smoothing."""
    p_smooth = p + eps
    q_smooth = q + eps
    p_smooth = p_smooth / p_smooth.sum()
    q_smooth = q_smooth / q_smooth.sum()
    return np.sum(p_smooth * np.log(p_smooth / q_smooth))

def js_divergence(p, q, eps=1e-10):
    """Jensen-Shannon divergence (symmetric KL)."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return np.dot(a, b) / (na * nb)

def cohens_d_one_sample(x):
    s = np.std(x, ddof=1)
    return np.mean(x) / s if s > 0 else 0.0

def parse_history(s):
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '':
        return []
    return s.strip().split()

def parse_imp_fast(s):
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '':
        return [], []
    clicked, shown = [], []
    for item in s.strip().split():
        nid, label = item.rsplit('-', 1)
        shown.append(nid)
        if label == '1':
            clicked.append(nid)
    return clicked, shown


# ============================================================
# 1. LOAD AND PARSE BEHAVIORS
# ============================================================
beh_cols = ['impression_id', 'user_id', 'timestamp', 'click_history', 'impressions']

print("\nLoading behaviors...")
beh_train = pd.read_csv(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
                         sep='\t', header=None, names=beh_cols)
beh_dev = pd.read_csv(os.path.join(BASE, "MINDlarge_dev", "behaviors.tsv"),
                       sep='\t', header=None, names=beh_cols)
beh_labeled = pd.concat([beh_train, beh_dev], ignore_index=True)
del beh_train, beh_dev
beh_labeled['time'] = pd.to_datetime(beh_labeled['timestamp'], format='%m/%d/%Y %I:%M:%S %p')
beh_labeled['day'] = beh_labeled['time'].dt.day
print(f"  Labeled impressions: {len(beh_labeled):,}")

def day_to_period(d):
    if 9 <= d <= 11: return 'P1'
    if 12 <= d <= 15: return 'P2'
    return None

beh_labeled['period'] = beh_labeled['day'].map(day_to_period)

print("  Parsing impressions...")
parsed = beh_labeled['impressions'].apply(parse_imp_fast)
beh_labeled['clicked_list'] = parsed.apply(lambda x: x[0])
beh_labeled['shown_list'] = parsed.apply(lambda x: x[1])
del parsed

print("  Parsing histories...")
beh_labeled['hist_list'] = beh_labeled['click_history'].apply(parse_history)
beh_labeled['n_history'] = beh_labeled['hist_list'].apply(len)

idx_max_hist = beh_labeled.groupby('user_id')['n_history'].idxmax()
user_hist = beh_labeled.loc[idx_max_hist, ['user_id', 'hist_list']].set_index('user_id')['hist_list'].to_dict()
del idx_max_hist

beh_p1p2 = beh_labeled[beh_labeled['period'].isin(['P1', 'P2'])].copy()
print(f"  P1+P2 impressions: {len(beh_p1p2):,}")

print("  Aggregating P1/P2 data...")
user_data = defaultdict(lambda: {'P1_click': [], 'P2_click': [], 'P2_shown': []})
uids_arr = beh_p1p2['user_id'].values
periods_arr = beh_p1p2['period'].values
clicked_arr = beh_p1p2['clicked_list'].values
shown_arr = beh_p1p2['shown_list'].values

for i in range(len(uids_arr)):
    uid = uids_arr[i]
    p = periods_arr[i]
    if p == 'P1':
        user_data[uid]['P1_click'].extend(clicked_arr[i])
    elif p == 'P2':
        user_data[uid]['P2_click'].extend(clicked_arr[i])
        user_data[uid]['P2_shown'].extend(shown_arr[i])

del beh_p1p2, uids_arr, periods_arr, clicked_arr, shown_arr, beh_labeled
print(f"  Users with P1/P2 data: {len(user_data):,}")

# Load test set for P3
print("\nLoading test behaviors...")
beh_test = pd.read_csv(os.path.join(BASE, "MINDlarge_test", "behaviors.tsv"),
                        sep='\t', header=None, names=beh_cols)
beh_test['hist_list'] = beh_test['click_history'].apply(parse_history)
beh_test['n_history'] = beh_test['hist_list'].apply(len)
idx_max_test = beh_test.groupby('user_id')['n_history'].idxmax()
user_test_hist = beh_test.loc[idx_max_test, ['user_id', 'hist_list']].set_index('user_id')['hist_list'].to_dict()
del beh_test, idx_max_test
print(f"  Users in test: {len(user_test_hist):,}")

print("  Inferring P3 clicks...")
user_p3_clicks = {}
overlap = set(user_hist.keys()) & set(user_test_hist.keys())
for uid in overlap:
    old = set(user_hist[uid])
    seen = set()
    new_clicks = []
    for nid in user_test_hist[uid]:
        if nid not in old and nid not in seen:
            new_clicks.append(nid)
            seen.add(nid)
    if new_clicks:
        user_p3_clicks[uid] = new_clicks

del user_test_hist
print(f"  Users with P3 growth: {len(user_p3_clicks):,}")


# ============================================================
# 2. COMPUTE CATEGORY VECTORS
# ============================================================
print("\nComputing category vectors...")

user_vecs = {}
for uid in user_data:
    rec = user_data[uid]
    p1_raw = nids_to_cat_vec(rec['P1_click'])
    p2_click_raw = nids_to_cat_vec(rec['P2_click'])
    p2_shown_raw = nids_to_cat_vec(rec['P2_shown'])

    n_p1, n_p2_click, n_p2_shown = int(p1_raw.sum()), int(p2_click_raw.sum()), int(p2_shown_raw.sum())

    if uid not in user_p3_clicks:
        continue
    p3_raw = nids_to_cat_vec(user_p3_clicks[uid])
    n_p3 = int(p3_raw.sum())

    if n_p1 < 2 or n_p2_click < 2 or n_p2_shown < 5 or n_p3 < 2:
        continue

    p2_clicked_set = set(rec['P2_click'])
    p2_shown_clicked = [nid for nid in rec['P2_shown'] if nid in p2_clicked_set]
    p2_shown_notclicked = [nid for nid in rec['P2_shown'] if nid not in p2_clicked_set]

    user_vecs[uid] = {
        'p1_vec': normalize(p1_raw),
        'p2_click_vec': normalize(p2_click_raw),
        'p2_shown_vec': normalize(p2_shown_raw),
        'p2_shown_clicked_vec': normalize(nids_to_cat_vec(p2_shown_clicked)),
        'p2_shown_notclicked_vec': normalize(nids_to_cat_vec(p2_shown_notclicked)),
        'p3_vec': normalize(p3_raw),
    }

eligible_uids = list(user_vecs.keys())
print(f"Eligible users: {len(eligible_uids):,}")

del user_data, user_p3_clicks


# ============================================================
# 3. DISTANCE-BASED "DID TASTE MOVE CLOSER?" ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("DISTANCE-BASED ANALYSIS: Did taste move closer to the reference?")
print("For each user: Δd = d(θ_P3, ref) - d(θ_P1, ref)")
print("Negative Δd means taste moved CLOSER to the reference vector.")
print("=" * 70)

reference_vectors = ['p2_click_vec', 'p2_shown_vec', 'p2_shown_clicked_vec', 'p2_shown_notclicked_vec']
ref_labels = ['P2 clicks', 'P2 shown (all)', 'P2 shown+clicked', 'P2 shown+NOT clicked']

distance_fns = {
    'Hellinger': hellinger_dist,
    'Total variation': total_variation,
    'JS divergence': js_divergence,
    'Cosine distance': lambda p, q: 1 - cosine_sim(p, q) if not np.isnan(cosine_sim(p, q)) else np.nan,
}

all_results = {}

for ref_key, ref_label in zip(reference_vectors, ref_labels):
    print(f"\n--- Reference: {ref_label} ---")
    print(f"  {'Metric':<20s} {'N':>8s} {'Mean Δd':>10s} {'Median Δd':>10s} {'t':>10s} {'p':>12s} {'Cohen d':>10s} {'% closer':>10s}")
    print("  " + "-" * 95)

    ref_results = {}

    for dist_name, dist_fn in distance_fns.items():
        deltas = []
        closer_count = 0

        for uid in eligible_uids:
            v = user_vecs[uid]
            ref = v[ref_key]

            d_p1 = dist_fn(v['p1_vec'], ref)
            d_p3 = dist_fn(v['p3_vec'], ref)

            if np.isnan(d_p1) or np.isnan(d_p3):
                continue

            delta = d_p3 - d_p1  # negative = moved closer
            deltas.append(delta)
            if delta < 0:
                closer_count += 1

        deltas = np.array(deltas)
        if len(deltas) < 10:
            continue

        t_val, p_val = stats.ttest_1samp(deltas, 0)
        d_val = cohens_d_one_sample(deltas)
        pct_closer = closer_count / len(deltas) * 100

        print(f"  {dist_name:<20s} {len(deltas):>8,} {np.mean(deltas):>10.4f} {np.median(deltas):>10.4f} "
              f"{t_val:>10.2f} {p_val:>12.2e} {d_val:>10.4f} {pct_closer:>9.1f}%")

        ref_results[dist_name] = {
            'n': int(len(deltas)),
            'mean_delta': float(np.mean(deltas)),
            'median_delta': float(np.median(deltas)),
            'std_delta': float(np.std(deltas, ddof=1)),
            't': float(t_val),
            'p': float(p_val),
            'd': float(d_val),
            'pct_closer': float(pct_closer),
        }

    all_results[ref_label] = ref_results


# ============================================================
# 4. SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Δd = d(θ_P3, ref) - d(θ_P1, ref)")
print("Negative = taste moved closer to reference")
print("=" * 70)

print(f"\n  {'Reference':<25s}", end="")
for dist_name in distance_fns:
    print(f" {dist_name:>18s}", end="")
print()
print("  " + "-" * 100)

for ref_label in ref_labels:
    print(f"  {ref_label:<25s}", end="")
    for dist_name in distance_fns:
        if dist_name in all_results[ref_label]:
            val = all_results[ref_label][dist_name]['mean_delta']
            sig = all_results[ref_label][dist_name]['p'] < 0.001
            marker = "***" if sig else ""
            print(f" {val:>14.4f}{marker}", end="")
        else:
            print(f" {'---':>18s}", end="")
    print()

print("\n  *** = p < 0.001")
print("  Negative values = taste moved closer to the reference")

# Also print % closer
print(f"\n  {'Reference':<25s}", end="")
for dist_name in distance_fns:
    print(f" {dist_name:>18s}", end="")
print()
print("  " + "-" * 100)

for ref_label in ref_labels:
    print(f"  {ref_label:<25s}", end="")
    for dist_name in distance_fns:
        if dist_name in all_results[ref_label]:
            val = all_results[ref_label][dist_name]['pct_closer']
            print(f" {val:>17.1f}%", end="")
        else:
            print(f" {'---':>18s}", end="")
    print()

print("\n  (% of users whose taste moved closer to the reference)")


# ============================================================
# SAVE
# ============================================================
output = {
    'n_eligible': len(eligible_uids),
    'n_cats': n_cats,
    'distance_results': all_results,
}

with open(os.path.join(OUT, 'hellinger_tv_dynamics_results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {OUT}/hellinger_tv_dynamics_results.json")
print("=" * 70)
