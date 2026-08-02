"""
Metric Robustness Check for Taste Dynamics Findings
====================================================
Tests whether the taste shift alignment (Finding 2) and backfire (Finding 4)
results are robust to the choice of similarity/alignment metric.

Replicates Analysis B of exposure_effect_full.py using:
  - Cosine similarity (original)
  - Raw inner product
  - Spearman rank correlation
  - Component-wise sign agreement (fraction of categories where shift and
    consumption point in the same direction)

All analyses use the category-share taste vector on the simplex, consistent
with the theory model.

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
print("METRIC ROBUSTNESS — Taste Shift Alignment")
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

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return np.dot(a, b) / (na * nb)

def inner_product(a, b):
    return np.dot(a, b)

def spearman_corr(a, b):
    """Spearman rank correlation between two vectors."""
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    r, _ = stats.spearmanr(a, b)
    return r

def sign_agreement(shift, direction):
    """Fraction of components where shift and direction have same sign,
    excluding components where either is zero."""
    nonzero = (shift != 0) & (direction != 0)
    if nonzero.sum() == 0:
        return np.nan
    return np.mean(np.sign(shift[nonzero]) == np.sign(direction[nonzero]))

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

# Per-user history
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

# P3 new clicks
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
# 3. MULTI-METRIC ALIGNMENT ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("MULTI-METRIC ALIGNMENT: shift = p3_vec - p1_vec")
print("=" * 70)

# For each user, compute alignment between taste shift and four reference vectors
# using five different alignment metrics
reference_vectors = ['p2_click_vec', 'p2_shown_vec', 'p2_shown_clicked_vec', 'p2_shown_notclicked_vec']
ref_labels = ['P2 clicks', 'P2 shown (all)', 'P2 shown+clicked', 'P2 shown+NOT clicked']

metric_names = ['Cosine', 'Inner product', 'Spearman ρ', 'Sign agreement']

# Collect results
all_results = {}

for ref_key, ref_label in zip(reference_vectors, ref_labels):
    print(f"\n--- Reference: {ref_label} ---")

    cos_vals, ip_vals, spear_vals, sign_vals = [], [], [], []

    for uid in eligible_uids:
        v = user_vecs[uid]
        shift = v['p3_vec'] - v['p1_vec']
        ref = v[ref_key]

        cs = cosine_sim(shift, ref)
        ip = inner_product(shift, ref)
        sp = spearman_corr(shift, ref)
        sa = sign_agreement(shift, ref)

        if not np.isnan(cs):
            cos_vals.append(cs)
        if not np.isnan(ip):
            ip_vals.append(ip)
        if not np.isnan(sp):
            spear_vals.append(sp)
        if not np.isnan(sa):
            sign_vals.append(sa)

    cos_vals = np.array(cos_vals)
    ip_vals = np.array(ip_vals)
    spear_vals = np.array(spear_vals)
    sign_vals = np.array(sign_vals)

    print(f"\n  {'Metric':<25s} {'N':>8s} {'Mean':>10s} {'t':>10s} {'p':>12s} {'Cohen d':>10s}")
    print("  " + "-" * 78)

    ref_results = {}
    for name, vals, null_val in [
        ('Cosine', cos_vals, 0),
        ('Inner product', ip_vals, 0),
        ('Spearman ρ', spear_vals, 0),
        ('Sign agreement', sign_vals, 0.5),  # null = random sign agreement = 50%
    ]:
        if len(vals) < 10:
            continue
        t_val, p_val = stats.ttest_1samp(vals, null_val)
        d_val = (np.mean(vals) - null_val) / np.std(vals, ddof=1) if np.std(vals, ddof=1) > 0 else 0
        print(f"  {name:<25s} {len(vals):>8,} {np.mean(vals):>10.4f} {t_val:>10.2f} {p_val:>12.2e} {d_val:>10.4f}")
        ref_results[name] = {
            'n': int(len(vals)),
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)),
            'median': float(np.median(vals)),
            't': float(t_val),
            'p': float(p_val),
            'd': float(d_val),
        }

    all_results[ref_label] = ref_results


# ============================================================
# 4. BACKFIRE ROBUSTNESS
# ============================================================
print("\n" + "=" * 70)
print("BACKFIRE ROBUSTNESS — shown-not-clicked categories")
print("=" * 70)
print("For each (user, category) where articles were shown but not clicked,")
print("does taste move toward or away from that category?")

toward_deltas, away_deltas = [], []
toward_count, away_count, zero_count = 0, 0, 0

for uid in eligible_uids:
    v = user_vecs[uid]
    for ci in range(n_cats):
        # Was shown in this category but did NOT click
        was_shown = v.get('p2_shown_notclicked_vec', np.zeros(n_cats))[ci] > 0.001
        if not was_shown:
            continue

        delta = v['p3_vec'][ci] - v['p1_vec'][ci]
        if delta > 0.001:
            toward_count += 1
            toward_deltas.append(delta)
        elif delta < -0.001:
            away_count += 1
            away_deltas.append(abs(delta))
        else:
            zero_count += 1

toward_deltas = np.array(toward_deltas)
away_deltas = np.array(away_deltas)
total_nz = toward_count + away_count

print(f"\n  Total (user, category) pairs: {toward_count + away_count + zero_count:,}")
print(f"  Toward (taste increases):     {toward_count:,} ({toward_count/total_nz*100:.1f}%)")
print(f"  Away (taste decreases):       {away_count:,} ({away_count/total_nz*100:.1f}%)")
print(f"  No change:                    {zero_count:,}")

if len(toward_deltas) > 0 and len(away_deltas) > 0:
    mean_toward = np.mean(toward_deltas)
    mean_away = np.mean(away_deltas)
    t_bf, p_bf = stats.ttest_ind(away_deltas, toward_deltas, equal_var=False)
    print(f"\n  Mean |Δ| toward:  {mean_toward:.4f}")
    print(f"  Mean |Δ| away:    {mean_away:.4f}")
    print(f"  t-test (away > toward): t = {t_bf:.2f}, p = {p_bf:.2e}")

    # Binomial test: is the fraction moving away significantly > 50%?
    binom_p = stats.binom_test(away_count, toward_count + away_count, 0.5)
    print(f"  Binomial test (away > 50%): p = {binom_p:.2e}")

backfire_results = {
    'toward_count': int(toward_count),
    'away_count': int(away_count),
    'zero_count': int(zero_count),
    'frac_away': float(away_count / total_nz) if total_nz > 0 else 0,
    'mean_toward': float(np.mean(toward_deltas)) if len(toward_deltas) > 0 else 0,
    'mean_away': float(np.mean(away_deltas)) if len(away_deltas) > 0 else 0,
}


# ============================================================
# 5. SUMMARY TABLE
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: ALIGNMENT OF TASTE SHIFT WITH REFERENCE VECTORS")
print("=" * 70)

print(f"\n  {'Reference vector':<25s}", end="")
for m in metric_names:
    print(f" {m:>15s}", end="")
print()
print("  " + "-" * 90)

for ref_label in ref_labels:
    print(f"  {ref_label:<25s}", end="")
    for m in metric_names:
        if m in all_results[ref_label]:
            val = all_results[ref_label][m]['mean']
            sig = all_results[ref_label][m]['p'] < 0.001
            marker = "***" if sig else ""
            print(f" {val:>11.4f}{marker}", end="")
        else:
            print(f" {'---':>15s}", end="")
    print()

print("\n  *** = p < 0.001")
print(f"\n  Note: Sign agreement null is 0.50 (random); all others null is 0.00")


# ============================================================
# SAVE
# ============================================================
output = {
    'n_eligible': len(eligible_uids),
    'n_cats': n_cats,
    'alignment_results': {},
    'backfire': backfire_results,
}

for ref_label in ref_labels:
    output['alignment_results'][ref_label] = all_results[ref_label]

with open(os.path.join(OUT, 'metric_robustness_results.json'), 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {OUT}/metric_robustness_results.json")
print("=" * 70)
