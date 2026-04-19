"""
Exposure Effect Analysis: Does Recommendation Exposure Shift Taste?
===================================================================
Tests whether the category composition of articles SHOWN (recommended)
to users in P2 predicts their taste shift from P1 to P3, after
controlling for what they actually CLICKED in P2.

Uses MINDlarge three-period design:
  P1: Nov 9-11   (baseline taste from clicks)
  P2: Nov 12-15  (treatment period: both shown and clicked observed)
  P3: Nov 16-22  (outcome taste, inferred from test history growth)

Analyses:
  A. Stacked OLS: p3_frac = β0 + β1*p1 + β2*p2_click + β3*p2_shown + ε
  B. Cosine alignment of taste shift with shown vs clicked
  C. Shown-but-NOT-clicked (cleanest identification)
  D. Dose-response within shown-not-clicked
  E. Per-category breakdown

Author: Hossein Piri / Claude
Date: 2026-03-30
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
import functools
# Force unbuffered output
print = functools.partial(print, flush=True)

BASE = "/Users/piri/Desktop/Recommendation Systems/Dataset"
OUT = "/Users/piri/Desktop/Recommendation Systems/code/output_large"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("EXPOSURE EFFECT ANALYSIS — Does Recommendation Shift Taste?")
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

def parse_history(s):
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '':
        return []
    return s.strip().split()

def parse_impressions_labeled(s):
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '':
        return [], []
    clicked, shown = [], []
    for item in s.strip().split():
        nid, label = item.rsplit('-', 1)
        shown.append(nid)
        if label == '1':
            clicked.append(nid)
    return clicked, shown

def ols_clustered(y, X, cluster_ids):
    """OLS with clustered standard errors (fast, using sorted-index splitting)."""
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # IID standard errors
    XtX_inv = np.linalg.inv(X.T @ X)
    se_iid = np.sqrt(ss_res / (n - k) * np.diag(XtX_inv))

    # Clustered (sandwich) standard errors — fast via sorted index splitting
    # Sort by cluster, then split at boundaries
    sort_idx = np.argsort(cluster_ids)
    X_sorted = X[sort_idx]
    e_sorted = resid[sort_idx]
    cl_sorted = cluster_ids[sort_idx]

    # Find cluster boundaries
    boundaries = np.where(np.diff(cl_sorted) != 0)[0] + 1
    n_clusters = len(boundaries) + 1

    # Compute per-cluster scores: X_g.T @ e_g for each cluster
    # Split into groups and compute scores
    X_groups = np.split(X_sorted, boundaries)
    e_groups = np.split(e_sorted, boundaries)

    B = np.zeros((k, k))
    for X_g, e_g in zip(X_groups, e_groups):
        score = X_g.T @ e_g  # (k,)
        B += np.outer(score, score)

    correction = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))
    V_clust = correction * XtX_inv @ B @ XtX_inv
    se_clust = np.sqrt(np.diag(V_clust))

    t_clust = beta / se_clust
    p_clust = 2 * (1 - stats.t.cdf(np.abs(t_clust), df=n_clusters - 1))

    return {
        'beta': beta, 'se_iid': se_iid, 'se_clust': se_clust,
        't_clust': t_clust, 'p_clust': p_clust,
        'r2': r2, 'n_obs': n, 'n_clusters': n_clusters
    }

def cohens_d_one_sample(x):
    return np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 0.0

def benjamini_hochberg(pvals):
    """Return BH-adjusted p-values."""
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        if i == n - 1:
            adjusted[order[i]] = pvals[order[i]]
        else:
            adjusted[order[i]] = min(adjusted[order[i + 1]],
                                     pvals[order[i]] * n / rank)
    return adjusted


# ============================================================
# 1. LOAD AND PARSE BEHAVIORS (vectorized for speed)
# ============================================================
beh_cols = ['impression_id', 'user_id', 'timestamp', 'click_history', 'impressions']

print("\nLoading train behaviors...", flush=True)
beh_train = pd.read_csv(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
                         sep='\t', header=None, names=beh_cols)
print("Loading dev behaviors...", flush=True)
beh_dev = pd.read_csv(os.path.join(BASE, "MINDlarge_dev", "behaviors.tsv"),
                       sep='\t', header=None, names=beh_cols)

beh_labeled = pd.concat([beh_train, beh_dev], ignore_index=True)
del beh_train, beh_dev  # free memory
beh_labeled['time'] = pd.to_datetime(beh_labeled['timestamp'], format='%m/%d/%Y %I:%M:%S %p')
beh_labeled['day'] = beh_labeled['time'].dt.day
print(f"  Labeled impressions: {len(beh_labeled):,}", flush=True)

# Assign periods by day number (Nov 9-11 = P1, Nov 12-15 = P2)
def day_to_period(d):
    if 9 <= d <= 11: return 'P1'
    if 12 <= d <= 15: return 'P2'
    return None

beh_labeled['period'] = beh_labeled['day'].map(day_to_period)

# Parse impressions vectorized (apply is much faster than iterrows)
print("  Parsing impressions...", flush=True)

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

parsed = beh_labeled['impressions'].apply(parse_imp_fast)
beh_labeled['clicked_list'] = parsed.apply(lambda x: x[0])
beh_labeled['shown_list'] = parsed.apply(lambda x: x[1])
del parsed

# Parse histories
print("  Parsing histories...", flush=True)
beh_labeled['hist_list'] = beh_labeled['click_history'].apply(parse_history)
beh_labeled['n_history'] = beh_labeled['hist_list'].apply(len)

# --- Build per-user aggregates using groupby + Python loop on groups ---
print("\nBuilding per-user aggregates (vectorized)...", flush=True)

# Get longest history per user (for baseline taste)
idx_max_hist = beh_labeled.groupby('user_id')['n_history'].idxmax()
user_hist_df = beh_labeled.loc[idx_max_hist, ['user_id', 'hist_list']].set_index('user_id')
user_hist = user_hist_df['hist_list'].to_dict()
del user_hist_df, idx_max_hist
print(f"  Users with history: {len(user_hist):,}", flush=True)

# Filter to P1 and P2 only, aggregate clicks and shown per user per period
beh_p1p2 = beh_labeled[beh_labeled['period'].isin(['P1', 'P2'])].copy()
print(f"  P1+P2 impressions: {len(beh_p1p2):,}", flush=True)

# Group by user_id and period, concatenate all clicked/shown lists
print("  Aggregating P1 clicks, P2 clicks, P2 shown...", flush=True)

user_data = defaultdict(lambda: {'P1_click': [], 'P2_click': [], 'P2_shown': []})

# Use numpy arrays for fast iteration
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

del beh_p1p2, uids_arr, periods_arr, clicked_arr, shown_arr
print(f"  Users with P1/P2 data: {len(user_data):,}", flush=True)

# --- Load test set for P3 inference ---
print("\nLoading test behaviors...", flush=True)
beh_test = pd.read_csv(os.path.join(BASE, "MINDlarge_test", "behaviors.tsv"),
                        sep='\t', header=None, names=beh_cols)
print(f"  Test impressions: {len(beh_test):,}", flush=True)

print("  Parsing test histories...", flush=True)
beh_test['hist_list'] = beh_test['click_history'].apply(parse_history)
beh_test['n_history'] = beh_test['hist_list'].apply(len)

# Get longest test history per user
idx_max_test = beh_test.groupby('user_id')['n_history'].idxmax()
user_test_hist = beh_test.loc[idx_max_test, ['user_id', 'hist_list']].set_index('user_id')['hist_list'].to_dict()
del beh_test, idx_max_test
print(f"  Users in test: {len(user_test_hist):,}", flush=True)

# Compute P3 new clicks = test_history - labeled_history
print("  Inferring P3 clicks from test history growth...", flush=True)
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
print(f"  Users with P3 growth: {len(user_p3_clicks):,}", flush=True)


# ============================================================
# 3. COMPUTE CATEGORY VECTORS
# ============================================================
print("\nComputing category vectors...")

user_vecs = {}
n_eligible = 0

for uid in user_data:
    rec = user_data[uid]

    p1_raw = nids_to_cat_vec(rec['P1_click'])
    p2_click_raw = nids_to_cat_vec(rec['P2_click'])
    p2_shown_raw = nids_to_cat_vec(rec['P2_shown'])

    n_p1 = int(p1_raw.sum())
    n_p2_click = int(p2_click_raw.sum())
    n_p2_shown = int(p2_shown_raw.sum())

    if uid not in user_p3_clicks:
        continue
    p3_raw = nids_to_cat_vec(user_p3_clicks[uid])
    n_p3 = int(p3_raw.sum())

    # Eligibility: sufficient clicks in all periods + sufficient shown articles
    if n_p1 < 2 or n_p2_click < 2 or n_p2_shown < 5 or n_p3 < 2:
        continue

    n_eligible += 1
    user_vecs[uid] = {
        'p1_vec': normalize(p1_raw),
        'p2_click_vec': normalize(p2_click_raw),
        'p2_shown_vec': normalize(p2_shown_raw),
        'p3_vec': normalize(p3_raw),
        'p1_raw': p1_raw,
        'p2_click_raw': p2_click_raw,
        'p2_shown_raw': p2_shown_raw,
        'p3_raw': p3_raw,
    }

eligible_uids = list(user_vecs.keys())
print(f"Eligible users (P1≥2, P2click≥2, P2shown≥5, P3≥2): {n_eligible:,}")


# ============================================================
# ANALYSIS A: STACKED OLS REGRESSION
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS A: Stacked OLS — p3_frac = β0 + β1*p1 + β2*p2_click + β3*p2_shown")
print("=" * 70)

uid_idx = {uid: i for i, uid in enumerate(eligible_uids)}

# Build matrices for OLS (vectorized)
n_users = len(eligible_uids)
p1_mat = np.zeros((n_users, n_cats))
p2c_mat = np.zeros((n_users, n_cats))
p2s_mat = np.zeros((n_users, n_cats))
p3_mat = np.zeros((n_users, n_cats))

for i, uid in enumerate(eligible_uids):
    v = user_vecs[uid]
    p1_mat[i] = v['p1_vec']
    p2c_mat[i] = v['p2_click_vec']
    p2s_mat[i] = v['p2_shown_vec']
    p3_mat[i] = v['p3_vec']

# Flatten to (user, category) rows, filter out all-zero rows
y_flat = p3_mat.ravel()
p1_flat = p1_mat.ravel()
p2c_flat = p2c_mat.ravel()
p2s_flat = p2s_mat.ravel()
cl_flat = np.repeat(np.arange(n_users), n_cats)

# Keep rows where at least one variable is non-zero
mask = (p1_flat != 0) | (p2c_flat != 0) | (p2s_flat != 0) | (y_flat != 0)
y_a = y_flat[mask]
X_a = np.column_stack([np.ones(mask.sum()), p1_flat[mask], p2c_flat[mask], p2s_flat[mask]])
cl_a = cl_flat[mask]
del p1_mat, p2c_mat, p2s_mat, p3_mat, y_flat, p1_flat, p2c_flat, p2s_flat, cl_flat, mask

res_a = ols_clustered(y_a, X_a, cl_a)

var_names = ['Intercept', 'P1 taste (β1)', 'P2 clicks (β2)', 'P2 shown (β3)']
print(f"\nN obs = {res_a['n_obs']:,}   N clusters = {res_a['n_clusters']:,}   R² = {res_a['r2']:.4f}")
print(f"\n{'Variable':<20s} {'Coef':>8s} {'SE(clust)':>10s} {'t':>8s} {'p':>12s}")
print("-" * 62)
for i, name in enumerate(var_names):
    print(f"{name:<20s} {res_a['beta'][i]:8.4f} {res_a['se_clust'][i]:10.4f} "
          f"{res_a['t_clust'][i]:8.2f} {res_a['p_clust'][i]:12.2e}")

beta3 = res_a['beta'][3]
p3_val = res_a['p_clust'][3]
print(f"\n→ β3 (exposure effect) = {beta3:.4f}, p = {p3_val:.2e}")
if beta3 > 0 and p3_val < 0.05:
    print("  ✓ Showing articles in a category shifts taste toward that category,")
    print("    AFTER controlling for baseline taste and actual consumption.")
else:
    print("  Exposure effect is not statistically significant after controls.")


# ============================================================
# ANALYSIS B: COSINE ALIGNMENT OF TASTE SHIFT
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS B: Cosine alignment — shift = p3 - p1")
print("=" * 70)

shown_aligns, click_aligns = [], []
for uid in eligible_uids:
    v = user_vecs[uid]
    shift = v['p3_vec'] - v['p1_vec']
    cs_shown = cosine_sim(shift, v['p2_shown_vec'])
    cs_click = cosine_sim(shift, v['p2_click_vec'])
    if not np.isnan(cs_shown) and not np.isnan(cs_click):
        shown_aligns.append(cs_shown)
        click_aligns.append(cs_click)

shown_aligns = np.array(shown_aligns)
click_aligns = np.array(click_aligns)

mean_shown = np.mean(shown_aligns)
mean_click = np.mean(click_aligns)

t_shown, p_shown = stats.ttest_1samp(shown_aligns, 0)
t_click, p_click = stats.ttest_1samp(click_aligns, 0)
t_pair, p_pair = stats.ttest_rel(shown_aligns, click_aligns)

d_shown = cohens_d_one_sample(shown_aligns)
d_click = cohens_d_one_sample(click_aligns)

print(f"\nN users with valid alignment: {len(shown_aligns):,}")
print(f"\n{'Alignment':<35s} {'Mean':>8s} {'t':>10s} {'p':>12s} {'Cohen d':>10s}")
print("-" * 78)
print(f"{'cos(shift, P2 shown)':<35s} {mean_shown:8.4f} {t_shown:10.2f} {p_shown:12.2e} {d_shown:10.4f}")
print(f"{'cos(shift, P2 clicks)':<35s} {mean_click:8.4f} {t_click:10.2f} {p_click:12.2e} {d_click:10.4f}")
print(f"\nPaired test (shown vs click alignment): t = {t_pair:.2f}, p = {p_pair:.2e}")

# Permutation test for shown alignment (vectorized)
print("\nPermutation test (5000 shuffles) for shown alignment...")
obs_mean_shown = mean_shown
n_perm_b = 5000

# Build matrices for vectorized cosine: shifts (N x d), shown_vecs (N x d)
shifts_b = []
shown_vecs_b = []
for uid in eligible_uids:
    v = user_vecs[uid]
    shift = v['p3_vec'] - v['p1_vec']
    cs_check = cosine_sim(shift, v['p2_shown_vec'])
    if not np.isnan(cs_check):
        shifts_b.append(shift)
        shown_vecs_b.append(v['p2_shown_vec'])

S = np.array(shifts_b)   # (N, d)
V = np.array(shown_vecs_b)  # (N, d)
# Normalize rows for fast cosine via dot product
S_norm = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

np.random.seed(42)
null_means_b = []
for _ in range(n_perm_b):
    perm = np.random.permutation(len(S_norm))
    # Vectorized cosine: row-wise dot of S_norm and permuted V_norm
    sims = np.sum(S_norm * V_norm[perm], axis=1)
    null_means_b.append(np.nanmean(sims))

null_mu_b = np.mean(null_means_b)
null_sd_b = np.std(null_means_b)
z_b = (obs_mean_shown - null_mu_b) / null_sd_b if null_sd_b > 0 else 0
count_b = sum(1 for n in null_means_b if n >= obs_mean_shown)
perm_p_b = (count_b + 1) / (n_perm_b + 1)

print(f"  Observed mean cos(shift, shown): {obs_mean_shown:.4f}")
print(f"  Null mean: {null_mu_b:.4f}, Null SD: {null_sd_b:.4f}")
print(f"  Z-score: {z_b:.2f}")
print(f"  Permutation p-value: {perm_p_b:.4f}")


# ============================================================
# ANALYSIS C: SHOWN-BUT-NOT-CLICKED (cleanest identification)
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS C: Shown-but-NOT-clicked vs. Control categories")
print("=" * 70)

treatment_deltas = []  # (uid_idx, delta) for shown>0 & click=0
control_deltas = []    # (uid_idx, delta) for shown=0 & click=0
user_paired = defaultdict(lambda: {'treatment': [], 'control': []})
treatment_cat_counts = defaultdict(int)
control_cat_counts = defaultdict(int)

for uid in eligible_uids:
    v = user_vecs[uid]
    for ci in range(n_cats):
        was_shown = v['p2_shown_raw'][ci] > 0
        was_clicked = v['p2_click_raw'][ci] > 0
        delta = v['p3_vec'][ci] - v['p1_vec'][ci]

        if was_clicked:
            continue  # Exclude consumed categories

        if was_shown:
            treatment_deltas.append(delta)
            user_paired[uid]['treatment'].append(delta)
            treatment_cat_counts[ci] += 1
        else:
            control_deltas.append(delta)
            user_paired[uid]['control'].append(delta)
            control_cat_counts[ci] += 1

treatment_deltas = np.array(treatment_deltas)
control_deltas = np.array(control_deltas)

print(f"\nTotal (user, category) pairs:")
print(f"  Treatment (shown, not clicked): {len(treatment_deltas):,}")
print(f"  Control (not shown, not clicked): {len(control_deltas):,}")

print(f"\nTreatment category distribution:")
for ci in sorted(treatment_cat_counts.keys()):
    cname = categories[ci]
    print(f"  {cname:<18s} Treatment={treatment_cat_counts[ci]:>7,}  Control={control_cat_counts.get(ci, 0):>7,}")

# Pooled comparison
t_pool, p_pool = stats.ttest_ind(treatment_deltas, control_deltas)
mean_t = np.mean(treatment_deltas)
mean_c = np.mean(control_deltas)
diff_pool = mean_t - mean_c

print(f"\nPooled comparison:")
print(f"  Treatment mean Δ: {mean_t:+.6f}")
print(f"  Control mean Δ:   {mean_c:+.6f}")
print(f"  Difference:       {diff_pool:+.6f}")
print(f"  t = {t_pool:.2f}, p = {p_pool:.2e}")

# Within-user paired comparison (stronger test)
paired_diffs = []
for uid in user_paired:
    t_vals = user_paired[uid]['treatment']
    c_vals = user_paired[uid]['control']
    if len(t_vals) > 0 and len(c_vals) > 0:
        paired_diffs.append(np.mean(t_vals) - np.mean(c_vals))

paired_diffs = np.array(paired_diffs)
n_paired = len(paired_diffs)
t_paired, p_paired = stats.ttest_1samp(paired_diffs, 0)
d_paired = cohens_d_one_sample(paired_diffs)

print(f"\nWithin-user paired comparison (primary test):")
print(f"  Users with both treatment & control categories: {n_paired:,}")
print(f"  Mean within-user (treatment - control) Δ: {np.mean(paired_diffs):+.6f}")
print(f"  t = {t_paired:.2f}, p = {p_paired:.2e}, Cohen's d = {d_paired:.4f}")

if np.mean(paired_diffs) > 0 and p_paired < 0.05:
    print("  ✓ Taste shifts MORE toward categories that were shown (but not clicked)")
    print("    than toward categories that were neither shown nor clicked.")
    print("    This is evidence of a pure exposure effect.")

# Binary outcome: did taste increase?
treat_increased = np.mean(treatment_deltas > 0)
ctrl_increased = np.mean(control_deltas > 0)
print(f"\nBinary outcome (fraction with Δ > 0):")
print(f"  Treatment: {treat_increased:.4f}  ({treat_increased*100:.1f}%)")
print(f"  Control:   {ctrl_increased:.4f}  ({ctrl_increased*100:.1f}%)")

# Baseline balance check
print("\nBaseline balance check (mean P1 fraction):")
treat_p1 = []
ctrl_p1 = []
for uid in eligible_uids:
    v = user_vecs[uid]
    for ci in range(n_cats):
        was_shown = v['p2_shown_raw'][ci] > 0
        was_clicked = v['p2_click_raw'][ci] > 0
        if was_clicked:
            continue
        if was_shown:
            treat_p1.append(v['p1_vec'][ci])
        else:
            ctrl_p1.append(v['p1_vec'][ci])

print(f"  Treatment mean P1 frac: {np.mean(treat_p1):.6f}")
print(f"  Control mean P1 frac:   {np.mean(ctrl_p1):.6f}")

# Permutation test for within-user paired difference
print(f"\nPermutation test (5000 shuffles) for within-user paired test...")
np.random.seed(42)
n_perm_c = 5000
obs_paired_mean = np.mean(paired_diffs)

# For each user, we can randomly flip the sign of the paired difference
# (equivalent to randomly swapping treatment/control labels within user)
null_means_c = []
for _ in range(n_perm_c):
    signs = np.random.choice([-1, 1], size=n_paired)
    null_means_c.append(np.mean(paired_diffs * signs))

null_mu_c = np.mean(null_means_c)
null_sd_c = np.std(null_means_c)
z_c = (obs_paired_mean - null_mu_c) / null_sd_c if null_sd_c > 0 else 0
count_c = sum(1 for n in null_means_c if n >= obs_paired_mean)
perm_p_c = (count_c + 1) / (n_perm_c + 1)

print(f"  Observed mean paired diff: {obs_paired_mean:+.6f}")
print(f"  Null mean: {null_mu_c:+.6f}")
print(f"  Z-score: {z_c:.2f}")
print(f"  Permutation p-value: {perm_p_c:.4f}")


# ============================================================
# ANALYSIS D: DOSE-RESPONSE WITHIN SHOWN-NOT-CLICKED
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS D: Dose-response — more exposure → larger taste shift?")
print("=" * 70)

dose_shown_frac = []
dose_delta = []
dose_cluster = []

for uid in eligible_uids:
    v = user_vecs[uid]
    for ci in range(n_cats):
        if v['p2_shown_raw'][ci] > 0 and v['p2_click_raw'][ci] == 0:
            dose_shown_frac.append(v['p2_shown_vec'][ci])
            dose_delta.append(v['p3_vec'][ci] - v['p1_vec'][ci])
            dose_cluster.append(uid)

dose_shown_frac = np.array(dose_shown_frac)
dose_delta = np.array(dose_delta)

rho_dose, p_dose = stats.spearmanr(dose_shown_frac, dose_delta)
print(f"\nSpearman correlation (shown fraction vs taste shift): ρ = {rho_dose:.4f}, p = {p_dose:.2e}")

# Binned analysis (terciles of shown fraction)
try:
    dose_bins = pd.qcut(dose_shown_frac, 3, labels=False, duplicates='drop')
    n_bins_dose = len(np.unique(dose_bins))
    print(f"\nBinned by shown fraction ({n_bins_dose} bins):")
    print(f"  {'Bin':<6s} {'Mean shown frac':>16s} {'Mean Δ':>12s} {'N':>10s}")
    print("  " + "-" * 48)
    for b in sorted(np.unique(dose_bins)):
        mask = dose_bins == b
        print(f"  {b:<6d} {np.mean(dose_shown_frac[mask]):16.4f} "
              f"{np.mean(dose_delta[mask]):+12.6f} {mask.sum():10,}")
except Exception as e:
    print(f"  Could not create dose bins: {e}")

# Regression: delta = a + b * shown_frac (with clustered SEs)
X_dose = np.column_stack([np.ones(len(dose_shown_frac)), dose_shown_frac])
cl_dose = np.array([uid_idx.get(uid, 0) for uid in dose_cluster if uid in uid_idx])
if len(cl_dose) == len(dose_delta):
    res_dose = ols_clustered(dose_delta, X_dose, cl_dose)
    print(f"\nOLS: Δ = {res_dose['beta'][0]:.6f} + {res_dose['beta'][1]:.6f} × shown_frac")
    print(f"  β(shown_frac) SE(clust) = {res_dose['se_clust'][1]:.6f}, "
          f"t = {res_dose['t_clust'][1]:.2f}, p = {res_dose['p_clust'][1]:.2e}")


# ============================================================
# ANALYSIS E: PER-CATEGORY BREAKDOWN
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS E: Per-category exposure effects")
print("=" * 70)

cat_results = []
for ci in range(n_cats):
    treat_d = []
    ctrl_d = []
    for uid in eligible_uids:
        v = user_vecs[uid]
        was_clicked = v['p2_click_raw'][ci] > 0
        if was_clicked:
            continue
        delta = v['p3_vec'][ci] - v['p1_vec'][ci]
        if v['p2_shown_raw'][ci] > 0:
            treat_d.append(delta)
        else:
            ctrl_d.append(delta)

    cname = categories[ci]
    if len(treat_d) >= 50 and len(ctrl_d) >= 50:
        treat_d = np.array(treat_d)
        ctrl_d = np.array(ctrl_d)
        t_val, p_val = stats.ttest_ind(treat_d, ctrl_d)
        effect = np.mean(treat_d) - np.mean(ctrl_d)
        cat_results.append({
            'category': cname, 'ci': ci,
            'treat_mean': np.mean(treat_d), 'ctrl_mean': np.mean(ctrl_d),
            'effect': effect, 't': t_val, 'p': p_val,
            'n_treat': len(treat_d), 'n_ctrl': len(ctrl_d)
        })

# BH correction
raw_pvals = [r['p'] for r in cat_results]
adj_pvals = benjamini_hochberg(np.array(raw_pvals))
for i, r in enumerate(cat_results):
    r['p_adj'] = adj_pvals[i]

# Sort by effect size
cat_results.sort(key=lambda x: x['effect'], reverse=True)

print(f"\n{'Category':<18s} {'Treat Δ':>10s} {'Ctrl Δ':>10s} {'Effect':>10s} {'t':>8s} {'p':>10s} {'p(BH)':>10s} {'N_t':>8s} {'N_c':>8s}")
print("-" * 100)
for r in cat_results:
    sig = "*" if r['p_adj'] < 0.05 else " "
    print(f"{r['category']:<18s} {r['treat_mean']:+10.6f} {r['ctrl_mean']:+10.6f} "
          f"{r['effect']:+10.6f} {r['t']:8.2f} {r['p']:10.2e} {r['p_adj']:10.2e}{sig} "
          f"{r['n_treat']:8,} {r['n_ctrl']:8,}")

n_sig = sum(1 for r in cat_results if r['p_adj'] < 0.05)
print(f"\nSignificant after BH correction: {n_sig}/{len(cat_results)} categories")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Does Recommendation Exposure Shift Taste?")
print("=" * 70)

print(f"""
Analysis A (Stacked OLS):
  β(P2 shown) = {beta3:.4f}, p = {p3_val:.2e}
  After controlling for baseline taste (P1) and actual consumption (P2 clicks),
  the category composition of shown articles {"predicts" if p3_val < 0.05 else "does not predict"} future taste.

Analysis B (Cosine alignment):
  cos(shift, P2 shown) = {mean_shown:+.4f}  (t = {t_shown:.1f}, p = {p_shown:.2e})
  cos(shift, P2 clicks) = {mean_click:+.4f}  (t = {t_click:.1f}, p = {p_click:.2e})
  Permutation z-score for shown alignment: {z_b:.1f}

Analysis C (Shown-but-not-clicked — primary test):
  Within-user paired Δ(treatment - control) = {np.mean(paired_diffs):+.6f}
  t = {t_paired:.2f}, p = {p_paired:.2e}, Cohen's d = {d_paired:.4f}
  Permutation z = {z_c:.2f}, perm p = {perm_p_c:.4f}
  Binary: {treat_increased:.1%} of treatment categories shifted up vs {ctrl_increased:.1%} of control

Analysis D (Dose-response):
  Spearman ρ(shown fraction, taste shift) = {rho_dose:.4f}, p = {p_dose:.2e}

Analysis E (Per-category):
  {n_sig}/{len(cat_results)} categories significant after BH correction
""")


# ============================================================
# FIGURES
# ============================================================
print("Generating figures...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Exposure Effect Analysis: Does Recommendation Shift Taste?", fontsize=14, fontweight='bold')

# Panel A: OLS coefficients
ax = axes[0, 0]
coefs = res_a['beta'][1:]
ses = res_a['se_clust'][1:]
labels_a = ['β₁: P1 taste', 'β₂: P2 clicks', 'β₃: P2 shown']
colors_a = ['#2196F3', '#FF9800', '#E91E63']
x_pos = np.arange(len(coefs))
bars = ax.bar(x_pos, coefs, yerr=1.96 * ses, capsize=5, color=colors_a, alpha=0.8, edgecolor='black')
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_a, fontsize=9)
ax.set_ylabel('Coefficient')
ax.set_title('(A) OLS Regression Coefficients\n(clustered 95% CI)')

# Panel B: Cosine alignment distributions
ax = axes[0, 1]
ax.hist(shown_aligns, bins=80, alpha=0.5, color='#E91E63', label=f'cos(shift, shown) μ={mean_shown:.3f}', density=True)
ax.hist(click_aligns, bins=80, alpha=0.5, color='#FF9800', label=f'cos(shift, clicks) μ={mean_click:.3f}', density=True)
ax.axvline(mean_shown, color='#E91E63', linestyle='--', linewidth=2)
ax.axvline(mean_click, color='#FF9800', linestyle='--', linewidth=2)
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
ax.legend(fontsize=8)
ax.set_xlabel('Cosine similarity')
ax.set_ylabel('Density')
ax.set_title('(B) Taste Shift Alignment\nwith Shown vs Clicked')

# Panel C: Treatment vs Control
ax = axes[0, 2]
means_c = [mean_t, mean_c]
sems_c = [np.std(treatment_deltas) / np.sqrt(len(treatment_deltas)),
          np.std(control_deltas) / np.sqrt(len(control_deltas))]
bars_c = ax.bar(['Shown\n(not clicked)', 'Not shown\n(not clicked)'], means_c,
                yerr=[1.96 * s for s in sems_c], capsize=5,
                color=['#E91E63', '#9E9E9E'], alpha=0.8, edgecolor='black')
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_ylabel('Mean taste shift (P3 - P1)')
ax.set_title(f'(C) Shown-Not-Clicked vs Control\npaired t={t_paired:.1f}, p={p_paired:.1e}')

# Panel D: Dose-response
ax = axes[1, 0]
try:
    dose_bins_plot = pd.qcut(dose_shown_frac, 5, labels=False, duplicates='drop')
    bin_means_x, bin_means_y, bin_sems = [], [], []
    for b in sorted(np.unique(dose_bins_plot)):
        mask = dose_bins_plot == b
        bin_means_x.append(np.mean(dose_shown_frac[mask]))
        bin_means_y.append(np.mean(dose_delta[mask]))
        bin_sems.append(np.std(dose_delta[mask]) / np.sqrt(mask.sum()))
    ax.errorbar(bin_means_x, bin_means_y, yerr=[1.96 * s for s in bin_sems],
                fmt='o-', color='#E91E63', capsize=4, markersize=6)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
except Exception:
    ax.text(0.5, 0.5, 'Insufficient variation', ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel('P2 shown fraction (category)')
ax.set_ylabel('Taste shift (P3 - P1)')
ax.set_title(f'(D) Dose-Response\nρ = {rho_dose:.4f}, p = {p_dose:.2e}')

# Panel E: Per-category effects
ax = axes[1, 1]
cat_names_e = [r['category'] for r in cat_results]
effects_e = [r['effect'] for r in cat_results]
colors_e = ['#E91E63' if r['p_adj'] < 0.05 else '#BDBDBD' for r in cat_results]
y_pos_e = np.arange(len(cat_names_e))
ax.barh(y_pos_e, effects_e, color=colors_e, edgecolor='black', alpha=0.8)
ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_yticks(y_pos_e)
ax.set_yticklabels(cat_names_e, fontsize=8)
ax.set_xlabel('Treatment - Control effect')
ax.set_title('(E) Per-Category Exposure Effect\n(pink = sig. after BH)')
ax.invert_yaxis()

# Panel F: Permutation null distribution
ax = axes[1, 2]
ax.hist(null_means_c, bins=50, alpha=0.7, color='#9E9E9E', edgecolor='black', label='Null distribution')
ax.axvline(obs_paired_mean, color='#E91E63', linewidth=2, linestyle='--', label=f'Observed = {obs_paired_mean:+.5f}')
ax.axvline(null_mu_c, color='black', linewidth=1, linestyle=':', label=f'Null mean = {null_mu_c:+.5f}')
ax.legend(fontsize=8)
ax.set_xlabel('Mean paired difference')
ax.set_ylabel('Frequency')
ax.set_title(f'(F) Permutation Test (Analysis C)\nz = {z_c:.1f}, p = {perm_p_c:.4f}')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "exposure_effect.png"), dpi=150, bbox_inches='tight')
fig.savefig(os.path.join(OUT, "exposure_effect.pdf"), bbox_inches='tight')
print(f"Saved: {OUT}/exposure_effect.png")
print(f"Saved: {OUT}/exposure_effect.pdf")

print("\nDone.")
