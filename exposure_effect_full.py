"""
Exposure Effect Analysis: Does Recommendation Exposure Shift Taste?
===================================================================
Comprehensive analysis on MINDlarge 14-day dataset.

Three-period design:
  P1: Nov 9-11   (baseline taste from clicks)
  P2: Nov 12-15  (treatment period: both shown and clicked observed)
  P3: Nov 16-22  (outcome taste, inferred from test history growth)

Analyses:
  A. Stacked OLS: p3_frac = β0 + β1*p1 + β2*p2_click + β3*p2_shown + ε
  B. Cosine alignment of taste shift with shown vs clicked
  C. Shown-but-NOT-clicked vs Control (pure exposure)
  D. Dose-response within shown-not-clicked
  E. Per-category breakdown
  F. DECOMPOSITION: clicked exposure vs non-clicked exposure
     (shown+clicked → taste shift vs shown+NOT-clicked → taste shift)
  G. Backfire test: does shown-not-clicked push taste AWAY?

Author: Hossein Piri / Claude
Date: 2026-04-01
"""

import os, json
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
print = functools.partial(print, flush=True)

BASE = "/Users/piri/Desktop/Recommendation Systems/Mind-Data-Large"
OUT = "/Users/piri/Desktop/Recommendation Systems/code/output_large"
os.makedirs(OUT, exist_ok=True)

RESULTS_FILE = os.path.join(OUT, "exposure_results.json")

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

def ols_clustered(y, X, cluster_ids):
    """OLS with clustered standard errors."""
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    XtX_inv = np.linalg.inv(X.T @ X)
    se_iid = np.sqrt(ss_res / (n - k) * np.diag(XtX_inv))

    sort_idx = np.argsort(cluster_ids)
    X_sorted = X[sort_idx]
    e_sorted = resid[sort_idx]
    cl_sorted = cluster_ids[sort_idx]

    boundaries = np.where(np.diff(cl_sorted) != 0)[0] + 1
    n_clusters = len(boundaries) + 1

    X_groups = np.split(X_sorted, boundaries)
    e_groups = np.split(e_sorted, boundaries)

    B = np.zeros((k, k))
    for X_g, e_g in zip(X_groups, e_groups):
        score = X_g.T @ e_g
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
# 1. LOAD AND PARSE BEHAVIORS
# ============================================================
beh_cols = ['impression_id', 'user_id', 'timestamp', 'click_history', 'impressions']

print("\nLoading train behaviors...", flush=True)
beh_train = pd.read_csv(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
                         sep='\t', header=None, names=beh_cols)
print("Loading dev behaviors...", flush=True)
beh_dev = pd.read_csv(os.path.join(BASE, "MINDlarge_dev", "behaviors.tsv"),
                       sep='\t', header=None, names=beh_cols)

beh_labeled = pd.concat([beh_train, beh_dev], ignore_index=True)
del beh_train, beh_dev
beh_labeled['time'] = pd.to_datetime(beh_labeled['timestamp'], format='%m/%d/%Y %I:%M:%S %p')
beh_labeled['day'] = beh_labeled['time'].dt.day
print(f"  Labeled impressions: {len(beh_labeled):,}", flush=True)

def day_to_period(d):
    if 9 <= d <= 11: return 'P1'
    if 12 <= d <= 15: return 'P2'
    return None

beh_labeled['period'] = beh_labeled['day'].map(day_to_period)

print("  Parsing impressions...", flush=True)
parsed = beh_labeled['impressions'].apply(parse_imp_fast)
beh_labeled['clicked_list'] = parsed.apply(lambda x: x[0])
beh_labeled['shown_list'] = parsed.apply(lambda x: x[1])
del parsed

print("  Parsing histories...", flush=True)
beh_labeled['hist_list'] = beh_labeled['click_history'].apply(parse_history)
beh_labeled['n_history'] = beh_labeled['hist_list'].apply(len)

# Per-user aggregates
print("\nBuilding per-user aggregates...", flush=True)
idx_max_hist = beh_labeled.groupby('user_id')['n_history'].idxmax()
user_hist_df = beh_labeled.loc[idx_max_hist, ['user_id', 'hist_list']].set_index('user_id')
user_hist = user_hist_df['hist_list'].to_dict()
del user_hist_df, idx_max_hist
print(f"  Users with history: {len(user_hist):,}", flush=True)

beh_p1p2 = beh_labeled[beh_labeled['period'].isin(['P1', 'P2'])].copy()
print(f"  P1+P2 impressions: {len(beh_p1p2):,}", flush=True)

print("  Aggregating P1 clicks, P2 clicks, P2 shown...", flush=True)
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
print(f"  Users with P1/P2 data: {len(user_data):,}", flush=True)

# Load test set for P3 inference
print("\nLoading test behaviors...", flush=True)
beh_test = pd.read_csv(os.path.join(BASE, "MINDlarge_test", "behaviors.tsv"),
                        sep='\t', header=None, names=beh_cols)
print(f"  Test impressions: {len(beh_test):,}", flush=True)

print("  Parsing test histories...", flush=True)
beh_test['hist_list'] = beh_test['click_history'].apply(parse_history)
beh_test['n_history'] = beh_test['hist_list'].apply(len)

idx_max_test = beh_test.groupby('user_id')['n_history'].idxmax()
user_test_hist = beh_test.loc[idx_max_test, ['user_id', 'hist_list']].set_index('user_id')['hist_list'].to_dict()
del beh_test, idx_max_test
print(f"  Users in test: {len(user_test_hist):,}", flush=True)

# P3 new clicks
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
# 2. COMPUTE CATEGORY VECTORS
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

    if n_p1 < 2 or n_p2_click < 2 or n_p2_shown < 5 or n_p3 < 2:
        continue

    n_eligible += 1

    # Also compute shown-and-clicked vs shown-and-NOT-clicked vectors
    p2_clicked_set = set(rec['P2_click'])
    p2_shown_clicked = []
    p2_shown_notclicked = []
    for nid in rec['P2_shown']:
        if nid in p2_clicked_set:
            p2_shown_clicked.append(nid)
        else:
            p2_shown_notclicked.append(nid)

    p2_shown_clicked_raw = nids_to_cat_vec(p2_shown_clicked)
    p2_shown_notclicked_raw = nids_to_cat_vec(p2_shown_notclicked)

    user_vecs[uid] = {
        'p1_vec': normalize(p1_raw),
        'p2_click_vec': normalize(p2_click_raw),
        'p2_shown_vec': normalize(p2_shown_raw),
        'p2_shown_clicked_vec': normalize(p2_shown_clicked_raw),
        'p2_shown_notclicked_vec': normalize(p2_shown_notclicked_raw),
        'p3_vec': normalize(p3_raw),
        'p1_raw': p1_raw,
        'p2_click_raw': p2_click_raw,
        'p2_shown_raw': p2_shown_raw,
        'p2_shown_clicked_raw': p2_shown_clicked_raw,
        'p2_shown_notclicked_raw': p2_shown_notclicked_raw,
        'p3_raw': p3_raw,
    }

eligible_uids = list(user_vecs.keys())
print(f"Eligible users (P1>=2, P2click>=2, P2shown>=5, P3>=2): {n_eligible:,}")

# Dictionary to collect all results for JSON export
results = {'n_eligible': n_eligible, 'n_cats': n_cats, 'categories': categories}


# ============================================================
# ANALYSIS A: STACKED OLS REGRESSION
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS A: Stacked OLS — p3_frac = β0 + β1*p1 + β2*p2_click + β3*p2_shown")
print("=" * 70)

uid_idx = {uid: i for i, uid in enumerate(eligible_uids)}
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

y_flat = p3_mat.ravel()
p1_flat = p1_mat.ravel()
p2c_flat = p2c_mat.ravel()
p2s_flat = p2s_mat.ravel()
cl_flat = np.repeat(np.arange(n_users), n_cats)

mask = (p1_flat != 0) | (p2c_flat != 0) | (p2s_flat != 0) | (y_flat != 0)
y_a = y_flat[mask]
X_a = np.column_stack([np.ones(mask.sum()), p1_flat[mask], p2c_flat[mask], p2s_flat[mask]])
cl_a = cl_flat[mask]

res_a = ols_clustered(y_a, X_a, cl_a)

var_names = ['Intercept', 'P1 taste (β1)', 'P2 clicks (β2)', 'P2 shown (β3)']
print(f"\nN obs = {res_a['n_obs']:,}   N clusters = {res_a['n_clusters']:,}   R² = {res_a['r2']:.4f}")
print(f"\n{'Variable':<20s} {'Coef':>8s} {'SE(clust)':>10s} {'t':>8s} {'p':>12s}")
print("-" * 62)
for i, name in enumerate(var_names):
    print(f"{name:<20s} {res_a['beta'][i]:8.4f} {res_a['se_clust'][i]:10.4f} "
          f"{res_a['t_clust'][i]:8.2f} {res_a['p_clust'][i]:12.2e}")

results['analysis_a'] = {
    'n_obs': int(res_a['n_obs']),
    'n_clusters': int(res_a['n_clusters']),
    'r2': float(res_a['r2']),
    'beta': [float(b) for b in res_a['beta']],
    'se_clust': [float(s) for s in res_a['se_clust']],
    't_clust': [float(t) for t in res_a['t_clust']],
    'p_clust': [float(p) for p in res_a['p_clust']],
    'var_names': var_names
}


# ============================================================
# ANALYSIS A2: DECOMPOSED OLS — separate shown-clicked and shown-not-clicked
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS A2: Decomposed OLS — p3 = β0 + β1*p1 + β2*p2_shown_clicked + β3*p2_shown_notclicked")
print("=" * 70)

p2sc_mat = np.zeros((n_users, n_cats))
p2snc_mat = np.zeros((n_users, n_cats))

for i, uid in enumerate(eligible_uids):
    v = user_vecs[uid]
    p2sc_mat[i] = v['p2_shown_clicked_vec']
    p2snc_mat[i] = v['p2_shown_notclicked_vec']

p2sc_flat = p2sc_mat.ravel()
p2snc_flat = p2snc_mat.ravel()

mask2 = (p1_flat != 0) | (p2sc_flat != 0) | (p2snc_flat != 0) | (y_flat != 0)
y_a2 = y_flat[mask2]
X_a2 = np.column_stack([np.ones(mask2.sum()), p1_flat[mask2], p2sc_flat[mask2], p2snc_flat[mask2]])
cl_a2 = cl_flat[mask2]

res_a2 = ols_clustered(y_a2, X_a2, cl_a2)

var_names_a2 = ['Intercept', 'P1 taste (β1)', 'P2 shown+clicked (β2)', 'P2 shown+NOT clicked (β3)']
print(f"\nN obs = {res_a2['n_obs']:,}   N clusters = {res_a2['n_clusters']:,}   R² = {res_a2['r2']:.4f}")
print(f"\n{'Variable':<30s} {'Coef':>8s} {'SE(clust)':>10s} {'t':>8s} {'p':>12s}")
print("-" * 72)
for i, name in enumerate(var_names_a2):
    print(f"{name:<30s} {res_a2['beta'][i]:8.4f} {res_a2['se_clust'][i]:10.4f} "
          f"{res_a2['t_clust'][i]:8.2f} {res_a2['p_clust'][i]:12.2e}")

beta_sc = res_a2['beta'][2]
beta_snc = res_a2['beta'][3]
print(f"\n→ β(shown+clicked) = {beta_sc:.4f}")
print(f"→ β(shown+NOT clicked) = {beta_snc:.4f}")
print(f"→ Ratio: shown+clicked is {beta_sc/beta_snc:.1f}x the effect of shown+NOT clicked" if beta_snc != 0 else "")

if beta_snc < 0:
    print("  *** BACKFIRE: shown-but-not-clicked has NEGATIVE coefficient ***")
    print("  Exposure without engagement pushes taste AWAY from the shown category.")

results['analysis_a2'] = {
    'n_obs': int(res_a2['n_obs']),
    'n_clusters': int(res_a2['n_clusters']),
    'r2': float(res_a2['r2']),
    'beta': [float(b) for b in res_a2['beta']],
    'se_clust': [float(s) for s in res_a2['se_clust']],
    't_clust': [float(t) for t in res_a2['t_clust']],
    'p_clust': [float(p) for p in res_a2['p_clust']],
    'var_names': var_names_a2
}

del p1_mat, p2c_mat, p2s_mat, p3_mat, p2sc_mat, p2snc_mat
del y_flat, p1_flat, p2c_flat, p2s_flat, p2sc_flat, p2snc_flat, cl_flat


# ============================================================
# ANALYSIS B: COSINE ALIGNMENT OF TASTE SHIFT
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS B: Cosine alignment — shift = p3 - p1")
print("=" * 70)

shown_aligns, click_aligns = [], []
shown_clicked_aligns, shown_notclicked_aligns = [], []

for uid in eligible_uids:
    v = user_vecs[uid]
    shift = v['p3_vec'] - v['p1_vec']
    cs_shown = cosine_sim(shift, v['p2_shown_vec'])
    cs_click = cosine_sim(shift, v['p2_click_vec'])
    cs_sc = cosine_sim(shift, v['p2_shown_clicked_vec'])
    cs_snc = cosine_sim(shift, v['p2_shown_notclicked_vec'])

    if not np.isnan(cs_shown) and not np.isnan(cs_click):
        shown_aligns.append(cs_shown)
        click_aligns.append(cs_click)
    if not np.isnan(cs_sc):
        shown_clicked_aligns.append(cs_sc)
    if not np.isnan(cs_snc):
        shown_notclicked_aligns.append(cs_snc)

shown_aligns = np.array(shown_aligns)
click_aligns = np.array(click_aligns)
shown_clicked_aligns = np.array(shown_clicked_aligns)
shown_notclicked_aligns = np.array(shown_notclicked_aligns)

mean_shown = np.mean(shown_aligns)
mean_click = np.mean(click_aligns)
mean_sc = np.mean(shown_clicked_aligns)
mean_snc = np.mean(shown_notclicked_aligns)

t_shown, p_shown = stats.ttest_1samp(shown_aligns, 0)
t_click, p_click = stats.ttest_1samp(click_aligns, 0)
t_sc, p_sc = stats.ttest_1samp(shown_clicked_aligns, 0)
t_snc, p_snc = stats.ttest_1samp(shown_notclicked_aligns, 0)

d_shown = cohens_d_one_sample(shown_aligns)
d_click = cohens_d_one_sample(click_aligns)
d_sc = cohens_d_one_sample(shown_clicked_aligns)
d_snc = cohens_d_one_sample(shown_notclicked_aligns)

print(f"\n{'Alignment':<45s} {'Mean':>8s} {'t':>10s} {'p':>12s} {'Cohen d':>10s}")
print("-" * 88)
print(f"{'cos(shift, P2 shown [all])':<45s} {mean_shown:8.4f} {t_shown:10.2f} {p_shown:12.2e} {d_shown:10.4f}")
print(f"{'cos(shift, P2 clicks)':<45s} {mean_click:8.4f} {t_click:10.2f} {p_click:12.2e} {d_click:10.4f}")
print(f"{'cos(shift, P2 shown+CLICKED)':<45s} {mean_sc:8.4f} {t_sc:10.2f} {p_sc:12.2e} {d_sc:10.4f}")
print(f"{'cos(shift, P2 shown+NOT clicked)':<45s} {mean_snc:8.4f} {t_snc:10.2f} {p_snc:12.2e} {d_snc:10.4f}")

# Paired tests
t_sc_vs_snc, p_sc_vs_snc = stats.ttest_rel(
    shown_clicked_aligns[:min(len(shown_clicked_aligns), len(shown_notclicked_aligns))],
    shown_notclicked_aligns[:min(len(shown_clicked_aligns), len(shown_notclicked_aligns))]
)
print(f"\nPaired test (shown+clicked vs shown+NOT clicked): t = {t_sc_vs_snc:.2f}, p = {p_sc_vs_snc:.2e}")

results['analysis_b'] = {
    'n_users': int(len(shown_aligns)),
    'mean_shown': float(mean_shown), 't_shown': float(t_shown), 'p_shown': float(p_shown), 'd_shown': float(d_shown),
    'mean_click': float(mean_click), 't_click': float(t_click), 'p_click': float(p_click), 'd_click': float(d_click),
    'mean_sc': float(mean_sc), 't_sc': float(t_sc), 'p_sc': float(p_sc), 'd_sc': float(d_sc),
    'mean_snc': float(mean_snc), 't_snc': float(t_snc), 'p_snc': float(p_snc), 'd_snc': float(d_snc),
    'n_sc': int(len(shown_clicked_aligns)),
    'n_snc': int(len(shown_notclicked_aligns)),
    't_sc_vs_snc': float(t_sc_vs_snc), 'p_sc_vs_snc': float(p_sc_vs_snc),
}

# Permutation test for shown alignment
print("\nPermutation test (5000 shuffles) for shown alignment...")
obs_mean_shown = mean_shown
n_perm = 5000

shifts_b = []
shown_vecs_b = []
for uid in eligible_uids:
    v = user_vecs[uid]
    shift = v['p3_vec'] - v['p1_vec']
    cs_check = cosine_sim(shift, v['p2_shown_vec'])
    if not np.isnan(cs_check):
        shifts_b.append(shift)
        shown_vecs_b.append(v['p2_shown_vec'])

S = np.array(shifts_b)
V = np.array(shown_vecs_b)
S_norm = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

np.random.seed(42)
null_means_b = []
for _ in range(n_perm):
    perm = np.random.permutation(len(S_norm))
    sims = np.sum(S_norm * V_norm[perm], axis=1)
    null_means_b.append(np.nanmean(sims))

null_mu_b = np.mean(null_means_b)
null_sd_b = np.std(null_means_b)
z_b = (obs_mean_shown - null_mu_b) / null_sd_b if null_sd_b > 0 else 0
count_b = sum(1 for n in null_means_b if n >= obs_mean_shown)
perm_p_b = (count_b + 1) / (n_perm + 1)

print(f"  Observed mean cos(shift, shown): {obs_mean_shown:.4f}")
print(f"  Null mean: {null_mu_b:.4f}, Null SD: {null_sd_b:.6f}")
print(f"  Z-score: {z_b:.2f}")
print(f"  Permutation p-value: {perm_p_b:.4f}")

results['analysis_b']['perm_z'] = float(z_b)
results['analysis_b']['perm_p'] = float(perm_p_b)


# ============================================================
# ANALYSIS C: SHOWN-BUT-NOT-CLICKED vs CONTROL
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS C: Shown-but-NOT-clicked vs Control categories")
print("=" * 70)

treatment_deltas = []
control_deltas = []
user_paired = defaultdict(lambda: {'treatment': [], 'control': []})

for uid in eligible_uids:
    v = user_vecs[uid]
    for ci in range(n_cats):
        was_shown = v['p2_shown_raw'][ci] > 0
        was_clicked = v['p2_click_raw'][ci] > 0
        delta = v['p3_vec'][ci] - v['p1_vec'][ci]

        if was_clicked:
            continue

        if was_shown:
            treatment_deltas.append(delta)
            user_paired[uid]['treatment'].append(delta)
        else:
            control_deltas.append(delta)
            user_paired[uid]['control'].append(delta)

treatment_deltas = np.array(treatment_deltas)
control_deltas = np.array(control_deltas)

print(f"\nTotal (user, category) pairs:")
print(f"  Treatment (shown, not clicked): {len(treatment_deltas):,}")
print(f"  Control (not shown, not clicked): {len(control_deltas):,}")

t_pool, p_pool = stats.ttest_ind(treatment_deltas, control_deltas)
mean_t = np.mean(treatment_deltas)
mean_c = np.mean(control_deltas)
diff_pool = mean_t - mean_c

print(f"\nPooled comparison:")
print(f"  Treatment mean Δ: {mean_t:+.6f}")
print(f"  Control mean Δ:   {mean_c:+.6f}")
print(f"  Difference:       {diff_pool:+.6f}")
print(f"  t = {t_pool:.2f}, p = {p_pool:.2e}")

# Within-user paired comparison
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
print(f"  Users with both treatment & control: {n_paired:,}")
print(f"  Mean within-user (treatment - control) Δ: {np.mean(paired_diffs):+.6f}")
print(f"  t = {t_paired:.2f}, p = {p_paired:.2e}, Cohen's d = {d_paired:.4f}")

treat_increased = np.mean(treatment_deltas > 0)
ctrl_increased = np.mean(control_deltas > 0)
print(f"\nBinary outcome (fraction with Δ > 0):")
print(f"  Treatment: {treat_increased:.4f}  ({treat_increased*100:.1f}%)")
print(f"  Control:   {ctrl_increased:.4f}  ({ctrl_increased*100:.1f}%)")

# Permutation test
print(f"\nPermutation test (5000 shuffles) for within-user paired test...")
np.random.seed(42)
n_perm_c = 5000
obs_paired_mean = np.mean(paired_diffs)
null_means_c = []
for _ in range(n_perm_c):
    signs = np.random.choice([-1, 1], size=n_paired)
    null_means_c.append(np.mean(paired_diffs * signs))

null_mu_c = np.mean(null_means_c)
null_sd_c = np.std(null_means_c)
z_c = (obs_paired_mean - null_mu_c) / null_sd_c if null_sd_c > 0 else 0
count_c = sum(1 for n in null_means_c if n >= obs_paired_mean)
perm_p_c = (count_c + 1) / (n_perm_c + 1)

print(f"  Z-score: {z_c:.2f}, Permutation p = {perm_p_c:.4f}")

results['analysis_c'] = {
    'n_treatment': int(len(treatment_deltas)),
    'n_control': int(len(control_deltas)),
    'mean_treatment': float(mean_t),
    'mean_control': float(mean_c),
    'diff_pool': float(diff_pool),
    't_pool': float(t_pool), 'p_pool': float(p_pool),
    'n_paired': int(n_paired),
    'mean_paired_diff': float(np.mean(paired_diffs)),
    't_paired': float(t_paired), 'p_paired': float(p_paired),
    'd_paired': float(d_paired),
    'treat_increased_frac': float(treat_increased),
    'ctrl_increased_frac': float(ctrl_increased),
    'perm_z': float(z_c), 'perm_p': float(perm_p_c),
}


# ============================================================
# ANALYSIS F: THREE-WAY DECOMPOSITION (clicked / shown-not-clicked / neither)
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS F: Three-way decomposition — clicked vs shown-not-clicked vs neither")
print("=" * 70)

deltas_clicked = []
deltas_shown_notclicked = []
deltas_neither = []
user_three_way = defaultdict(lambda: {'clicked': [], 'shown_nc': [], 'neither': []})

for uid in eligible_uids:
    v = user_vecs[uid]
    for ci in range(n_cats):
        was_shown = v['p2_shown_raw'][ci] > 0
        was_clicked = v['p2_click_raw'][ci] > 0
        delta = v['p3_vec'][ci] - v['p1_vec'][ci]

        if was_clicked:
            deltas_clicked.append(delta)
            user_three_way[uid]['clicked'].append(delta)
        elif was_shown:
            deltas_shown_notclicked.append(delta)
            user_three_way[uid]['shown_nc'].append(delta)
        else:
            deltas_neither.append(delta)
            user_three_way[uid]['neither'].append(delta)

deltas_clicked = np.array(deltas_clicked)
deltas_shown_notclicked = np.array(deltas_shown_notclicked)
deltas_neither = np.array(deltas_neither)

mean_clicked = np.mean(deltas_clicked)
mean_shown_nc = np.mean(deltas_shown_notclicked)
mean_neither = np.mean(deltas_neither)

t_cl, p_cl = stats.ttest_1samp(deltas_clicked, 0)
t_snc2, p_snc2 = stats.ttest_1samp(deltas_shown_notclicked, 0)
t_nei, p_nei = stats.ttest_1samp(deltas_neither, 0)

print(f"\nPooled taste shift (P3 - P1) by exposure type:")
print(f"  {'Group':<30s} {'N':>10s} {'Mean Δ':>12s} {'t':>10s} {'p':>12s}")
print("  " + "-" * 78)
print(f"  {'Clicked (shown+clicked)':<30s} {len(deltas_clicked):10,} {mean_clicked:+12.6f} {t_cl:10.2f} {p_cl:12.2e}")
print(f"  {'Shown but NOT clicked':<30s} {len(deltas_shown_notclicked):10,} {mean_shown_nc:+12.6f} {t_snc2:10.2f} {p_snc2:12.2e}")
print(f"  {'Neither shown nor clicked':<30s} {len(deltas_neither):10,} {mean_neither:+12.6f} {t_nei:10.2f} {p_nei:12.2e}")

# Pairwise comparisons
t_cl_vs_snc, p_cl_vs_snc = stats.ttest_ind(deltas_clicked, deltas_shown_notclicked)
t_snc_vs_nei, p_snc_vs_nei = stats.ttest_ind(deltas_shown_notclicked, deltas_neither)
t_cl_vs_nei, p_cl_vs_nei = stats.ttest_ind(deltas_clicked, deltas_neither)

print(f"\nPairwise comparisons:")
print(f"  Clicked vs Shown-not-clicked: diff = {mean_clicked - mean_shown_nc:+.6f}, t = {t_cl_vs_snc:.2f}, p = {p_cl_vs_snc:.2e}")
print(f"  Shown-not-clicked vs Neither: diff = {mean_shown_nc - mean_neither:+.6f}, t = {t_snc_vs_nei:.2f}, p = {p_snc_vs_nei:.2e}")
print(f"  Clicked vs Neither:           diff = {mean_clicked - mean_neither:+.6f}, t = {t_cl_vs_nei:.2f}, p = {p_cl_vs_nei:.2e}")

# Within-user three-way (paired)
user_cl_nc_diffs = []
user_cl_nei_diffs = []
user_nc_nei_diffs = []

for uid in user_three_way:
    d = user_three_way[uid]
    if len(d['clicked']) > 0 and len(d['shown_nc']) > 0:
        user_cl_nc_diffs.append(np.mean(d['clicked']) - np.mean(d['shown_nc']))
    if len(d['clicked']) > 0 and len(d['neither']) > 0:
        user_cl_nei_diffs.append(np.mean(d['clicked']) - np.mean(d['neither']))
    if len(d['shown_nc']) > 0 and len(d['neither']) > 0:
        user_nc_nei_diffs.append(np.mean(d['shown_nc']) - np.mean(d['neither']))

user_cl_nc_diffs = np.array(user_cl_nc_diffs)
user_cl_nei_diffs = np.array(user_cl_nei_diffs)
user_nc_nei_diffs = np.array(user_nc_nei_diffs)

if len(user_cl_nc_diffs) > 0:
    t_wc, p_wc = stats.ttest_1samp(user_cl_nc_diffs, 0)
    print(f"\nWithin-user paired (clicked - shown_nc): mean = {np.mean(user_cl_nc_diffs):+.6f}, t = {t_wc:.2f}, p = {p_wc:.2e}, N = {len(user_cl_nc_diffs):,}")
if len(user_cl_nei_diffs) > 0:
    t_wn, p_wn = stats.ttest_1samp(user_cl_nei_diffs, 0)
    print(f"Within-user paired (clicked - neither):   mean = {np.mean(user_cl_nei_diffs):+.6f}, t = {t_wn:.2f}, p = {p_wn:.2e}, N = {len(user_cl_nei_diffs):,}")
if len(user_nc_nei_diffs) > 0:
    t_wnn, p_wnn = stats.ttest_1samp(user_nc_nei_diffs, 0)
    print(f"Within-user paired (shown_nc - neither):  mean = {np.mean(user_nc_nei_diffs):+.6f}, t = {t_wnn:.2f}, p = {p_wnn:.2e}, N = {len(user_nc_nei_diffs):,}")

results['analysis_f'] = {
    'n_clicked': int(len(deltas_clicked)),
    'n_shown_nc': int(len(deltas_shown_notclicked)),
    'n_neither': int(len(deltas_neither)),
    'mean_clicked': float(mean_clicked),
    'mean_shown_nc': float(mean_shown_nc),
    'mean_neither': float(mean_neither),
    't_clicked': float(t_cl), 'p_clicked': float(p_cl),
    't_shown_nc': float(t_snc2), 'p_shown_nc': float(p_snc2),
    't_neither': float(t_nei), 'p_neither': float(p_nei),
    'diff_cl_vs_snc': float(mean_clicked - mean_shown_nc),
    't_cl_vs_snc': float(t_cl_vs_snc), 'p_cl_vs_snc': float(p_cl_vs_snc),
    'diff_snc_vs_nei': float(mean_shown_nc - mean_neither),
    't_snc_vs_nei': float(t_snc_vs_nei), 'p_snc_vs_nei': float(p_snc_vs_nei),
    'diff_cl_vs_nei': float(mean_clicked - mean_neither),
    't_cl_vs_nei': float(t_cl_vs_nei), 'p_cl_vs_nei': float(p_cl_vs_nei),
}
if len(user_cl_nc_diffs) > 0:
    results['analysis_f']['within_user_cl_nc_mean'] = float(np.mean(user_cl_nc_diffs))
    results['analysis_f']['within_user_cl_nc_t'] = float(t_wc)
    results['analysis_f']['within_user_cl_nc_p'] = float(p_wc)
    results['analysis_f']['within_user_cl_nc_n'] = int(len(user_cl_nc_diffs))
if len(user_nc_nei_diffs) > 0:
    results['analysis_f']['within_user_nc_nei_mean'] = float(np.mean(user_nc_nei_diffs))
    results['analysis_f']['within_user_nc_nei_t'] = float(t_wnn)
    results['analysis_f']['within_user_nc_nei_p'] = float(p_wnn)
    results['analysis_f']['within_user_nc_nei_n'] = int(len(user_nc_nei_diffs))


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

# Binned analysis
try:
    dose_bins = pd.qcut(dose_shown_frac, 5, labels=False, duplicates='drop')
    n_bins_dose = len(np.unique(dose_bins))
    print(f"\nBinned by shown fraction ({n_bins_dose} bins):")
    print(f"  {'Bin':<6s} {'Mean shown frac':>16s} {'Mean Δ':>12s} {'N':>10s}")
    print("  " + "-" * 48)
    dose_bin_data = []
    for b in sorted(np.unique(dose_bins)):
        mask = dose_bins == b
        bm_x = np.mean(dose_shown_frac[mask])
        bm_y = np.mean(dose_delta[mask])
        bn = int(mask.sum())
        print(f"  {b:<6d} {bm_x:16.4f} {bm_y:+12.6f} {bn:10,}")
        dose_bin_data.append({'bin': int(b), 'mean_shown_frac': float(bm_x),
                              'mean_delta': float(bm_y), 'n': bn})
except Exception as e:
    print(f"  Could not create dose bins: {e}")
    dose_bin_data = []

# OLS dose-response
X_dose = np.column_stack([np.ones(len(dose_shown_frac)), dose_shown_frac])
cl_dose = np.array([uid_idx.get(uid, 0) for uid in dose_cluster if uid in uid_idx])
if len(cl_dose) == len(dose_delta):
    res_dose = ols_clustered(dose_delta, X_dose, cl_dose)
    print(f"\nOLS: Δ = {res_dose['beta'][0]:.6f} + {res_dose['beta'][1]:.6f} × shown_frac")
    print(f"  β(shown_frac) SE(clust) = {res_dose['se_clust'][1]:.6f}, "
          f"t = {res_dose['t_clust'][1]:.2f}, p = {res_dose['p_clust'][1]:.2e}")

results['analysis_d'] = {
    'rho': float(rho_dose), 'p_rho': float(p_dose),
    'bins': dose_bin_data,
}
if len(cl_dose) == len(dose_delta):
    results['analysis_d']['ols_intercept'] = float(res_dose['beta'][0])
    results['analysis_d']['ols_beta'] = float(res_dose['beta'][1])
    results['analysis_d']['ols_se'] = float(res_dose['se_clust'][1])
    results['analysis_d']['ols_t'] = float(res_dose['t_clust'][1])
    results['analysis_d']['ols_p'] = float(res_dose['p_clust'][1])


# ============================================================
# ANALYSIS E: PER-CATEGORY BREAKDOWN (three-way)
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS E: Per-category three-way decomposition")
print("=" * 70)

cat_three_way = []
for ci in range(n_cats):
    cl_d, snc_d, nei_d = [], [], []
    for uid in eligible_uids:
        v = user_vecs[uid]
        was_shown = v['p2_shown_raw'][ci] > 0
        was_clicked = v['p2_click_raw'][ci] > 0
        delta = v['p3_vec'][ci] - v['p1_vec'][ci]

        if was_clicked:
            cl_d.append(delta)
        elif was_shown:
            snc_d.append(delta)
        else:
            nei_d.append(delta)

    cname = categories[ci]
    if len(cl_d) >= 30 and len(snc_d) >= 30 and len(nei_d) >= 30:
        cl_d = np.array(cl_d)
        snc_d = np.array(snc_d)
        nei_d = np.array(nei_d)
        cat_three_way.append({
            'category': cname,
            'clicked_mean': float(np.mean(cl_d)),
            'shown_nc_mean': float(np.mean(snc_d)),
            'neither_mean': float(np.mean(nei_d)),
            'n_clicked': len(cl_d),
            'n_shown_nc': len(snc_d),
            'n_neither': len(nei_d),
            'click_effect': float(np.mean(cl_d) - np.mean(nei_d)),
            'exposure_effect': float(np.mean(snc_d) - np.mean(nei_d)),
        })

print(f"\n{'Category':<18s} {'Clicked Δ':>12s} {'Shown-NC Δ':>12s} {'Neither Δ':>12s} {'Click eff':>12s} {'Expos eff':>12s}")
print("-" * 82)
for r in sorted(cat_three_way, key=lambda x: x['click_effect'], reverse=True):
    print(f"{r['category']:<18s} {r['clicked_mean']:+12.6f} {r['shown_nc_mean']:+12.6f} "
          f"{r['neither_mean']:+12.6f} {r['click_effect']:+12.6f} {r['exposure_effect']:+12.6f}")

n_backfire = sum(1 for r in cat_three_way if r['exposure_effect'] < 0)
n_positive = sum(1 for r in cat_three_way if r['exposure_effect'] > 0)
print(f"\nExposure (shown-not-clicked) effect: {n_positive} positive, {n_backfire} negative (out of {len(cat_three_way)} categories)")

results['analysis_e'] = cat_three_way


# ============================================================
# ANALYSIS G: BACKFIRE DEEP DIVE
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS G: Backfire analysis — does exposure without click push taste AWAY?")
print("=" * 70)

# For each user: compute taste shift direction, then check if shown-not-clicked
# categories moved in the OPPOSITE direction
backfire_count = 0
reinforce_count = 0
total_count = 0

per_user_backfire = []

for uid in eligible_uids:
    v = user_vecs[uid]
    for ci in range(n_cats):
        was_shown = v['p2_shown_raw'][ci] > 0
        was_clicked = v['p2_click_raw'][ci] > 0
        if was_shown and not was_clicked:
            delta = v['p3_vec'][ci] - v['p1_vec'][ci]
            total_count += 1
            if delta < 0:
                backfire_count += 1
            elif delta > 0:
                reinforce_count += 1

frac_backfire = backfire_count / total_count if total_count > 0 else 0
frac_reinforce = reinforce_count / total_count if total_count > 0 else 0
frac_zero = 1 - frac_backfire - frac_reinforce

print(f"\nShown-but-not-clicked (user, category) pairs: {total_count:,}")
print(f"  Taste moved TOWARD shown category:  {reinforce_count:,} ({frac_reinforce:.1%})")
print(f"  Taste moved AWAY from shown category: {backfire_count:,} ({frac_backfire:.1%})")
print(f"  No change:                           {total_count - reinforce_count - backfire_count:,} ({frac_zero:.1%})")

# Magnitude comparison
toward_magnitudes = []
away_magnitudes = []

for uid in eligible_uids:
    v = user_vecs[uid]
    for ci in range(n_cats):
        was_shown = v['p2_shown_raw'][ci] > 0
        was_clicked = v['p2_click_raw'][ci] > 0
        if was_shown and not was_clicked:
            delta = v['p3_vec'][ci] - v['p1_vec'][ci]
            if delta > 0:
                toward_magnitudes.append(delta)
            elif delta < 0:
                away_magnitudes.append(abs(delta))

toward_magnitudes = np.array(toward_magnitudes)
away_magnitudes = np.array(away_magnitudes)

print(f"\nMagnitude comparison:")
print(f"  Mean magnitude when taste moves TOWARD: {np.mean(toward_magnitudes):.6f}")
print(f"  Mean magnitude when taste moves AWAY:   {np.mean(away_magnitudes):.6f}")
t_mag, p_mag = stats.ttest_ind(away_magnitudes, toward_magnitudes)
print(f"  t(away > toward) = {t_mag:.2f}, p = {p_mag:.2e}")

results['analysis_g'] = {
    'total_pairs': int(total_count),
    'n_toward': int(reinforce_count),
    'n_away': int(backfire_count),
    'frac_toward': float(frac_reinforce),
    'frac_away': float(frac_backfire),
    'mean_mag_toward': float(np.mean(toward_magnitudes)),
    'mean_mag_away': float(np.mean(away_magnitudes)),
    't_mag': float(t_mag), 'p_mag': float(p_mag),
}


# ============================================================
# SAVE RESULTS TO JSON
# ============================================================
with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("GRAND SUMMARY")
print("=" * 70)

print(f"""
ELIGIBLE USERS: {n_eligible:,}

A. Stacked OLS (p3 ~ p1 + p2_click + p2_shown):
   β(P1 taste) = {res_a['beta'][1]:.4f}
   β(P2 clicks) = {res_a['beta'][2]:.4f}
   β(P2 shown) = {res_a['beta'][3]:.4f}
   R² = {res_a['r2']:.4f}

A2. Decomposed OLS (p3 ~ p1 + p2_shown_clicked + p2_shown_notclicked):
   β(P1 taste) = {res_a2['beta'][1]:.4f}
   β(shown+clicked) = {res_a2['beta'][2]:.4f}
   β(shown+NOT clicked) = {res_a2['beta'][3]:.4f}
   R² = {res_a2['r2']:.4f}

B. Cosine alignment:
   cos(shift, P2 shown) = {mean_shown:+.4f} (t={t_shown:.1f})
   cos(shift, P2 clicks) = {mean_click:+.4f} (t={t_click:.1f})
   cos(shift, shown+clicked) = {mean_sc:+.4f} (t={t_sc:.1f})
   cos(shift, shown+NOT clicked) = {mean_snc:+.4f} (t={t_snc:.1f})

C. Shown-not-clicked vs control:
   Within-user Δ = {np.mean(paired_diffs):+.6f}, t={t_paired:.2f}, d={d_paired:.4f}

D. Dose-response: ρ = {rho_dose:.4f}

F. Three-way decomposition:
   Clicked Δ = {mean_clicked:+.6f}
   Shown-NC Δ = {mean_shown_nc:+.6f}
   Neither Δ = {mean_neither:+.6f}

G. Backfire: {frac_backfire:.1%} of shown-not-clicked pairs move AWAY
""")


# ============================================================
# FIGURES
# ============================================================
print("Generating figures...")
fig, axes = plt.subplots(2, 4, figsize=(24, 11))
fig.suptitle("Exposure Effect Analysis: How Does Recommendation Exposure Change Taste?",
             fontsize=14, fontweight='bold')

# Panel A: OLS coefficients (original)
ax = axes[0, 0]
coefs = res_a['beta'][1:]
ses = res_a['se_clust'][1:]
labels_a = ['β₁: P1 taste', 'β₂: P2 clicks', 'β₃: P2 shown']
colors_a = ['#2196F3', '#FF9800', '#E91E63']
x_pos = np.arange(len(coefs))
ax.bar(x_pos, coefs, yerr=1.96 * ses, capsize=5, color=colors_a, alpha=0.8, edgecolor='black')
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_a, fontsize=9)
ax.set_ylabel('Coefficient')
ax.set_title('(A) OLS: P3 taste ~ P1 + P2 clicks + P2 shown')

# Panel A2: Decomposed OLS
ax = axes[0, 1]
coefs2 = res_a2['beta'][1:]
ses2 = res_a2['se_clust'][1:]
labels_a2 = ['β₁: P1 taste', 'β₂: Shown+Clicked', 'β₃: Shown+NOT Clicked']
colors_a2 = ['#2196F3', '#4CAF50', '#F44336']
x_pos2 = np.arange(len(coefs2))
ax.bar(x_pos2, coefs2, yerr=1.96 * ses2, capsize=5, color=colors_a2, alpha=0.8, edgecolor='black')
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_xticks(x_pos2)
ax.set_xticklabels(labels_a2, fontsize=8)
ax.set_ylabel('Coefficient')
ax.set_title('(A2) Decomposed: Clicked vs Not-Clicked Exposure')

# Panel B: Cosine alignment (4-way)
ax = axes[0, 2]
bar_data = [
    ('Shown\n(all)', mean_shown, '#9C27B0'),
    ('Clicks', mean_click, '#FF9800'),
    ('Shown+\nClicked', mean_sc, '#4CAF50'),
    ('Shown+\nNot Clicked', mean_snc, '#F44336'),
]
x_b = np.arange(len(bar_data))
for i, (lab, val, col) in enumerate(bar_data):
    ax.bar(i, val, color=col, alpha=0.8, edgecolor='black')
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_xticks(x_b)
ax.set_xticklabels([d[0] for d in bar_data], fontsize=8)
ax.set_ylabel('Mean cosine(shift, vector)')
ax.set_title('(B) Taste Shift Alignment by Exposure Type')

# Panel C: Three-way decomposition
ax = axes[0, 3]
means_f = [mean_clicked, mean_shown_nc, mean_neither]
sems_f = [np.std(deltas_clicked) / np.sqrt(len(deltas_clicked)),
          np.std(deltas_shown_notclicked) / np.sqrt(len(deltas_shown_notclicked)),
          np.std(deltas_neither) / np.sqrt(len(deltas_neither))]
bars_f = ax.bar(['Clicked', 'Shown\nnot clicked', 'Neither'],
                means_f, yerr=[1.96 * s for s in sems_f], capsize=5,
                color=['#4CAF50', '#F44336', '#9E9E9E'], alpha=0.8, edgecolor='black')
ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_ylabel('Mean taste shift (P3 - P1)')
ax.set_title('(F) Three-Way Decomposition')

# Panel D: Dose-response
ax = axes[1, 0]
try:
    dose_bins_plot = pd.qcut(dose_shown_frac, 5, labels=False, duplicates='drop')
    bin_means_x, bin_means_y, bin_sems = [], [], []
    for b in sorted(np.unique(dose_bins_plot)):
        mask_d = dose_bins_plot == b
        bin_means_x.append(np.mean(dose_shown_frac[mask_d]))
        bin_means_y.append(np.mean(dose_delta[mask_d]))
        bin_sems.append(np.std(dose_delta[mask_d]) / np.sqrt(mask_d.sum()))
    ax.errorbar(bin_means_x, bin_means_y, yerr=[1.96 * s for s in bin_sems],
                fmt='o-', color='#E91E63', capsize=4, markersize=6)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
except Exception:
    ax.text(0.5, 0.5, 'Insufficient variation', ha='center', va='center', transform=ax.transAxes)
ax.set_xlabel('P2 shown fraction (category)')
ax.set_ylabel('Taste shift (P3 - P1)')
ax.set_title(f'(D) Dose-Response (shown-not-clicked)\nρ = {rho_dose:.4f}')

# Panel E: Per-category three-way
ax = axes[1, 1]
cat_sorted = sorted(cat_three_way, key=lambda x: x['click_effect'], reverse=True)
cat_names_e = [r['category'] for r in cat_sorted]
click_effs = [r['click_effect'] for r in cat_sorted]
expos_effs = [r['exposure_effect'] for r in cat_sorted]
y_pos_e = np.arange(len(cat_names_e))
w = 0.35
ax.barh(y_pos_e - w/2, click_effs, w, color='#4CAF50', alpha=0.8, edgecolor='black', label='Click effect')
ax.barh(y_pos_e + w/2, expos_effs, w, color='#F44336', alpha=0.8, edgecolor='black', label='Exposure effect')
ax.axvline(0, color='black', linestyle='--', linewidth=0.5)
ax.set_yticks(y_pos_e)
ax.set_yticklabels(cat_names_e, fontsize=8)
ax.set_xlabel('Effect size (vs neither)')
ax.set_title('(E) Per-Category: Click vs Exposure Effect')
ax.legend(fontsize=8, loc='lower right')
ax.invert_yaxis()

# Panel G: Backfire breakdown
ax = axes[1, 2]
ax.bar(['Toward\n(reinforced)', 'Away\n(backfire)', 'No change'],
       [frac_reinforce, frac_backfire, frac_zero],
       color=['#4CAF50', '#F44336', '#9E9E9E'], alpha=0.8, edgecolor='black')
ax.set_ylabel('Fraction of (user, category) pairs')
ax.set_title('(G) Shown-Not-Clicked: Direction of Taste Shift')

# Panel G2: Magnitude comparison
ax = axes[1, 3]
ax.hist(toward_magnitudes, bins=80, alpha=0.5, color='#4CAF50', label=f'Toward (μ={np.mean(toward_magnitudes):.4f})', density=True)
ax.hist(away_magnitudes, bins=80, alpha=0.5, color='#F44336', label=f'Away (μ={np.mean(away_magnitudes):.4f})', density=True)
ax.legend(fontsize=8)
ax.set_xlabel('|Taste shift|')
ax.set_ylabel('Density')
ax.set_title('(G2) Magnitude: Toward vs Away Shifts')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "exposure_effect_full.png"), dpi=150, bbox_inches='tight')
fig.savefig(os.path.join(OUT, "exposure_effect_full.pdf"), bbox_inches='tight')
print(f"Saved: {OUT}/exposure_effect_full.png")
print(f"Saved: {OUT}/exposure_effect_full.pdf")

print("\nDone.")
