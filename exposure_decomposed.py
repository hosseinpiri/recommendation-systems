"""
Decomposed Exposure Analysis — for output_full master report
=============================================================
Runs the analyses present in output_large but missing from the
full_empirical_report.tex master report:

  1. Decomposed OLS: shown+clicked vs shown+NOT-clicked as separate regressors
  2. Four-way cosine alignment (shown-all, clicks, shown+clicked, shown+NOT-clicked)
  3. Three-way group decomposition (clicked / shown-NC / neither)
  4. Quintile dose-response (vs tercile in current master)
  5. Full per-category decomposition with click effect and exposure effect
  6. Magnitude asymmetry (direction + magnitude of shown-not-clicked shifts)

Data: MINDlarge 14-day (Mind-Data-Large), three-period design.
Output: output_full/

Author: Hossein Piri / Claude
Date: 2026-04-01
"""

import os, json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import functools
print = functools.partial(print, flush=True)

BASE = "/Users/piri/Desktop/Recommendation Systems/Mind-Data-Large"
OUT = "/Users/piri/Desktop/Recommendation Systems/code/output_full"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("DECOMPOSED EXPOSURE ANALYSIS — for master report")
print("=" * 70)

# ============================================================
# 0. LOAD NEWS
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
print(f"  Total unique: {len(news):,}")

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
    n, k = X.shape
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    XtX_inv = np.linalg.inv(X.T @ X)
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
        'beta': beta, 'se_clust': se_clust,
        't_clust': t_clust, 'p_clust': p_clust,
        'r2': r2, 'n_obs': n, 'n_clusters': n_clusters
    }

def cohens_d_one_sample(x):
    return np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 0.0


# ============================================================
# 1. LOAD BEHAVIORS
# ============================================================
beh_cols = ['impression_id', 'user_id', 'timestamp', 'click_history', 'impressions']

print("\nLoading train behaviors...")
beh_train = pd.read_csv(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
                         sep='\t', header=None, names=beh_cols)
print("Loading dev behaviors...")
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

print("\nBuilding per-user aggregates...")
idx_max_hist = beh_labeled.groupby('user_id')['n_history'].idxmax()
user_hist = beh_labeled.loc[idx_max_hist, ['user_id', 'hist_list']].set_index('user_id')['hist_list'].to_dict()
del idx_max_hist
print(f"  Users with history: {len(user_hist):,}")

beh_p1p2 = beh_labeled[beh_labeled['period'].isin(['P1', 'P2'])].copy()
print(f"  P1+P2 impressions: {len(beh_p1p2):,}")

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

# Test set for P3
print("\nLoading test behaviors...")
beh_test = pd.read_csv(os.path.join(BASE, "MINDlarge_test", "behaviors.tsv"),
                        sep='\t', header=None, names=beh_cols)
print(f"  Test impressions: {len(beh_test):,}")

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
# 2. CATEGORY VECTORS
# ============================================================
print("\nComputing category vectors...")

user_vecs = {}
for uid in user_data:
    rec = user_data[uid]
    p1_raw = nids_to_cat_vec(rec['P1_click'])
    p2_click_raw = nids_to_cat_vec(rec['P2_click'])
    p2_shown_raw = nids_to_cat_vec(rec['P2_shown'])

    if uid not in user_p3_clicks:
        continue
    p3_raw = nids_to_cat_vec(user_p3_clicks[uid])

    if int(p1_raw.sum()) < 2 or int(p2_click_raw.sum()) < 2 or int(p2_shown_raw.sum()) < 5 or int(p3_raw.sum()) < 2:
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
        'p1_raw': p1_raw,
        'p2_click_raw': p2_click_raw,
        'p2_shown_raw': p2_shown_raw,
        'p2_shown_clicked_raw': nids_to_cat_vec(p2_shown_clicked),
        'p2_shown_notclicked_raw': nids_to_cat_vec(p2_shown_notclicked),
        'p3_raw': p3_raw,
    }

eligible_uids = list(user_vecs.keys())
n_users = len(eligible_uids)
uid_idx = {uid: i for i, uid in enumerate(eligible_uids)}
print(f"Eligible users: {n_users:,}")

results = {}

# ============================================================
# ANALYSIS 1: DECOMPOSED OLS
# ============================================================
print("\n" + "=" * 70)
print("1. Decomposed OLS: p3 ~ p1 + shown_clicked + shown_not_clicked")
print("=" * 70)

p1_mat = np.zeros((n_users, n_cats))
p2sc_mat = np.zeros((n_users, n_cats))
p2snc_mat = np.zeros((n_users, n_cats))
p3_mat = np.zeros((n_users, n_cats))

for i, uid in enumerate(eligible_uids):
    v = user_vecs[uid]
    p1_mat[i] = v['p1_vec']
    p2sc_mat[i] = v['p2_shown_clicked_vec']
    p2snc_mat[i] = v['p2_shown_notclicked_vec']
    p3_mat[i] = v['p3_vec']

y_flat = p3_mat.ravel()
p1_flat = p1_mat.ravel()
p2sc_flat = p2sc_mat.ravel()
p2snc_flat = p2snc_mat.ravel()
cl_flat = np.repeat(np.arange(n_users), n_cats)

mask = (p1_flat != 0) | (p2sc_flat != 0) | (p2snc_flat != 0) | (y_flat != 0)
y_m = y_flat[mask]
X_m = np.column_stack([np.ones(mask.sum()), p1_flat[mask], p2sc_flat[mask], p2snc_flat[mask]])
cl_m = cl_flat[mask]

res_dec = ols_clustered(y_m, X_m, cl_m)

var_names = ['Intercept', 'P1 taste', 'Shown+clicked', 'Shown+NOT clicked']
print(f"\nN={res_dec['n_obs']:,}  Clusters={res_dec['n_clusters']:,}  R²={res_dec['r2']:.4f}")
print(f"\n{'Variable':<25s} {'Coef':>8s} {'SE':>10s} {'t':>8s} {'p':>12s}")
print("-" * 67)
for i, name in enumerate(var_names):
    print(f"{name:<25s} {res_dec['beta'][i]:8.4f} {res_dec['se_clust'][i]:10.4f} "
          f"{res_dec['t_clust'][i]:8.2f} {res_dec['p_clust'][i]:12.2e}")

ratio = res_dec['beta'][2] / res_dec['beta'][3] if res_dec['beta'][3] != 0 else float('inf')
print(f"\nShown+clicked is {ratio:.1f}x shown+NOT clicked")

results['decomposed_ols'] = {
    'n_obs': int(res_dec['n_obs']), 'n_clusters': int(res_dec['n_clusters']),
    'r2': float(res_dec['r2']),
    'beta': [float(b) for b in res_dec['beta']],
    'se': [float(s) for s in res_dec['se_clust']],
    't': [float(t) for t in res_dec['t_clust']],
    'p': [float(p) for p in res_dec['p_clust']],
    'ratio': float(ratio),
}

del p1_mat, p2sc_mat, p2snc_mat, p3_mat, y_flat, p1_flat, p2sc_flat, p2snc_flat, cl_flat


# ============================================================
# ANALYSIS 2: FOUR-WAY COSINE ALIGNMENT
# ============================================================
print("\n" + "=" * 70)
print("2. Four-way cosine alignment")
print("=" * 70)

shown_aligns, click_aligns = [], []
sc_aligns, snc_aligns = [], []

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
        sc_aligns.append(cs_sc)
    if not np.isnan(cs_snc):
        snc_aligns.append(cs_snc)

for name, arr in [('shown (all)', shown_aligns), ('clicks', click_aligns),
                   ('shown+clicked', sc_aligns), ('shown+NOT clicked', snc_aligns)]:
    arr = np.array(arr)
    t_val, p_val = stats.ttest_1samp(arr, 0)
    d_val = cohens_d_one_sample(arr)
    print(f"  cos(shift, {name:25s}): mean={np.mean(arr):+.4f}  t={t_val:8.1f}  d={d_val:+.4f}  N={len(arr):,}")

shown_aligns = np.array(shown_aligns)
click_aligns = np.array(click_aligns)
sc_aligns = np.array(sc_aligns)
snc_aligns = np.array(snc_aligns)

t_pair, p_pair = stats.ttest_rel(
    sc_aligns[:min(len(sc_aligns), len(snc_aligns))],
    snc_aligns[:min(len(sc_aligns), len(snc_aligns))]
)
print(f"\n  Paired (shown+clicked vs shown+NC): t={t_pair:.1f}, p={p_pair:.2e}")

results['cosine_4way'] = {
    'shown_all': {'mean': float(np.mean(shown_aligns)), 't': float(stats.ttest_1samp(shown_aligns, 0)[0]),
                  'd': float(cohens_d_one_sample(shown_aligns)), 'n': len(shown_aligns)},
    'clicks': {'mean': float(np.mean(click_aligns)), 't': float(stats.ttest_1samp(click_aligns, 0)[0]),
               'd': float(cohens_d_one_sample(click_aligns)), 'n': len(click_aligns)},
    'shown_clicked': {'mean': float(np.mean(sc_aligns)), 't': float(stats.ttest_1samp(sc_aligns, 0)[0]),
                      'd': float(cohens_d_one_sample(sc_aligns)), 'n': len(sc_aligns)},
    'shown_nc': {'mean': float(np.mean(snc_aligns)), 't': float(stats.ttest_1samp(snc_aligns, 0)[0]),
                 'd': float(cohens_d_one_sample(snc_aligns)), 'n': len(snc_aligns)},
    'paired_t': float(t_pair), 'paired_p': float(p_pair),
}


# ============================================================
# ANALYSIS 3: THREE-WAY DECOMPOSITION
# ============================================================
print("\n" + "=" * 70)
print("3. Three-way decomposition (clicked / shown-NC / neither)")
print("=" * 70)

deltas_clicked, deltas_snc, deltas_neither = [], [], []
user_three = defaultdict(lambda: {'cl': [], 'snc': [], 'nei': []})

for uid in eligible_uids:
    v = user_vecs[uid]
    for ci in range(n_cats):
        was_shown = v['p2_shown_raw'][ci] > 0
        was_clicked = v['p2_click_raw'][ci] > 0
        delta = v['p3_vec'][ci] - v['p1_vec'][ci]
        if was_clicked:
            deltas_clicked.append(delta)
            user_three[uid]['cl'].append(delta)
        elif was_shown:
            deltas_snc.append(delta)
            user_three[uid]['snc'].append(delta)
        else:
            deltas_neither.append(delta)
            user_three[uid]['nei'].append(delta)

deltas_clicked = np.array(deltas_clicked)
deltas_snc = np.array(deltas_snc)
deltas_neither = np.array(deltas_neither)

for name, arr in [('Clicked', deltas_clicked), ('Shown-NC', deltas_snc), ('Neither', deltas_neither)]:
    t_val, p_val = stats.ttest_1samp(arr, 0)
    print(f"  {name:15s}: N={len(arr):>10,}  mean Δ={np.mean(arr):+.6f}  t={t_val:8.1f}")

# Pairwise
for n1, a1, n2, a2 in [('Clicked', deltas_clicked, 'Shown-NC', deltas_snc),
                         ('Clicked', deltas_clicked, 'Neither', deltas_neither),
                         ('Shown-NC', deltas_snc, 'Neither', deltas_neither)]:
    t_val, p_val = stats.ttest_ind(a1, a2)
    print(f"  {n1} - {n2}: diff={np.mean(a1)-np.mean(a2):+.6f}  t={t_val:.1f}  p={p_val:.2e}")

# Within-user paired
for label, k1, k2 in [('cl-snc', 'cl', 'snc'), ('cl-nei', 'cl', 'nei'), ('snc-nei', 'snc', 'nei')]:
    diffs = []
    for uid in user_three:
        d = user_three[uid]
        if len(d[k1]) > 0 and len(d[k2]) > 0:
            diffs.append(np.mean(d[k1]) - np.mean(d[k2]))
    diffs = np.array(diffs)
    t_val, p_val = stats.ttest_1samp(diffs, 0)
    d_val = cohens_d_one_sample(diffs)
    print(f"  Within-user ({label}): mean={np.mean(diffs):+.6f}  t={t_val:.1f}  d={d_val:.4f}  N={len(diffs):,}")

results['three_way'] = {
    'clicked': {'n': len(deltas_clicked), 'mean': float(np.mean(deltas_clicked))},
    'shown_nc': {'n': len(deltas_snc), 'mean': float(np.mean(deltas_snc))},
    'neither': {'n': len(deltas_neither), 'mean': float(np.mean(deltas_neither))},
}


# ============================================================
# ANALYSIS 4: QUINTILE DOSE-RESPONSE
# ============================================================
print("\n" + "=" * 70)
print("4. Quintile dose-response")
print("=" * 70)

dose_frac, dose_delta, dose_cl = [], [], []
for uid in eligible_uids:
    v = user_vecs[uid]
    for ci in range(n_cats):
        if v['p2_shown_raw'][ci] > 0 and v['p2_click_raw'][ci] == 0:
            dose_frac.append(v['p2_shown_vec'][ci])
            dose_delta.append(v['p3_vec'][ci] - v['p1_vec'][ci])
            dose_cl.append(uid)

dose_frac = np.array(dose_frac)
dose_delta = np.array(dose_delta)

rho, p_rho = stats.spearmanr(dose_frac, dose_delta)
print(f"  Spearman ρ = {rho:.4f}, p = {p_rho:.2e}")

dose_bins = pd.qcut(dose_frac, 5, labels=False, duplicates='drop')
print(f"\n  {'Q':<4s} {'Mean frac':>10s} {'Mean Δ':>12s} {'N':>10s}")
print("  " + "-" * 40)
quintile_data = []
for b in sorted(np.unique(dose_bins)):
    m = dose_bins == b
    qd = {'q': int(b), 'frac': float(np.mean(dose_frac[m])),
           'delta': float(np.mean(dose_delta[m])), 'n': int(m.sum())}
    quintile_data.append(qd)
    print(f"  {b:<4d} {qd['frac']:10.4f} {qd['delta']:+12.6f} {qd['n']:10,}")

# OLS
X_d = np.column_stack([np.ones(len(dose_frac)), dose_frac])
cl_d = np.array([uid_idx.get(uid, 0) for uid in dose_cl])
res_d = ols_clustered(dose_delta, X_d, cl_d)
print(f"\n  OLS: Δ = {res_d['beta'][0]:.6f} + {res_d['beta'][1]:.6f} × frac  (t={res_d['t_clust'][1]:.1f})")

results['dose_quintile'] = {
    'rho': float(rho), 'bins': quintile_data,
    'ols_intercept': float(res_d['beta'][0]), 'ols_beta': float(res_d['beta'][1]),
    'ols_t': float(res_d['t_clust'][1]),
}


# ============================================================
# ANALYSIS 5: FULL PER-CATEGORY DECOMPOSITION
# ============================================================
print("\n" + "=" * 70)
print("5. Full per-category decomposition")
print("=" * 70)

cat_decomp = []
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
        cat_decomp.append({
            'category': cname,
            'clicked_mean': float(np.mean(cl_d)),
            'snc_mean': float(np.mean(snc_d)),
            'neither_mean': float(np.mean(nei_d)),
            'click_effect': float(np.mean(cl_d) - np.mean(nei_d)),
            'exposure_effect': float(np.mean(snc_d) - np.mean(nei_d)),
            'n_cl': len(cl_d), 'n_snc': len(snc_d), 'n_nei': len(nei_d),
        })

print(f"\n{'Category':<16s} {'Clicked':>9s} {'Shown-NC':>9s} {'Neither':>9s} {'Click eff':>10s} {'Expos eff':>10s}")
print("-" * 67)
for r in sorted(cat_decomp, key=lambda x: x['click_effect'], reverse=True):
    print(f"{r['category']:<16s} {r['clicked_mean']:+9.4f} {r['snc_mean']:+9.4f} "
          f"{r['neither_mean']:+9.4f} {r['click_effect']:+10.4f} {r['exposure_effect']:+10.4f}")

n_neg = sum(1 for r in cat_decomp if r['exposure_effect'] < 0)
print(f"\nExposure effect negative: {n_neg}/{len(cat_decomp)} categories")

results['per_category'] = cat_decomp


# ============================================================
# ANALYSIS 6: MAGNITUDE ASYMMETRY
# ============================================================
print("\n" + "=" * 70)
print("6. Magnitude asymmetry (shown-not-clicked)")
print("=" * 70)

n_toward, n_away, n_zero = 0, 0, 0
mag_toward, mag_away = [], []

for uid in eligible_uids:
    v = user_vecs[uid]
    for ci in range(n_cats):
        if v['p2_shown_raw'][ci] > 0 and v['p2_click_raw'][ci] == 0:
            delta = v['p3_vec'][ci] - v['p1_vec'][ci]
            if delta > 0:
                n_toward += 1
                mag_toward.append(delta)
            elif delta < 0:
                n_away += 1
                mag_away.append(abs(delta))
            else:
                n_zero += 1

total = n_toward + n_away + n_zero
print(f"  Total shown-NC pairs: {total:,}")
print(f"  Toward:  {n_toward:>10,}  ({n_toward/total:.4%})")
print(f"  Away:    {n_away:>10,}  ({n_away/total:.4%})")
print(f"  Zero:    {n_zero:>10,}  ({n_zero/total:.4%})")

mag_toward = np.array(mag_toward)
mag_away = np.array(mag_away)
t_mag, p_mag = stats.ttest_ind(mag_away, mag_toward)
print(f"\n  Mean |toward|: {np.mean(mag_toward):.6f}")
print(f"  Mean |away|:   {np.mean(mag_away):.6f}  ({np.mean(mag_away)/np.mean(mag_toward):.1f}x)")
print(f"  t={t_mag:.2f}, p={p_mag:.2e}")

results['magnitude'] = {
    'total': total, 'n_toward': n_toward, 'n_away': n_away, 'n_zero': n_zero,
    'frac_toward': n_toward/total, 'frac_away': n_away/total,
    'mean_mag_toward': float(np.mean(mag_toward)),
    'mean_mag_away': float(np.mean(mag_away)),
    'ratio': float(np.mean(mag_away)/np.mean(mag_toward)),
    't': float(t_mag), 'p': float(p_mag),
}


# ============================================================
# SAVE RESULTS
# ============================================================
with open(os.path.join(OUT, "decomposed_exposure_results.json"), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT}/decomposed_exposure_results.json")


# ============================================================
# FIGURES
# ============================================================
print("\nGenerating figures...")
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Decomposed Exposure Analysis", fontsize=14, fontweight='bold')

# Panel 1: Decomposed OLS
ax = axes[0, 0]
coefs = res_dec['beta'][1:]
ses = res_dec['se_clust'][1:]
labels = ['P1 taste', 'Shown+\nClicked', 'Shown+\nNOT Clicked']
colors = ['#2196F3', '#4CAF50', '#F44336']
ax.bar(range(3), coefs, yerr=1.96*ses, capsize=5, color=colors, alpha=0.8, edgecolor='black')
ax.axhline(0, color='black', ls='--', lw=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('Coefficient')
ax.set_title('Decomposed OLS')

# Panel 2: 4-way cosine
ax = axes[0, 1]
cos_data = [
    ('Shown\n(all)', np.mean(shown_aligns), '#9C27B0'),
    ('Clicks', np.mean(click_aligns), '#FF9800'),
    ('Shown+\nClicked', np.mean(sc_aligns), '#4CAF50'),
    ('Shown+\nNot Clicked', np.mean(snc_aligns), '#F44336'),
]
for i, (lab, val, col) in enumerate(cos_data):
    ax.bar(i, val, color=col, alpha=0.8, edgecolor='black')
ax.axhline(0, color='black', ls='--', lw=0.5)
ax.set_xticks(range(4))
ax.set_xticklabels([d[0] for d in cos_data], fontsize=8)
ax.set_ylabel('Mean cos(shift, vector)')
ax.set_title('Taste Shift Alignment')

# Panel 3: Three-way decomposition
ax = axes[0, 2]
means3 = [np.mean(deltas_clicked), np.mean(deltas_snc), np.mean(deltas_neither)]
sems3 = [np.std(deltas_clicked)/np.sqrt(len(deltas_clicked)),
         np.std(deltas_snc)/np.sqrt(len(deltas_snc)),
         np.std(deltas_neither)/np.sqrt(len(deltas_neither))]
ax.bar(['Clicked', 'Shown\nnot clicked', 'Neither'], means3,
       yerr=[1.96*s for s in sems3], capsize=5,
       color=['#4CAF50', '#F44336', '#9E9E9E'], alpha=0.8, edgecolor='black')
ax.axhline(0, color='black', ls='--', lw=0.5)
ax.set_ylabel('Mean taste shift (P3 - P1)')
ax.set_title('Three-Way Decomposition')

# Panel 4: Quintile dose-response
ax = axes[1, 0]
qx = [q['frac'] for q in quintile_data]
qy = [q['delta'] for q in quintile_data]
ax.plot(qx, qy, 'o-', color='#E91E63', markersize=6)
ax.axhline(0, color='black', ls='--', lw=0.5)
ax.set_xlabel('Shown fraction')
ax.set_ylabel('Taste shift (P3 - P1)')
ax.set_title(f'Dose-Response (ρ = {rho:.3f})')

# Panel 5: Per-category
ax = axes[1, 1]
cat_sorted = sorted(cat_decomp, key=lambda x: x['click_effect'], reverse=True)
cat_names = [r['category'] for r in cat_sorted]
cl_eff = [r['click_effect'] for r in cat_sorted]
ex_eff = [r['exposure_effect'] for r in cat_sorted]
y_pos = np.arange(len(cat_names))
w = 0.35
ax.barh(y_pos - w/2, cl_eff, w, color='#4CAF50', alpha=0.8, edgecolor='black', label='Click effect')
ax.barh(y_pos + w/2, ex_eff, w, color='#F44336', alpha=0.8, edgecolor='black', label='Exposure effect')
ax.axvline(0, color='black', ls='--', lw=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(cat_names, fontsize=8)
ax.set_xlabel('Effect size')
ax.set_title('Per-Category Effects')
ax.legend(fontsize=8, loc='lower right')
ax.invert_yaxis()

# Panel 6: Magnitude asymmetry
ax = axes[1, 2]
ax.bar(['Toward\n(reinforced)', 'Away\n(backfire)', 'No change'],
       [n_toward/total, n_away/total, n_zero/total],
       color=['#4CAF50', '#F44336', '#9E9E9E'], alpha=0.8, edgecolor='black')
ax.set_ylabel('Fraction')
ax.set_title('Shown-Not-Clicked: Shift Direction')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "decomposed_exposure.png"), dpi=150, bbox_inches='tight')
fig.savefig(os.path.join(OUT, "decomposed_exposure.pdf"), bbox_inches='tight')
print(f"Saved: {OUT}/decomposed_exposure.png")

print("\nDone.")
