"""
Category-Level Click Prediction Robustness Check
=================================================
Replicates Step 1 of mind_full_analysis.py using the 14-dim category-share
taste vector instead of the 285-dim subcategory one-hot vector.

Purpose: Test whether the inner-product utility model holds at the category
level (consistent with the theory model), not just at the subcategory level.

Also tests:
  - Raw inner product (not cosine) as predictor — matches theory primitive
  - Quintile analysis at category level
  - MNL AUC at category level

Author: Hossein Piri / Claude
Date: 2026-04-22
"""

import os, json, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

BASE = "/Users/piri/Desktop/Recommendation Systems/Mind-Data-Large"
OUT = "/Users/piri/Desktop/Recommendation Systems/code/output_full"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("CATEGORY-LEVEL CLICK PREDICTION — ROBUSTNESS CHECK")
print("=" * 70)

# ============================================================
# 0. LOAD DATA (same as mind_full_analysis.py)
# ============================================================
print("\n[0] LOADING DATA...")

news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract',
             'url', 'title_entities', 'abstract_entities']

news_all = []
for split in ['MINDlarge_train', 'MINDlarge_dev']:
    df = pd.read_csv(os.path.join(BASE, split, "news.tsv"), sep='\t',
                      header=None, names=news_cols)
    news_all.append(df)
news = pd.concat(news_all).drop_duplicates(subset='news_id')
print(f"  Articles: {len(news):,}")

news_cat = dict(zip(news['news_id'], news['category']))

cats = sorted(news['category'].dropna().unique())
cat_to_idx = {c: i for i, c in enumerate(cats)}
n_cats = len(cats)
print(f"  Categories: {n_cats} — {cats}")

# Also load subcategory info for comparison
news_subcat = dict(zip(news['news_id'], news['subcategory']))
subcats = sorted(news['subcategory'].dropna().unique())
subcat_to_idx = {s: i for i, s in enumerate(subcats)}
n_subcats = len(subcats)

# --- Behaviors ---
print("  Loading behaviors...")
beh_cols = ['impression_id', 'user_id', 'timestamp', 'click_history', 'impressions']
beh_train = pd.read_csv(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
                          sep='\t', header=None, names=beh_cols)
beh_dev = pd.read_csv(os.path.join(BASE, "MINDlarge_dev", "behaviors.tsv"),
                        sep='\t', header=None, names=beh_cols)
beh = pd.concat([beh_train, beh_dev], ignore_index=True)
del beh_train, beh_dev
print(f"  Total impressions: {len(beh):,}")

# Parse
def parse_history(s):
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '':
        return []
    return s.strip().split()

def parse_impressions(s):
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '':
        return [], []
    clicked, shown = [], []
    for item in s.strip().split():
        nid, label = item.rsplit('-', 1)
        shown.append(nid)
        if label == '1':
            clicked.append(nid)
    return clicked, shown

beh['hist_list'] = beh['click_history'].apply(parse_history)
parsed = beh['impressions'].apply(parse_impressions)
beh['clicked_list'] = parsed.apply(lambda x: x[0])
beh['shown_list'] = parsed.apply(lambda x: x[1])
beh['n_hist'] = beh['hist_list'].apply(len)

print(f"  Parsed. Users: {beh['user_id'].nunique():,}")

# ============================================================
# SAMPLE (same seed/size as original for comparability)
# ============================================================
np.random.seed(42)
SAMPLE_N = 50000

eligible = beh[beh['n_hist'] >= 5]['user_id'].unique()
sample_uids = set(np.random.choice(eligible, min(SAMPLE_N, len(eligible)), replace=False))
beh_sample = beh[beh['user_id'].isin(sample_uids)].copy()
print(f"\n  Sampled {len(sample_uids):,} users, {len(beh_sample):,} impressions")


# ============================================================
# BUILD USER TASTE VECTORS
# ============================================================

# --- Category-share taste vector (K-dim, sums to 1) ---
def build_user_taste_cat(hist_nids):
    """Category-share taste vector: fraction of clicks in each category."""
    vec = np.zeros(n_cats)
    for nid in hist_nids:
        c = news_cat.get(nid)
        if c and c in cat_to_idx:
            vec[cat_to_idx[c]] += 1
    s = vec.sum()
    if s > 0:
        vec /= s
    else:
        return None
    return vec

# --- Article category vector (one-hot, K-dim) ---
def get_cat_onehot(nid):
    """One-hot category vector for an article."""
    c = news_cat.get(nid)
    if c and c in cat_to_idx:
        vec = np.zeros(n_cats)
        vec[cat_to_idx[c]] = 1.0
        return vec
    return None

# --- Subcategory taste (for comparison) ---
def get_subcat_emb(nid):
    sc = news_subcat.get(nid)
    if sc and sc in subcat_to_idx:
        vec = np.zeros(n_subcats)
        vec[subcat_to_idx[sc]] = 1.0
        return vec
    return None

def build_user_taste_subcat(hist_nids):
    vecs = []
    for nid in hist_nids:
        v = get_subcat_emb(nid)
        if v is not None:
            vecs.append(v)
    if len(vecs) == 0:
        return None
    return np.mean(vecs, axis=0)

print("\n  Building user taste vectors...")
user_taste_cat = {}
user_taste_subcat = {}

for uid in sample_uids:
    rows = beh_sample[beh_sample['user_id'] == uid]
    best_idx = rows['n_hist'].idxmax()
    hist = rows.loc[best_idx, 'hist_list']

    v_cat = build_user_taste_cat(hist)
    if v_cat is not None:
        user_taste_cat[uid] = v_cat

    v_sub = build_user_taste_subcat(hist)
    if v_sub is not None:
        user_taste_subcat[uid] = v_sub

print(f"  Users with category taste:    {len(user_taste_cat):,}")
print(f"  Users with subcategory taste:  {len(user_taste_subcat):,}")


# ============================================================
# SIMILARITY/DISTANCE FUNCTIONS
# ============================================================

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return np.dot(a, b) / (na * nb)

def inner_product(a, b):
    return np.dot(a, b)

def hellinger_dist(p, q):
    """Hellinger distance between two distributions."""
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q))**2))

def total_variation(p, q):
    """Total variation distance between two distributions."""
    return 0.5 * np.sum(np.abs(p - q))


# ============================================================
# STEP 1: CLICK PREDICTION AT CATEGORY LEVEL
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: CLICK PREDICTION — CATEGORY vs SUBCATEGORY")
print("=" * 70)

print("\n--- Building click prediction dataset ---")
pred_rows = []
n_processed = 0

for _, row in beh_sample.iterrows():
    uid = row['user_id']
    if uid not in user_taste_cat:
        continue

    u_cat = user_taste_cat[uid]
    u_sub = user_taste_subcat.get(uid)

    clicked_set = set(row['clicked_list'])
    shown = row['shown_list']

    for nid in shown:
        a_cat = get_cat_onehot(nid)
        if a_cat is None:
            continue

        # Category-level metrics
        ip_cat = inner_product(u_cat, a_cat)     # = theta_k (the theory's primitive)
        cos_cat = cosine_sim(u_cat, a_cat)
        hell_cat = hellinger_dist(u_cat, a_cat)
        tv_cat = total_variation(u_cat, a_cat)

        # Subcategory-level (for comparison)
        cos_sub = np.nan
        ip_sub = np.nan
        if u_sub is not None:
            a_sub = get_subcat_emb(nid)
            if a_sub is not None:
                cos_sub = cosine_sim(u_sub, a_sub)
                ip_sub = inner_product(u_sub, a_sub)

        # Article category
        art_cat = news_cat.get(nid, '')

        pred_rows.append({
            'uid': uid,
            'nid': nid,
            'clicked': int(nid in clicked_set),
            'ip_cat': ip_cat,           # theta_k — theory primitive
            'cos_cat': cos_cat,
            'hell_cat': hell_cat,
            'tv_cat': tv_cat,
            'cos_sub': cos_sub,
            'ip_sub': ip_sub,
            'art_cat': art_cat,
            'n_shown': len(shown),
        })

    n_processed += 1
    if n_processed % 50000 == 0:
        print(f"  Processed {n_processed:,} impressions, {len(pred_rows):,} rows...")

df_pred = pd.DataFrame(pred_rows)
print(f"\nTotal article-level observations: {len(df_pred):,}")
print(f"Click rate: {df_pred['clicked'].mean():.4f}")


# ============================================================
# 1a: AUC COMPARISON — ALL METRICS
# ============================================================
print("\n" + "-" * 50)
print("1a: AUC COMPARISON ACROSS REPRESENTATIONS AND METRICS")
print("-" * 50)

results = {}

# Category-level metrics
auc_ip_cat = roc_auc_score(df_pred['clicked'], df_pred['ip_cat'])
results['Inner product (category, 14-dim)'] = auc_ip_cat
print(f"  AUC — Inner product θ_k (category, 14-dim):  {auc_ip_cat:.4f}  ← theory primitive")

auc_cos_cat = roc_auc_score(df_pred['clicked'], df_pred['cos_cat'])
results['Cosine similarity (category, 14-dim)'] = auc_cos_cat
print(f"  AUC — Cosine similarity (category, 14-dim):  {auc_cos_cat:.4f}")

# Hellinger and TV are distances (higher = less similar), so negate for AUC
auc_hell = roc_auc_score(df_pred['clicked'], -df_pred['hell_cat'])
results['Hellinger distance (category, 14-dim)'] = auc_hell
print(f"  AUC — Hellinger distance (category, 14-dim): {auc_hell:.4f}")

auc_tv = roc_auc_score(df_pred['clicked'], -df_pred['tv_cat'])
results['Total variation (category, 14-dim)'] = auc_tv
print(f"  AUC — Total variation (category, 14-dim):    {auc_tv:.4f}")

# Subcategory-level (for comparison)
df_sub_valid = df_pred.dropna(subset=['cos_sub'])
if len(df_sub_valid) > 1000:
    auc_cos_sub = roc_auc_score(df_sub_valid['clicked'], df_sub_valid['cos_sub'])
    results['Cosine similarity (subcategory, 285-dim)'] = auc_cos_sub
    print(f"  AUC — Cosine similarity (subcat, 285-dim):   {auc_cos_sub:.4f}  ← original paper")

    auc_ip_sub = roc_auc_score(df_sub_valid['clicked'], df_sub_valid['ip_sub'])
    results['Inner product (subcategory, 285-dim)'] = auc_ip_sub
    print(f"  AUC — Inner product (subcat, 285-dim):        {auc_ip_sub:.4f}")


# ============================================================
# 1b: QUINTILE ANALYSIS — CATEGORY LEVEL
# ============================================================
print("\n" + "-" * 50)
print("1b: QUINTILE ANALYSIS — Category-level inner product (θ_k)")
print("-" * 50)
print("  (This is the theory's click probability: g(θ_k))")

df_pred['ip_quintile'] = pd.qcut(df_pred['ip_cat'], 5, labels=False, duplicates='drop')
q_stats = df_pred.groupby('ip_quintile').agg(
    mean_ip=('ip_cat', 'mean'),
    ctr=('clicked', 'mean'),
    n=('clicked', 'count')
).reset_index()

print(f"\n{'Quintile':>10} {'Mean θ_k':>10} {'CTR':>10} {'N':>12}")
print("-" * 45)
for _, r in q_stats.iterrows():
    print(f"{int(r['ip_quintile'])+1:>10} {r['mean_ip']:>10.4f} {r['ctr']:>10.4f} {int(r['n']):>12,}")

# Correlation
corr_ip, p_ip = stats.pointbiserialr(df_pred['clicked'], df_pred['ip_cat'])
print(f"\nPoint-biserial r (θ_k ↔ click): {corr_ip:.4f}, p = {p_ip:.2e}")

# Also for cosine at category level
corr_cos, p_cos = stats.pointbiserialr(df_pred['clicked'], df_pred['cos_cat'])
print(f"Point-biserial r (cos ↔ click):  {corr_cos:.4f}, p = {p_cos:.2e}")


# ============================================================
# 1c: QUINTILE ANALYSIS — COSINE SIMILARITY (category level)
# ============================================================
print("\n" + "-" * 50)
print("1c: QUINTILE ANALYSIS — Category-level cosine similarity")
print("-" * 50)

df_pred['cos_quintile'] = pd.qcut(df_pred['cos_cat'], 5, labels=False, duplicates='drop')
q_cos = df_pred.groupby('cos_quintile').agg(
    mean_cos=('cos_cat', 'mean'),
    ctr=('clicked', 'mean'),
    n=('clicked', 'count')
).reset_index()

print(f"\n{'Quintile':>10} {'Mean cos':>10} {'CTR':>10} {'N':>12}")
print("-" * 45)
for _, r in q_cos.iterrows():
    print(f"{int(r['cos_quintile'])+1:>10} {r['mean_cos']:>10.4f} {r['ctr']:>10.4f} {int(r['n']):>12,}")


# ============================================================
# 1d: MNL MODEL — CATEGORY LEVEL
# ============================================================
print("\n" + "-" * 50)
print("1d: MNL CHOICE MODEL — CATEGORY LEVEL")
print("-" * 50)

# Inner product as utility
mnl_actual_ip = []
mnl_predicted_ip = []

# Cosine as utility
mnl_actual_cos = []
mnl_predicted_cos = []

# Subcategory cosine (original, for comparison)
mnl_actual_sub = []
mnl_predicted_sub = []

n_imp = 0

for _, row in beh_sample.iterrows():
    uid = row['user_id']
    if uid not in user_taste_cat:
        continue
    clicked_set = set(row['clicked_list'])
    shown = row['shown_list']
    if len(shown) < 2 or len(clicked_set) == 0:
        continue

    u_cat = user_taste_cat[uid]
    u_sub = user_taste_subcat.get(uid)

    # Category-level inner product utilities
    utils_ip = []
    utils_cos = []
    utils_sub = []
    labels = []
    valid_sub = True

    for nid in shown:
        a_cat = get_cat_onehot(nid)
        if a_cat is None:
            utils_ip.append(0)
            utils_cos.append(0)
        else:
            utils_ip.append(inner_product(u_cat, a_cat))
            utils_cos.append(cosine_sim(u_cat, a_cat))

        if u_sub is not None:
            a_sub = get_subcat_emb(nid)
            if a_sub is not None:
                utils_sub.append(cosine_sim(u_sub, a_sub))
            else:
                utils_sub.append(0)
                valid_sub = False
        else:
            valid_sub = False

        labels.append(int(nid in clicked_set))

    utils_ip = np.array(utils_ip)
    utils_cos = np.array(utils_cos)
    labels = np.array(labels)

    # MNL probabilities — inner product
    exp_u = np.exp(utils_ip * 5)
    probs_ip = exp_u / exp_u.sum()

    # MNL probabilities — cosine
    exp_u_cos = np.exp(utils_cos * 5)
    probs_cos = exp_u_cos / exp_u_cos.sum()

    for i in range(len(shown)):
        mnl_actual_ip.append(labels[i])
        mnl_predicted_ip.append(probs_ip[i])
        mnl_actual_cos.append(labels[i])
        mnl_predicted_cos.append(probs_cos[i])

    # Subcategory-level MNL
    if valid_sub and len(utils_sub) == len(shown):
        utils_sub = np.array(utils_sub)
        exp_u_sub = np.exp(utils_sub * 5)
        probs_sub = exp_u_sub / exp_u_sub.sum()
        for i in range(len(shown)):
            mnl_actual_sub.append(labels[i])
            mnl_predicted_sub.append(probs_sub[i])

    n_imp += 1
    if n_imp >= 100000:
        break

mnl_actual_ip = np.array(mnl_actual_ip)
mnl_predicted_ip = np.array(mnl_predicted_ip)
mnl_actual_cos = np.array(mnl_actual_cos)
mnl_predicted_cos = np.array(mnl_predicted_cos)

auc_mnl_ip = roc_auc_score(mnl_actual_ip, mnl_predicted_ip)
auc_mnl_cos = roc_auc_score(mnl_actual_cos, mnl_predicted_cos)
print(f"  MNL AUC (category inner product):  {auc_mnl_ip:.4f}  ← theory primitive")
print(f"  MNL AUC (category cosine):         {auc_mnl_cos:.4f}")
print(f"  (on {n_imp:,} impressions)")

if len(mnl_actual_sub) > 1000:
    mnl_actual_sub = np.array(mnl_actual_sub)
    mnl_predicted_sub = np.array(mnl_predicted_sub)
    auc_mnl_sub = roc_auc_score(mnl_actual_sub, mnl_predicted_sub)
    print(f"  MNL AUC (subcategory cosine):      {auc_mnl_sub:.4f}  ← original paper")


# ============================================================
# 1e: PER-CATEGORY CTR BREAKDOWN
# ============================================================
print("\n" + "-" * 50)
print("1e: PER-CATEGORY CTR vs MEAN θ_k")
print("-" * 50)
print("  (Does θ_k predict CTR within each category?)")

cat_stats = df_pred.groupby('art_cat').agg(
    mean_theta_k=('ip_cat', 'mean'),
    ctr=('clicked', 'mean'),
    n=('clicked', 'count'),
    corr_ip=('ip_cat', lambda x: stats.pointbiserialr(
        df_pred.loc[x.index, 'clicked'], x)[0] if len(x) > 100 else np.nan)
).reset_index()

print(f"\n{'Category':>20} {'Mean θ_k':>10} {'CTR':>8} {'r(θ_k,click)':>14} {'N':>10}")
print("-" * 65)
for _, r in cat_stats.sort_values('ctr', ascending=False).iterrows():
    print(f"{r['art_cat']:>20} {r['mean_theta_k']:>10.4f} {r['ctr']:>8.4f} {r['corr_ip']:>14.4f} {int(r['n']):>10,}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n  {'Metric':<45} {'AUC':>8}")
print("  " + "-" * 55)
for name, auc in sorted(results.items(), key=lambda x: -x[1]):
    marker = ""
    if "Inner product (category" in name:
        marker = " ← theory"
    elif "Cosine similarity (subcategory" in name:
        marker = " ← original"
    print(f"  {name:<45} {auc:>8.4f}{marker}")

print(f"\n  MNL AUC (category inner product):  {auc_mnl_ip:.4f}  ← theory")
print(f"  MNL AUC (category cosine):         {auc_mnl_cos:.4f}")
if len(mnl_actual_sub) > 1000:
    print(f"  MNL AUC (subcategory cosine):      {auc_mnl_sub:.4f}  ← original")

print("\n  Key question: Is category-level inner product θ_k a reasonable")
print("  predictor of clicks? AUC > 0.55 supports the theory model.")
print("=" * 70)

# Save results
import json
results_out = {
    'auc_comparison': results,
    'mnl_auc_cat_ip': float(auc_mnl_ip),
    'mnl_auc_cat_cos': float(auc_mnl_cos),
    'quintile_ip': q_stats.to_dict('records'),
    'quintile_cos': q_cos.to_dict('records'),
    'per_category': cat_stats.to_dict('records'),
    'correlation_ip': float(corr_ip),
    'correlation_cos': float(corr_cos),
}
if len(mnl_actual_sub) > 1000:
    results_out['mnl_auc_sub_cos'] = float(auc_mnl_sub)

with open(os.path.join(OUT, 'category_level_auc_results.json'), 'w') as f:
    json.dump(results_out, f, indent=2, default=str)
print(f"\nResults saved to {OUT}/category_level_auc_results.json")
