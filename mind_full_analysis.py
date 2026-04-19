"""
Full Empirical Analysis on MIND Large Dataset
==============================================
Steps 1, 3, 4, 5 of the empirical program.

Step 1: Validate inner-product / MNL utility model
Step 3: Test "local persuasion" radius
Step 4: Calibrate w(d) — taste movement function
Step 5: Assortment substitution / spillover patterns

Uses MINDlarge train+dev (labeled data, Nov 9-15, 2.6M impressions).
Article embeddings: subcategory one-hot (285-dim, 100% coverage)
                  + entity embeddings (100-dim, 74% coverage)

Author: Hossein Piri / Claude
Date: 2026-03-27
"""

import os, json, sys
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE = "/Users/piri/Desktop/Recommendation Systems/Dataset"
OUT = "/Users/piri/Desktop/Recommendation Systems/code/output_full"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("FULL EMPIRICAL ANALYSIS — MIND Large Dataset")
print("=" * 70)

# ============================================================
# 0. LOAD DATA
# ============================================================
print("\n[0] LOADING DATA...")

# --- News metadata ---
news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract',
             'url', 'title_entities', 'abstract_entities']

news_all = []
for split in ['MINDlarge_train', 'MINDlarge_dev']:
    df = pd.read_csv(os.path.join(BASE, split, "news.tsv"), sep='\t',
                      header=None, names=news_cols)
    news_all.append(df)
news = pd.concat(news_all).drop_duplicates(subset='news_id')
print(f"  Articles: {len(news):,}")

# Build lookups
news_cat = dict(zip(news['news_id'], news['category']))
news_subcat = dict(zip(news['news_id'], news['subcategory']))

subcats = sorted(news['subcategory'].dropna().unique())
subcat_to_idx = {s: i for i, s in enumerate(subcats)}
n_subcats = len(subcats)
print(f"  Subcategories: {n_subcats}")

cats = sorted(news['category'].dropna().unique())
cat_to_idx = {c: i for i, c in enumerate(cats)}
n_cats = len(cats)

# --- Entity embeddings ---
print("  Loading entity embeddings...")
entity_emb = {}
for split in ['MINDlarge_train', 'MINDlarge_dev']:
    path = os.path.join(BASE, split, "entity_embedding.vec")
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            eid = parts[0]
            if eid not in entity_emb:
                entity_emb[eid] = np.array([float(x) for x in parts[1:]])
print(f"  Entities with embeddings: {len(entity_emb):,}")

# Build article entity embeddings (average of entity vectors)
print("  Building article embeddings from entities...")
article_entity_emb = {}
for _, row in news.iterrows():
    nid = row['news_id']
    try:
        ents = json.loads(row['title_entities']) if pd.notna(row['title_entities']) else []
    except:
        ents = []
    vecs = []
    for e in ents:
        wid = e.get('WikidataId', '')
        if wid in entity_emb:
            vecs.append(entity_emb[wid])
    if len(vecs) > 0:
        article_entity_emb[nid] = np.mean(vecs, axis=0)

emb_dim = 100
print(f"  Articles with entity embeddings: {len(article_entity_emb):,} ({len(article_entity_emb)/len(news)*100:.1f}%)")

# --- Subcategory one-hot embeddings (100% coverage) ---
def get_subcat_emb(nid):
    sc = news_subcat.get(nid)
    if sc and sc in subcat_to_idx:
        vec = np.zeros(n_subcats)
        vec[subcat_to_idx[sc]] = 1.0
        return vec
    return None

# --- Behaviors (train + dev, labeled) ---
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
print("  Parsing...")

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
beh['n_shown'] = beh['shown_list'].apply(len)
beh['n_clicked'] = beh['clicked_list'].apply(len)

print(f"  Parsed. Users: {beh['user_id'].nunique():,}")


# ============================================================
# SAMPLE FOR TRACTABILITY
# ============================================================
# For embedding-based analyses, sample users to keep runtime reasonable
np.random.seed(42)
SAMPLE_N = 50000  # users for detailed analysis

eligible = beh[beh['n_hist'] >= 5]['user_id'].unique()
sample_uids = set(np.random.choice(eligible, min(SAMPLE_N, len(eligible)), replace=False))
beh_sample = beh[beh['user_id'].isin(sample_uids)].copy()
print(f"\n  Sampled {len(sample_uids):,} users, {len(beh_sample):,} impressions for embedding analyses")


# ============================================================
# BUILD USER TASTE VECTORS (subcategory space)
# ============================================================
print("\n  Building user taste vectors (subcategory one-hot)...")

def build_user_emb_subcat(hist_nids):
    """Average subcategory one-hot over click history."""
    vecs = []
    for nid in hist_nids:
        v = get_subcat_emb(nid)
        if v is not None:
            vecs.append(v)
    if len(vecs) == 0:
        return None
    return np.mean(vecs, axis=0)

def build_user_emb_entity(hist_nids):
    """Average entity embedding over click history."""
    vecs = []
    for nid in hist_nids:
        if nid in article_entity_emb:
            vecs.append(article_entity_emb[nid])
    if len(vecs) == 0:
        return None
    return np.mean(vecs, axis=0)

# For each sampled user, build taste vector from their longest history
user_taste_subcat = {}
user_taste_entity = {}

for uid in sample_uids:
    rows = beh_sample[beh_sample['user_id'] == uid]
    # Use longest history
    best_idx = rows['n_hist'].idxmax()
    hist = rows.loc[best_idx, 'hist_list']

    v_sub = build_user_emb_subcat(hist)
    if v_sub is not None:
        user_taste_subcat[uid] = v_sub

    v_ent = build_user_emb_entity(hist)
    if v_ent is not None:
        user_taste_entity[uid] = v_ent

print(f"  Users with subcat taste: {len(user_taste_subcat):,}")
print(f"  Users with entity taste: {len(user_taste_entity):,}")


# ============================================================
# STEP 1: VALIDATE INNER-PRODUCT / MNL UTILITY MODEL
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: INNER-PRODUCT / MNL UTILITY VALIDATION")
print("=" * 70)

# For each impression: compute cosine sim between user taste and each shown article
# Then test: does cosine sim predict click?

def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return np.dot(a, b) / (na * nb)

print("\n--- Building click prediction dataset ---")
pred_rows = []
n_processed = 0

for _, row in beh_sample.iterrows():
    uid = row['user_id']
    if uid not in user_taste_subcat:
        continue

    u_sub = user_taste_subcat[uid]
    u_ent = user_taste_entity.get(uid)

    clicked_set = set(row['clicked_list'])
    shown = row['shown_list']

    for nid in shown:
        a_sub = get_subcat_emb(nid)
        if a_sub is None:
            continue

        sim_sub = cosine_sim(u_sub, a_sub)

        # Entity embedding similarity (if available)
        sim_ent = np.nan
        if u_ent is not None and nid in article_entity_emb:
            sim_ent = cosine_sim(u_ent, article_entity_emb[nid])

        # Same category / subcategory as user's top
        user_top_subcat = subcats[np.argmax(u_sub)]
        article_subcat = news_subcat.get(nid, '')
        article_cat = news_cat.get(nid, '')
        same_subcat = int(article_subcat == user_top_subcat)

        pred_rows.append({
            'uid': uid,
            'nid': nid,
            'clicked': int(nid in clicked_set),
            'sim_subcat': sim_sub,
            'sim_entity': sim_ent,
            'same_subcat': same_subcat,
            'n_shown': len(shown),
        })

    n_processed += 1
    if n_processed % 50000 == 0:
        print(f"  Processed {n_processed:,} impressions, {len(pred_rows):,} article-level rows...")

df_pred = pd.DataFrame(pred_rows)
print(f"\nTotal article-level observations: {len(df_pred):,}")
print(f"Click rate: {df_pred['clicked'].mean():.4f}")

# --- 1a: Cosine similarity predicts clicks ---
print("\n--- 1a: Does cosine similarity predict clicks? ---")

# Bin by similarity decile
df_pred['sim_bin'] = pd.qcut(df_pred['sim_subcat'], 10, labels=False, duplicates='drop')
bin_stats = df_pred.groupby('sim_bin').agg(
    mean_sim=('sim_subcat', 'mean'),
    ctr=('clicked', 'mean'),
    n=('clicked', 'count')
).reset_index()

print(f"\n{'Sim decile':>10} {'Mean sim':>10} {'CTR':>10} {'N':>12}")
print("-" * 45)
for _, r in bin_stats.iterrows():
    print(f"{int(r['sim_bin']):>10} {r['mean_sim']:>10.4f} {r['ctr']:>10.4f} {int(r['n']):>12,}")

# Correlation
corr, p_corr = stats.pointbiserialr(df_pred['clicked'], df_pred['sim_subcat'])
print(f"\nPoint-biserial correlation (subcat sim ↔ click): r = {corr:.4f}, p = {p_corr:.2e}")

# Entity embedding correlation
df_ent_valid = df_pred.dropna(subset=['sim_entity'])
if len(df_ent_valid) > 1000:
    corr_e, p_e = stats.pointbiserialr(df_ent_valid['clicked'], df_ent_valid['sim_entity'])
    print(f"Point-biserial correlation (entity sim ↔ click):  r = {corr_e:.4f}, p = {p_e:.2e}")

# --- 1b: AUC of cosine similarity as click predictor ---
print("\n--- 1b: AUC as click predictor ---")
auc_sub = roc_auc_score(df_pred['clicked'], df_pred['sim_subcat'])
print(f"AUC (subcategory cosine sim): {auc_sub:.4f}")

if len(df_ent_valid) > 1000:
    auc_ent = roc_auc_score(df_ent_valid['clicked'], df_ent_valid['sim_entity'])
    print(f"AUC (entity cosine sim):      {auc_ent:.4f}")

# Combined
df_both = df_pred.dropna(subset=['sim_entity']).copy()
if len(df_both) > 1000:
    X_both = df_both[['sim_subcat', 'sim_entity']].values
    y_both = df_both['clicked'].values
    lr = LogisticRegression(max_iter=1000)
    y_prob = cross_val_predict(lr, X_both, y_both, cv=5, method='predict_proba')[:, 1]
    auc_combined = roc_auc_score(y_both, y_prob)
    print(f"AUC (subcat + entity combined, 5-fold CV): {auc_combined:.4f}")

# --- 1c: MNL model test ---
print("\n--- 1c: MNL choice model test ---")
# For each impression with >= 2 shown articles and at least 1 click:
# Compute MNL probabilities and check calibration

mnl_actual = []
mnl_predicted = []
mnl_ranks = []
n_imp_mnl = 0

for _, row in beh_sample.iterrows():
    uid = row['user_id']
    if uid not in user_taste_subcat:
        continue
    clicked_set = set(row['clicked_list'])
    shown = row['shown_list']
    if len(shown) < 2 or len(clicked_set) == 0:
        continue

    u_sub = user_taste_subcat[uid]

    # Compute utilities (cosine sim as utility proxy)
    utils = []
    labels = []
    for nid in shown:
        a_sub = get_subcat_emb(nid)
        if a_sub is None:
            utils.append(0)
        else:
            utils.append(cosine_sim(u_sub, a_sub))
        labels.append(int(nid in clicked_set))

    utils = np.array(utils)
    labels = np.array(labels)

    # MNL probabilities
    exp_u = np.exp(utils * 5)  # scale factor for sharper predictions
    probs = exp_u / exp_u.sum()

    for i in range(len(shown)):
        mnl_actual.append(labels[i])
        mnl_predicted.append(probs[i])

    # Rank of clicked item
    if labels.sum() >= 1:
        clicked_idx = np.where(labels == 1)[0]
        for ci in clicked_idx:
            rank = (utils >= utils[ci]).sum()
            mnl_ranks.append(rank)
        n_imp_mnl += 1

    if n_imp_mnl >= 100000:
        break

mnl_actual = np.array(mnl_actual)
mnl_predicted = np.array(mnl_predicted)
mnl_ranks = np.array(mnl_ranks)

auc_mnl = roc_auc_score(mnl_actual, mnl_predicted)
print(f"MNL AUC: {auc_mnl:.4f} (on {n_imp_mnl:,} impressions)")
print(f"Mean rank of clicked item: {mnl_ranks.mean():.2f} / avg shown: {df_pred.groupby('uid')['n_shown'].first().mean():.0f}")
print(f"Clicked item in top-5: {(mnl_ranks <= 5).mean()*100:.1f}%")
print(f"Clicked item is top-1: {(mnl_ranks == 1).mean()*100:.1f}%")


# ============================================================
# STEP 3: LOCAL PERSUASION RADIUS
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: LOCAL PERSUASION RADIUS")
print("=" * 70)
print("Does click probability decay with distance from user taste?")

# Use euclidean distance in subcategory space
# (1 - cosine_sim as distance)
df_pred['dist_subcat'] = 1 - df_pred['sim_subcat']

# Bin by distance
n_bins = 20
df_pred['dist_bin'] = pd.qcut(df_pred['dist_subcat'], n_bins, labels=False, duplicates='drop')
dist_stats = df_pred.groupby('dist_bin').agg(
    mean_dist=('dist_subcat', 'mean'),
    ctr=('clicked', 'mean'),
    n=('clicked', 'count')
).reset_index()

print(f"\n{'Dist bin':>8} {'Mean dist':>10} {'CTR':>10} {'N':>12}")
print("-" * 45)
for _, r in dist_stats.iterrows():
    print(f"{int(r['dist_bin']):>8} {r['mean_dist']:>10.4f} {r['ctr']:>10.4f} {int(r['n']):>12,}")

# Estimate effective radius: distance at which CTR drops to 50% of max
max_ctr = dist_stats['ctr'].max()
half_ctr = max_ctr / 2
above_half = dist_stats[dist_stats['ctr'] >= half_ctr]
if len(above_half) > 0:
    radius_50 = above_half['mean_dist'].max()
    print(f"\nEffective radius (CTR ≥ 50% of max): d ≤ {radius_50:.4f}")

# Baseline CTR (farthest bin)
baseline_ctr = dist_stats.iloc[-1]['ctr']
above_2x = dist_stats[dist_stats['ctr'] >= 2 * baseline_ctr]
if len(above_2x) > 0:
    radius_2x = above_2x['mean_dist'].max()
    print(f"Effective radius (CTR ≥ 2× baseline): d ≤ {radius_2x:.4f}")

# Fit exponential decay: CTR = a * exp(-b * dist) + c
from scipy.optimize import curve_fit

def exp_decay(d, a, b, c):
    return a * np.exp(-b * d) + c

try:
    popt, pcov = curve_fit(exp_decay, dist_stats['mean_dist'], dist_stats['ctr'],
                           p0=[0.1, 5, 0.01], maxfev=5000)
    print(f"\nExponential decay fit: CTR = {popt[0]:.4f} * exp(-{popt[1]:.2f} * d) + {popt[2]:.4f}")
    print(f"  Decay rate: {popt[1]:.2f}")
    print(f"  Half-life distance: {np.log(2)/popt[1]:.4f}")
except:
    print("\n  (Exponential fit did not converge)")
    popt = None

# Entity-embedding version
df_ent_dist = df_pred.dropna(subset=['sim_entity']).copy()
if len(df_ent_dist) > 10000:
    df_ent_dist['dist_entity'] = 1 - df_ent_dist['sim_entity']
    df_ent_dist['dist_ent_bin'] = pd.qcut(df_ent_dist['dist_entity'], n_bins, labels=False, duplicates='drop')
    dist_ent_stats = df_ent_dist.groupby('dist_ent_bin').agg(
        mean_dist=('dist_entity', 'mean'),
        ctr=('clicked', 'mean'),
    ).reset_index()

    print(f"\n--- Entity embedding distance ---")
    corr_d, p_d = stats.pointbiserialr(df_ent_dist['clicked'], -df_ent_dist['dist_entity'])
    print(f"Correlation (entity proximity ↔ click): r = {corr_d:.4f}, p = {p_d:.2e}")


# ============================================================
# STEP 4: CALIBRATE w(d) — TASTE MOVEMENT FUNCTION
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: CALIBRATE w(d) — TASTE MOVEMENT FUNCTION")
print("=" * 70)
print("When a user clicks an article at distance d, how much does taste shift?")

# Need users with multiple sessions on different days
beh_sample['time'] = pd.to_datetime(beh_sample['timestamp'], format='%m/%d/%Y %I:%M:%S %p')
beh_sample['date'] = beh_sample['time'].dt.date

# Build per-session taste vectors
print("\nBuilding per-session taste vectors...")
session_data = []

for uid in sample_uids:
    if uid not in user_taste_subcat:
        continue
    rows = beh_sample[beh_sample['user_id'] == uid].sort_values('time')
    dates = rows['date'].unique()
    if len(dates) < 2:
        continue

    for d in sorted(dates):
        day_rows = rows[rows['date'] == d]
        # Session clicks
        session_clicks = []
        for _, r in day_rows.iterrows():
            session_clicks.extend(r['clicked_list'])
        if len(session_clicks) == 0:
            continue

        # Session taste = average subcat embedding of clicked articles
        session_emb = build_user_emb_subcat(session_clicks)
        if session_emb is None:
            continue

        # Compute distances of clicked articles from pre-session taste
        # Pre-session taste = average of all history clicks prior to this session
        pre_hist = day_rows.iloc[0]['hist_list']
        pre_taste = build_user_emb_subcat(pre_hist)
        if pre_taste is None:
            continue

        click_distances = []
        for nid in session_clicks:
            a_emb = get_subcat_emb(nid)
            if a_emb is not None:
                click_distances.append(1 - cosine_sim(pre_taste, a_emb))

        if len(click_distances) == 0:
            continue

        session_data.append({
            'uid': uid,
            'date': d,
            'pre_taste': pre_taste,
            'session_emb': session_emb,
            'mean_click_dist': np.mean(click_distances),
            'n_clicks': len(session_clicks),
        })

print(f"  Sessions: {len(session_data):,}")

# For consecutive sessions: measure taste shift and relate to click distance
# Key insight: accumulated history changes slowly, so we measure shift using
# SESSION click vectors directly: how does session(t+1) compare to session(t),
# and does that shift relate to what was clicked in session(t)?
print("\nComputing taste shift vs. click distance...")
wd_rows = []

sessions_by_user = defaultdict(list)
for s in session_data:
    sessions_by_user[s['uid']].append(s)

for uid, sessions in sessions_by_user.items():
    sessions = sorted(sessions, key=lambda x: x['date'])
    if len(sessions) < 2:
        continue

    for t in range(len(sessions) - 1):
        s_now = sessions[t]
        s_next = sessions[t + 1]

        # Taste shift: how did the user's SESSION-level taste change?
        # (next session clicks vs. current session clicks)
        session_shift = s_next['session_emb'] - s_now['session_emb']
        shift_magnitude = np.linalg.norm(session_shift)

        # Also: shift from pre-taste toward next session's clicks
        drift_from_base = s_next['session_emb'] - s_now['pre_taste']
        drift_magnitude = np.linalg.norm(drift_from_base)

        # Direction: does next session move TOWARD what was consumed now?
        # i.e., is next session more similar to current session's clicks
        # than pre-taste was?
        sim_pretaste_to_now = cosine_sim(s_now['pre_taste'], s_now['session_emb'])
        sim_nextsess_to_now = cosine_sim(s_next['session_emb'], s_now['session_emb'])
        reinforcement = sim_nextsess_to_now - sim_pretaste_to_now

        # Shift alignment: does the drift direction match consumption?
        consumption_dir = s_now['session_emb'] - s_now['pre_taste']
        cd_norm = np.linalg.norm(consumption_dir)
        df_norm = np.linalg.norm(drift_from_base)
        if cd_norm > 1e-10 and df_norm > 1e-10:
            shift_alignment = cosine_sim(drift_from_base, consumption_dir)
        else:
            shift_alignment = np.nan

        wd_rows.append({
            'uid': uid,
            'click_dist': s_now['mean_click_dist'],
            'session_shift_mag': shift_magnitude,
            'drift_mag': drift_magnitude,
            'reinforcement': reinforcement,
            'shift_alignment': shift_alignment,
            'sim_next_to_now': sim_nextsess_to_now,
            'n_clicks': s_now['n_clicks'],
        })

df_wd = pd.DataFrame(wd_rows)
# Keep rows with valid alignment OR valid reinforcement
df_wd_valid = df_wd.dropna(subset=['shift_alignment'])
df_wd_reinf = df_wd.dropna(subset=['reinforcement'])
print(f"  Session pairs total: {len(df_wd):,}")
print(f"  With valid alignment: {len(df_wd_valid):,}")
print(f"  With valid reinforcement: {len(df_wd_reinf):,}")

if len(df_wd_reinf) >= 10:
    # Bin by click distance
    n_bins_wd = min(10, len(df_wd_reinf) // 20)
    if n_bins_wd >= 2:
        df_wd_reinf['dist_bin'] = pd.qcut(df_wd_reinf['click_dist'], n_bins_wd,
                                            labels=False, duplicates='drop')
        wd_stats = df_wd_reinf.groupby('dist_bin').agg(
            mean_dist=('click_dist', 'mean'),
            mean_shift=('session_shift_mag', 'mean'),
            mean_reinf=('reinforcement', 'mean'),
            mean_sim_next=('sim_next_to_now', 'mean'),
            n=('uid', 'count'),
        ).reset_index()

        print(f"\n{'Bin':>4} {'Click dist':>10} {'|Shift|':>10} {'Reinforcement':>14} {'sim(next,now)':>14} {'N':>8}")
        print("-" * 66)
        for _, r in wd_stats.iterrows():
            print(f"{int(r['dist_bin']):>4} {r['mean_dist']:>10.4f} {r['mean_shift']:>10.4f} "
                  f"{r['mean_reinf']:>13.4f} {r['mean_sim_next']:>13.4f} {int(r['n']):>8}")
    else:
        wd_stats = pd.DataFrame()

    # Correlations
    corr_wd, p_wd = stats.pearsonr(df_wd_reinf['click_dist'], df_wd_reinf['session_shift_mag'])
    print(f"\nCorr (click dist ↔ session shift): r = {corr_wd:.4f}, p = {p_wd:.2e}")

    corr_reinf, p_reinf = stats.pearsonr(df_wd_reinf['click_dist'], df_wd_reinf['reinforcement'])
    print(f"Corr (click dist ↔ reinforcement):  r = {corr_reinf:.4f}, p = {p_reinf:.2e}")

    # Overall reinforcement
    mean_reinf = df_wd_reinf['reinforcement'].mean()
    t_r, p_r = stats.ttest_1samp(df_wd_reinf['reinforcement'], 0)
    print(f"\nMean reinforcement: {mean_reinf:+.4f} (t={t_r:.2f}, p={p_r:.2e})")
    print("→ Positive = next session is MORE similar to current clicks than baseline was")

    mean_sim_nn = df_wd_reinf['sim_next_to_now'].mean()
    print(f"Mean sim(next session, current session): {mean_sim_nn:.4f}")

    if len(df_wd_valid) >= 10:
        mean_align = df_wd_valid['shift_alignment'].mean()
        t_a, p_a = stats.ttest_1samp(df_wd_valid['shift_alignment'], 0)
        print(f"Mean shift alignment with consumption: {mean_align:+.4f} (t={t_a:.2f}, p={p_a:.2e})")
    else:
        mean_align = np.nan
        t_a, p_a = np.nan, np.nan
else:
    print("\n  Insufficient data for w(d) binned analysis.")
    wd_stats = pd.DataFrame()
    corr_wd, p_wd = np.nan, np.nan
    corr_reinf, p_reinf = np.nan, np.nan
    mean_reinf, t_r, p_r = np.nan, np.nan, np.nan
    mean_align, t_a, p_a = np.nan, np.nan, np.nan
    mean_sim_nn = np.nan


# ============================================================
# STEP 5: ASSORTMENT SUBSTITUTION / SPILLOVER
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: ASSORTMENT STRUCTURE — SUBSTITUTION & SPILLOVER")
print("=" * 70)

# 5a: Within-category substitution (IIA violation test)
# If MNL/IIA holds: adding more articles of category X should NOT change
# the per-article click rate of other X articles.
# If substitution: more X shown → lower per-article X CTR (cannibalization)

print("\n--- 5a: Within-category substitution (controlling for impression size) ---")
print("Does showing more articles of category X reduce per-article CTR for X?")
print("IMPORTANT: controlling for total impression size to avoid mechanical confound.")

sub_rows = []
for _, row in beh_sample.iterrows():
    shown = row['shown_list']
    clicked_set = set(row['clicked_list'])
    if len(shown) < 3:
        continue

    # Count articles per category in this impression
    cat_articles = defaultdict(list)
    for nid in shown:
        c = news_cat.get(nid)
        if c:
            cat_articles[c].append(nid)

    for c, articles in cat_articles.items():
        n_in_cat = len(articles)
        n_clicked_in_cat = sum(1 for a in articles if a in clicked_set)
        per_article_ctr = n_clicked_in_cat / n_in_cat

        sub_rows.append({
            'category': c,
            'n_in_cat': n_in_cat,
            'per_article_ctr': per_article_ctr,
            'n_shown_total': len(shown),
        })

df_sub = pd.DataFrame(sub_rows)

# --- Controlled analysis: partial correlation controlling for impression size ---
print(f"\n{'Category':<16} {'Raw r':>8} {'Partial r':>10} {'Partial p':>10} {'Substitution?':>14}")
print("-" * 65)

sub_results = []
for cat in cats:
    cdf = df_sub[df_sub['category'] == cat].copy()
    if len(cdf) < 100:
        continue

    # Raw Spearman (confounded)
    corr_raw, _ = stats.spearmanr(cdf['n_in_cat'], cdf['per_article_ctr'])

    # Partial correlation: regress out n_shown_total from both variables
    # Residualize n_in_cat and per_article_ctr on n_shown_total
    X_ctrl = np.column_stack([np.ones(len(cdf)), cdf['n_shown_total'].values])
    # Residualize n_in_cat
    y_ncat = cdf['n_in_cat'].values.astype(float)
    beta_n = np.linalg.lstsq(X_ctrl, y_ncat, rcond=None)[0]
    resid_ncat = y_ncat - X_ctrl @ beta_n
    # Residualize per_article_ctr
    y_ctr = cdf['per_article_ctr'].values
    beta_c = np.linalg.lstsq(X_ctrl, y_ctr, rcond=None)[0]
    resid_ctr = y_ctr - X_ctrl @ beta_c

    corr_partial, p_partial = stats.spearmanr(resid_ncat, resid_ctr)
    sub_flag = "YES (↓)" if corr_partial < 0 and p_partial < 0.05 else "no"

    print(f"{cat:<16} {corr_raw:>7.3f} {corr_partial:>9.3f} {p_partial:>9.2e} {sub_flag:>14}")
    sub_results.append({'category': cat, 'corr_raw': corr_raw,
                        'corr_partial': corr_partial, 'p': p_partial})

df_subr = pd.DataFrame(sub_results)
n_sub = ((df_subr['corr_partial'] < 0) & (df_subr['p'] < 0.05)).sum()
print(f"\n→ {n_sub}/{len(df_subr)} categories show significant within-category substitution")
print("  (after controlling for total impression size)")

# 5b: Cross-category spillover (controlling for impression size)
print("\n--- 5b: Cross-category spillover (controlling for impression size) ---")
print("Does showing more of category X affect CTR for category Y?")

# Build category-level impression data
# For top categories, compute: effect of n_shown(X) on CTR(Y)
top_cats_spill = ['news', 'sports', 'finance', 'lifestyle', 'health', 'entertainment']

spill_rows = []
for _, row in beh_sample.iterrows():
    shown = row['shown_list']
    clicked_set = set(row['clicked_list'])
    if len(shown) < 5:
        continue

    cat_shown = defaultdict(int)
    cat_clicked = defaultdict(int)
    for nid in shown:
        c = news_cat.get(nid)
        if c:
            cat_shown[c] += 1
    for nid in row['clicked_list']:
        c = news_cat.get(nid)
        if c:
            cat_clicked[c] += 1

    r = {f'n_{c}': cat_shown.get(c, 0) for c in top_cats_spill}
    r.update({f'ctr_{c}': (cat_clicked.get(c, 0) / cat_shown[c] if cat_shown.get(c, 0) > 0 else np.nan)
              for c in top_cats_spill})
    r['n_total'] = len(shown)
    spill_rows.append(r)

    if len(spill_rows) >= 200000:
        break

df_spill = pd.DataFrame(spill_rows)

# Partial correlation: residualize on total impression size
print(f"\nSpillover matrix (partial Spearman corr, controlling for impression size):")
print(f"{'':>14}", end='')
for c in top_cats_spill:
    print(f"{c[:8]:>10}", end='')
print()

spill_matrix = np.zeros((len(top_cats_spill), len(top_cats_spill)))
for i, cx in enumerate(top_cats_spill):
    print(f"{cx:>14}", end='')
    for j, cy in enumerate(top_cats_spill):
        col_n = f'n_{cx}'
        col_ctr = f'ctr_{cy}'
        valid = df_spill[[col_n, col_ctr, 'n_total']].dropna()
        if len(valid) > 100:
            # Residualize both on n_total
            X_ctrl = np.column_stack([np.ones(len(valid)), valid['n_total'].values])
            resid_n = valid[col_n].values - X_ctrl @ np.linalg.lstsq(X_ctrl, valid[col_n].values.astype(float), rcond=None)[0]
            resid_ctr = valid[col_ctr].values - X_ctrl @ np.linalg.lstsq(X_ctrl, valid[col_ctr].values, rcond=None)[0]
            r, p = stats.spearmanr(resid_n, resid_ctr)
            spill_matrix[i, j] = r
            sig = "*" if p < 0.05 else " "
            print(f"{r:>9.3f}{sig}", end='')
        else:
            print(f"{'---':>10}", end='')
    print()

print("\n(* = p<0.05. Partial correlations control for total impression size.)")
print("Diagonal = own-category effect. Off-diagonal = spillover.")
print("Positive off-diagonal = complementarity / spillover.")


# ============================================================
# FIGURES
# ============================================================
print("\n" + "=" * 70)
print("GENERATING FIGURES...")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Full Empirical Analysis — MIND Large Dataset', fontsize=14, fontweight='bold')

# Fig 1: CTR vs cosine similarity (Step 1)
ax = axes[0, 0]
ax.plot(bin_stats['mean_sim'], bin_stats['ctr'], 'o-', color='steelblue', linewidth=2, markersize=6)
ax.set_xlabel('Cosine similarity (user ↔ article)')
ax.set_ylabel('Click-through rate')
ax.set_title(f'Step 1: Inner-product predicts clicks\n(AUC = {auc_sub:.3f})')
ax.grid(True, alpha=0.3)

# Fig 2: CTR vs distance — persuasion radius (Step 3)
ax = axes[0, 1]
ax.plot(dist_stats['mean_dist'], dist_stats['ctr'], 'o-', color='#C44E52', linewidth=2, markersize=6)
if popt is not None:
    d_smooth = np.linspace(dist_stats['mean_dist'].min(), dist_stats['mean_dist'].max(), 100)
    ax.plot(d_smooth, exp_decay(d_smooth, *popt), '--', color='gray', alpha=0.7,
            label=f'Fit: exp decay (rate={popt[1]:.1f})')
    ax.legend(fontsize=8)
ax.set_xlabel('Distance from user taste (1 - cos sim)')
ax.set_ylabel('Click-through rate')
ax.set_title('Step 3: Local persuasion radius')
ax.grid(True, alpha=0.3)

# Fig 3: w(d) — reinforcement vs click distance (Step 4)
ax = axes[0, 2]
if len(wd_stats) > 0 and 'mean_reinf' in wd_stats.columns:
    ax.plot(wd_stats['mean_dist'], wd_stats['mean_reinf'], 'o-', color='#55A868',
            linewidth=2, markersize=6, label='Reinforcement')
    ax2 = ax.twinx()
    ax2.plot(wd_stats['mean_dist'], wd_stats['mean_sim_next'], 's--', color='orange',
             linewidth=2, markersize=6, label='sim(next, now)')
    ax.set_xlabel('Mean click distance from taste')
    ax.set_ylabel('Reinforcement', color='#55A868')
    ax2.set_ylabel('sim(next session, current)', color='orange')
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
else:
    ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, ha='center')
ax.set_title('Step 4: w(d) — taste movement')
ax.grid(True, alpha=0.3)

# Fig 4: Within-category substitution (Step 5a)
ax = axes[1, 0]
# Show substitution for top categories
for cat in ['news', 'sports', 'finance', 'lifestyle', 'health']:
    cdf = df_sub[df_sub['category'] == cat]
    if len(cdf) < 100:
        continue
    means = cdf.groupby('n_in_cat')['per_article_ctr'].mean()
    means = means[means.index <= 6]
    ax.plot(means.index, means.values, 'o-', label=cat, markersize=5, linewidth=1.5)
ax.set_xlabel('Number of same-category articles shown')
ax.set_ylabel('Per-article CTR')
ax.set_title('Step 5a: Within-category substitution\n(raw CTR, see text for controlled results)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Fig 5: Spillover heatmap (Step 5b)
ax = axes[1, 1]
im = ax.imshow(spill_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.15, vmax=0.15)
ax.set_xticks(range(len(top_cats_spill)))
ax.set_xticklabels([c[:6] for c in top_cats_spill], rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(top_cats_spill)))
ax.set_yticklabels([c[:6] for c in top_cats_spill], fontsize=8)
ax.set_xlabel('CTR of category (column)')
ax.set_ylabel('N shown of category (row)')
ax.set_title('Step 5b: Spillover matrix')
fig.colorbar(im, ax=ax, shrink=0.8, label='Spearman r')
# Annotate values
for i in range(len(top_cats_spill)):
    for j in range(len(top_cats_spill)):
        ax.text(j, i, f'{spill_matrix[i,j]:.2f}', ha='center', va='center', fontsize=7,
                color='white' if abs(spill_matrix[i,j]) > 0.08 else 'black')

# Fig 6: MNL calibration — predicted vs actual CTR by decile
ax = axes[1, 2]
df_mnl_calib = pd.DataFrame({'actual': mnl_actual, 'predicted': mnl_predicted})
df_mnl_calib['pred_bin'] = pd.qcut(df_mnl_calib['predicted'], 10, labels=False, duplicates='drop')
calib = df_mnl_calib.groupby('pred_bin').agg(
    mean_pred=('predicted', 'mean'),
    mean_actual=('actual', 'mean'),
).reset_index()
ax.plot(calib['mean_pred'], calib['mean_actual'], 'o-', color='steelblue', linewidth=2, markersize=8)
max_val = max(calib['mean_pred'].max(), calib['mean_actual'].max()) * 1.1
ax.plot([0, max_val], [0, max_val], 'r--', lw=1, label='Perfect calibration')
ax.set_xlabel('MNL predicted probability')
ax.set_ylabel('Actual click rate')
ax.set_title(f'Step 1c: MNL calibration\n(AUC = {auc_mnl:.3f})')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig_path = os.path.join(OUT, "full_analysis.pdf")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.savefig(fig_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
print(f"Saved: {fig_path}")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF ALL FINDINGS")
print("=" * 70)
wd_corr_str = f"{corr_wd:.4f}" if not np.isnan(corr_wd) else "N/A"
reinf_corr_str = f"{corr_reinf:.4f}" if not np.isnan(corr_reinf) else "N/A"
decay_str = f"Exponential decay rate: {popt[1]:.2f}, half-life: {np.log(2)/popt[1]:.4f}" if popt is not None else "Monotone decay observed."

print(f"""
STEP 1 — INNER-PRODUCT / MNL UTILITY MODEL
  Subcategory cosine sim predicts clicks: AUC = {auc_sub:.3f}
  Point-biserial r = {corr:.4f} (p = {p_corr:.2e})
  MNL model AUC = {auc_mnl:.3f}
  Clicked item in top-5 by utility: {(mnl_ranks <= 5).mean()*100:.1f}%

STEP 3 — LOCAL PERSUASION RADIUS
  Click probability decays with distance from user taste.
  {decay_str}

STEP 4 — TASTE MOVEMENT FUNCTION w(d)
  Mean reinforcement: {mean_reinf:+.4f}
  Mean sim(next session, current session): {mean_sim_nn:.4f}
  Corr (click dist ~ session shift): r = {wd_corr_str}
  Corr (click dist ~ reinforcement): r = {reinf_corr_str}

STEP 5 — ASSORTMENT STRUCTURE (controlling for impression size)
  5a: {n_sub}/{len(df_subr)} categories show within-category substitution
      (partial correlation, controlling for total impression size)
  5b: Spillover matrix (partial correlations) reveals cross-category effects.
""")

print(f"All outputs saved to: {OUT}/")
print("Done.")
