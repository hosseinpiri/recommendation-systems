"""
Deep Behavioral Analysis — MINDlarge
=====================================
Seven analyses beyond the core five-step program:
  1. Polarization vs. Diversification (who narrows vs. broadens)
  2. Satiation Curves (diminishing returns of reinforcement)
  3. Position Bias × Taste Distance interaction
  4. Gateway Categories (directed taste contagion graph)
  5. Engagement Intensity (multi-click sessions and taste shift)
  6. Entry/Exit Asymmetry (traps vs. revolving doors)
  7. Rebound Effect (permanent vs. transient taste shift)

Author: Hossein Piri / Claude
Date: 2026-03-30
"""

import os, sys
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

BASE = "/Users/piri/Desktop/Recommendation Systems/Dataset"
OUT = "/Users/piri/Desktop/Recommendation Systems/code/output_large"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("DEEP BEHAVIORAL ANALYSIS — MINDlarge")
print("=" * 70)

# ============================================================
# DATA LOADING (same fast pipeline as exposure_effect_analysis.py)
# ============================================================
news_cols = ['news_id', 'category', 'subcategory', 'title', 'abstract',
             'url', 'title_entities', 'abstract_entities']
print("\nLoading news metadata...")
dfs_news = []
for split in ['MINDlarge_train', 'MINDlarge_dev', 'MINDlarge_test']:
    path = os.path.join(BASE, split, "news.tsv")
    df = pd.read_csv(path, sep='\t', header=None, names=news_cols, usecols=[0, 1, 2])
    dfs_news.append(df)
news = pd.concat(dfs_news).drop_duplicates(subset='news_id')
news_cat = dict(zip(news['news_id'], news['category']))
news_subcat = dict(zip(news['news_id'], news['subcategory']))
cat_counts = news['category'].value_counts()
categories = sorted(cat_counts[cat_counts >= 50].index.tolist())
cat_to_idx = {c: i for i, c in enumerate(categories)}
n_cats = len(categories)
print(f"  {len(news):,} articles, {n_cats} categories")

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
    if na == 0 or nb == 0: return np.nan
    return np.dot(a, b) / (na * nb)

def entropy(v):
    p = v[v > 0]
    return -np.sum(p * np.log2(p)) if len(p) > 0 else 0.0

def parse_history(s):
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '': return []
    return s.strip().split()

def parse_imp_fast(s):
    if pd.isna(s) or not isinstance(s, str) or s.strip() == '':
        return [], []
    clicked, shown = [], []
    for item in s.strip().split():
        nid, label = item.rsplit('-', 1)
        shown.append(nid)
        if label == '1': clicked.append(nid)
    return clicked, shown

# Load behaviors
beh_cols = ['impression_id', 'user_id', 'timestamp', 'click_history', 'impressions']
print("\nLoading train+dev behaviors...")
beh_train = pd.read_csv(os.path.join(BASE, "MINDlarge_train", "behaviors.tsv"),
                         sep='\t', header=None, names=beh_cols)
beh_dev = pd.read_csv(os.path.join(BASE, "MINDlarge_dev", "behaviors.tsv"),
                       sep='\t', header=None, names=beh_cols)
beh = pd.concat([beh_train, beh_dev], ignore_index=True)
del beh_train, beh_dev
beh['time'] = pd.to_datetime(beh['timestamp'], format='%m/%d/%Y %I:%M:%S %p')
beh['day'] = beh['time'].dt.day
print(f"  {len(beh):,} labeled impressions")

print("  Parsing impressions...")
parsed = beh['impressions'].apply(parse_imp_fast)
beh['clicked_list'] = parsed.apply(lambda x: x[0])
beh['shown_list'] = parsed.apply(lambda x: x[1])
del parsed
beh['hist_list'] = beh['click_history'].apply(parse_history)
beh['n_history'] = beh['hist_list'].apply(len)
beh['n_clicked'] = beh['clicked_list'].apply(len)

# Period assignment
def day_to_period(d):
    if 9 <= d <= 11: return 'P1'
    if 12 <= d <= 15: return 'P2'
    return None
beh['period'] = beh['day'].map(day_to_period)

# Per-user aggregation
print("\nBuilding per-user aggregates...")
user_data = defaultdict(lambda: {'P1_click': [], 'P2_click': [], 'P2_shown': []})
user_hist = {}
uids_arr = beh['user_id'].values
periods_arr = beh['period'].values
clicked_arr = beh['clicked_list'].values
shown_arr = beh['shown_list'].values
hist_arr = beh['hist_list'].values
nhist_arr = beh['n_history'].values

for i in range(len(uids_arr)):
    uid = uids_arr[i]
    p = periods_arr[i]
    if p == 'P1':
        user_data[uid]['P1_click'].extend(clicked_arr[i])
    elif p == 'P2':
        user_data[uid]['P2_click'].extend(clicked_arr[i])
        user_data[uid]['P2_shown'].extend(shown_arr[i])
    if uid not in user_hist or nhist_arr[i] > len(user_hist[uid]):
        user_hist[uid] = hist_arr[i]

print(f"  Users with labeled data: {len(user_data):,}")

# P3 inference from test history growth
print("Loading test behaviors for P3 inference...")
beh_test = pd.read_csv(os.path.join(BASE, "MINDlarge_test", "behaviors.tsv"),
                        sep='\t', header=None, names=beh_cols)
beh_test['hist_list'] = beh_test['click_history'].apply(parse_history)
beh_test['n_history'] = beh_test['hist_list'].apply(len)
idx_max_test = beh_test.groupby('user_id')['n_history'].idxmax()
user_test_hist = beh_test.loc[idx_max_test, ['user_id', 'hist_list']].set_index('user_id')['hist_list'].to_dict()
del beh_test, idx_max_test

user_p3_clicks = {}
for uid in set(user_hist.keys()) & set(user_test_hist.keys()):
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

# Build user vectors
print("Computing user vectors...")
user_vecs = {}
for uid in user_data:
    rec = user_data[uid]
    p1_raw = nids_to_cat_vec(rec['P1_click'])
    p2c_raw = nids_to_cat_vec(rec['P2_click'])
    p2s_raw = nids_to_cat_vec(rec['P2_shown'])
    n_p1, n_p2c, n_p2s = int(p1_raw.sum()), int(p2c_raw.sum()), int(p2s_raw.sum())

    p3_raw = None
    n_p3 = 0
    if uid in user_p3_clicks:
        p3_raw = nids_to_cat_vec(user_p3_clicks[uid])
        n_p3 = int(p3_raw.sum())

    if n_p1 < 2 or n_p2c < 2 or n_p3 < 2:
        continue

    user_vecs[uid] = {
        'p1_vec': normalize(p1_raw), 'p2_click_vec': normalize(p2c_raw),
        'p2_shown_vec': normalize(p2s_raw), 'p3_vec': normalize(p3_raw),
        'p1_raw': p1_raw, 'p2_click_raw': p2c_raw, 'p2_shown_raw': p2s_raw, 'p3_raw': p3_raw,
        'n_p1': n_p1, 'n_p2c': n_p2c, 'n_p2s': n_p2s, 'n_p3': n_p3,
    }

eligible = list(user_vecs.keys())
print(f"Eligible users: {len(eligible):,}")


# ============================================================
# ANALYSIS 1: POLARIZATION vs. DIVERSIFICATION
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: Who Diversifies and Who Narrows?")
print("=" * 70)

ent_data = []
for uid in eligible:
    v = user_vecs[uid]
    e1 = entropy(v['p1_vec'])
    e3 = entropy(v['p3_vec'])
    de = e3 - e1
    p1_max_share = v['p1_vec'].max()
    p1_dom = categories[np.argmax(v['p1_vec'])]
    p2_shown_ent = entropy(v['p2_shown_vec']) if v['n_p2s'] > 0 else np.nan
    total_clicks = v['n_p1'] + v['n_p2c'] + v['n_p3']
    ent_data.append({
        'uid': uid, 'e1': e1, 'e3': e3, 'delta_ent': de,
        'narrows': int(de < 0), 'p1_max_share': p1_max_share,
        'p1_dom': p1_dom, 'p2_shown_ent': p2_shown_ent,
        'total_clicks': total_clicks, 'n_p1': v['n_p1']
    })

df_ent = pd.DataFrame(ent_data)
n_narrow = df_ent['narrows'].sum()
n_total = len(df_ent)
pct_narrow = n_narrow / n_total * 100

print(f"\nUsers who NARROW (entropy decreases): {n_narrow:,} ({pct_narrow:.1f}%)")
print(f"Users who BROADEN (entropy increases):  {n_total - n_narrow:,} ({100 - pct_narrow:.1f}%)")
print(f"\nMean delta entropy: {df_ent['delta_ent'].mean():.4f}")
print(f"Median delta entropy: {df_ent['delta_ent'].median():.4f}")
print(f"Std delta entropy: {df_ent['delta_ent'].std():.4f}")

# Predictors of narrowing
print("\nPredictors of narrowing (logistic-like comparison):")
for var in ['e1', 'p1_max_share', 'total_clicks', 'p2_shown_ent']:
    narrow_mean = df_ent.loc[df_ent['narrows'] == 1, var].mean()
    broad_mean = df_ent.loc[df_ent['narrows'] == 0, var].mean()
    t, p = stats.ttest_ind(df_ent.loc[df_ent['narrows'] == 1, var].dropna(),
                           df_ent.loc[df_ent['narrows'] == 0, var].dropna())
    print(f"  {var:<18s}  Narrowers: {narrow_mean:.4f}  Broadeners: {broad_mean:.4f}  t={t:.1f}  p={p:.2e}")

# By dominant P1 category
print("\nNarrowing rate by P1 dominant category:")
for cat in categories:
    sub = df_ent[df_ent['p1_dom'] == cat]
    if len(sub) > 100:
        rate = sub['narrows'].mean() * 100
        print(f"  {cat:<18s} {rate:5.1f}%  (n={len(sub):,})")


# ============================================================
# ANALYSIS 2: SATIATION CURVES
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: Diminishing Returns of Same-Category Reinforcement")
print("=" * 70)

# Build per-user session-level data from labeled impressions
# Use day as session boundary
print("Building session-level click sequences...")
beh_p1p2 = beh[beh['period'].isin(['P1', 'P2'])].copy()

# For a random sample of users (for speed)
np.random.seed(42)
sample_uids_2 = set(np.random.choice(eligible, min(50000, len(eligible)), replace=False))

session_data = defaultdict(lambda: defaultdict(list))  # uid -> day -> [clicked_cat_list]
for i in range(len(beh_p1p2)):
    uid = beh_p1p2.iloc[i]['user_id']
    if uid not in sample_uids_2:
        continue
    day = beh_p1p2.iloc[i]['day']
    for nid in beh_p1p2.iloc[i]['clicked_list']:
        c = news_cat.get(nid)
        if c and c in cat_to_idx:
            session_data[uid][day].append(c)

# For each user, track cumulative same-category clicks and next-session reinforcement
dosage_reinf = defaultdict(list)  # dosage_k -> list of (did_click_next_session)

for uid in session_data:
    days = sorted(session_data[uid].keys())
    cum_cat_clicks = defaultdict(int)

    for d_idx in range(len(days) - 1):
        today = days[d_idx]
        tomorrow = days[d_idx + 1]
        today_clicks = session_data[uid][today]
        tomorrow_cats = set(session_data[uid][tomorrow])

        for cat in today_clicks:
            k = cum_cat_clicks[cat]  # dosage BEFORE this click
            dosage_reinf[min(k, 10)].append(1 if cat in tomorrow_cats else 0)
            cum_cat_clicks[cat] += 1

print(f"\nReinforcement rate by cumulative same-category dosage:")
print(f"  {'Dosage k':<12s} {'P(next session)':<18s} {'N':>10s}")
print("  " + "-" * 44)
dosage_keys = sorted(dosage_reinf.keys())
dosage_rates = []
for k in dosage_keys:
    vals = dosage_reinf[k]
    rate = np.mean(vals)
    dosage_rates.append(rate)
    label = f"{k}" if k < 10 else "10+"
    print(f"  {label:<12s} {rate:.4f}             {len(vals):>10,}")

# Test for trend
if len(dosage_keys) >= 3:
    rho, p_rho = stats.spearmanr(dosage_keys, dosage_rates)
    print(f"\nSpearman trend (dosage vs reinforcement rate): ρ = {rho:.4f}, p = {p_rho:.4f}")
    if rho > 0:
        print("  → INCREASING returns: more consumption → stronger reinforcement (no satiation)")
    else:
        print("  → DIMINISHING returns: more consumption → weaker reinforcement (satiation)")


# ============================================================
# ANALYSIS 3: POSITION BIAS × TASTE DISTANCE
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: Position Bias × Taste Distance Interaction")
print("=" * 70)

# Sample users for speed
np.random.seed(42)
sample_uids_3 = set(np.random.choice(eligible, min(30000, len(eligible)), replace=False))

pos_data = []
for i in range(len(beh)):
    uid = beh.iloc[i]['user_id']
    if uid not in sample_uids_3 or uid not in user_vecs:
        continue
    shown = beh.iloc[i]['shown_list']
    clicked_set = set(beh.iloc[i]['clicked_list'])
    u_vec = user_vecs[uid]['p1_vec']  # use P1 taste as baseline
    if np.linalg.norm(u_vec) == 0:
        continue

    for pos, nid in enumerate(shown):
        c = news_cat.get(nid)
        if c and c in cat_to_idx:
            a_vec = np.zeros(n_cats)
            a_vec[cat_to_idx[c]] = 1.0
            dist = 1.0 - cosine_sim(u_vec, a_vec)
            if not np.isnan(dist):
                pos_data.append({
                    'position': pos / max(len(shown) - 1, 1),  # normalized [0,1]
                    'distance': dist,
                    'clicked': int(nid in clicked_set)
                })
    if len(pos_data) >= 2000000:
        break

df_pos = pd.DataFrame(pos_data)
print(f"Observations: {len(df_pos):,}")

# Bin by position quintile and distance quintile
df_pos['pos_q'] = pd.qcut(df_pos['position'], 5, labels=False, duplicates='drop')
df_pos['dist_q'] = pd.qcut(df_pos['distance'], 5, labels=False, duplicates='drop')

print("\nCTR by position quintile:")
for q in sorted(df_pos['pos_q'].unique()):
    sub = df_pos[df_pos['pos_q'] == q]
    print(f"  Q{q} (pos={sub['position'].mean():.2f}): CTR = {sub['clicked'].mean():.4f}  N={len(sub):,}")

print("\nCTR heatmap (position × distance):")
pivot = df_pos.groupby(['pos_q', 'dist_q'])['clicked'].mean().unstack()
print(pivot.round(4).to_string())

# Logistic-style regression: click ~ position + distance + position*distance
X_pos = df_pos[['position', 'distance']].values
X_pos_inter = np.column_stack([np.ones(len(X_pos)), X_pos, X_pos[:, 0] * X_pos[:, 1]])
y_pos = df_pos['clicked'].values

# Linear probability model (fast approximation)
beta_pos, _, _, _ = np.linalg.lstsq(X_pos_inter, y_pos, rcond=None)
print(f"\nLinear probability model: click ~ position + distance + position×distance")
for name, b in zip(['Intercept', 'Position', 'Distance', 'Pos × Dist'], beta_pos):
    print(f"  {name:<15s}: {b:.6f}")

if beta_pos[3] > 0:
    print("  → Positive interaction: position bias is LARGER for distant articles.")
    print("    Prominent placement can help compensate for taste mismatch.")
elif beta_pos[3] < 0:
    print("  → Negative interaction: position bias only helps for near-taste articles.")
    print("    The persuasion radius is a hard constraint that placement cannot soften.")


# ============================================================
# ANALYSIS 4: GATEWAY CATEGORIES
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: Gateway Categories — Taste Contagion Graph")
print("=" * 70)

# For each user, find categories they clicked in P3 but NOT in P1 (new entries)
entry_data = defaultdict(lambda: {'entered': 0, 'not_entered': 0, 'p1_cats': defaultdict(float)})

for uid in eligible:
    v = user_vecs[uid]
    for ci in range(n_cats):
        if v['p1_raw'][ci] == 0 and v['p3_raw'][ci] > 0:
            # User entered category ci in P3
            entry_data[ci]['entered'] += 1
            for cj in range(n_cats):
                if cj != ci:
                    entry_data[ci]['p1_cats'][cj] += v['p1_vec'][cj]
        elif v['p1_raw'][ci] == 0 and v['p3_raw'][ci] == 0:
            entry_data[ci]['not_entered'] += 1
            for cj in range(n_cats):
                if cj != ci:
                    entry_data[ci]['p1_cats'][cj] += 0  # counted in denominator only

print(f"\nCategory entry rates (P1=0 → P3>0):")
entry_rates = {}
for ci in range(n_cats):
    e = entry_data[ci]['entered']
    ne = entry_data[ci]['not_entered']
    total = e + ne
    if total > 100:
        rate = e / total
        entry_rates[categories[ci]] = rate
        print(f"  {categories[ci]:<18s} {rate:.4f}  ({e:,} / {total:,})")

# Gateway matrix: for users who entered category C in P3, what was their P1 composition?
print("\nGateway analysis: P1 dominant category of users who ENTERED each new category in P3:")
gateway_matrix = np.zeros((n_cats, n_cats))  # gateway_matrix[source, target] = overrepresentation

for target_ci in range(n_cats):
    entered_uids = []
    baseline_uids = []
    for uid in eligible:
        v = user_vecs[uid]
        if v['p1_raw'][target_ci] == 0:
            if v['p3_raw'][target_ci] > 0:
                entered_uids.append(uid)
            else:
                baseline_uids.append(uid)

    if len(entered_uids) < 50 or len(baseline_uids) < 50:
        continue

    for source_ci in range(n_cats):
        if source_ci == target_ci:
            continue
        entered_mean = np.mean([user_vecs[uid]['p1_vec'][source_ci] for uid in entered_uids])
        baseline_mean = np.mean([user_vecs[uid]['p1_vec'][source_ci] for uid in baseline_uids])
        if baseline_mean > 0:
            gateway_matrix[source_ci, target_ci] = entered_mean / baseline_mean

# Print top gateway pairs
print(f"\nTop gateway pairs (source → target, overrepresentation ratio):")
pairs = []
for si in range(n_cats):
    for ti in range(n_cats):
        if si != ti and gateway_matrix[si, ti] > 1.2:
            pairs.append((categories[si], categories[ti], gateway_matrix[si, ti]))
pairs.sort(key=lambda x: x[2], reverse=True)
for src, tgt, ratio in pairs[:20]:
    print(f"  {src:<18s} → {tgt:<18s}  {ratio:.2f}x")


# ============================================================
# ANALYSIS 5: ENGAGEMENT INTENSITY AND TASTE SHIFT
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 5: Multi-Click Sessions and Taste Shift")
print("=" * 70)

# Helper for category vector from category names
def nids_to_cat_vec_from_cats(cat_list):
    vec = np.zeros(n_cats)
    for c in cat_list:
        if c in cat_to_idx:
            vec[cat_to_idx[c]] += 1
    return vec

# Build session pairs from labeled data
session_pairs = []
for uid in session_data:
    if uid not in user_vecs:
        continue
    days = sorted(session_data[uid].keys())
    for d_idx in range(len(days) - 1):
        today = days[d_idx]
        tomorrow = days[d_idx + 1]
        t_clicks = session_data[uid][today]
        n_clicks = session_data[uid][tomorrow]
        if len(t_clicks) < 1 or len(n_clicks) < 1:
            continue

        t_vec = normalize(nids_to_cat_vec_from_cats(t_clicks))
        n_vec = normalize(nids_to_cat_vec_from_cats(n_clicks))

        sess_ent = entropy(normalize(nids_to_cat_vec_from_cats(t_clicks)))
        shift_mag = np.linalg.norm(n_vec - t_vec)
        sim_next = cosine_sim(t_vec, n_vec)
        if np.isnan(sim_next):
            continue

        session_pairs.append({
            'n_clicks': len(t_clicks),
            'session_entropy': sess_ent,
            'shift_mag': shift_mag,
            'sim_next': sim_next
        })

df_sess = pd.DataFrame(session_pairs)
print(f"Session pairs: {len(df_sess):,}")

# Bin by number of clicks
print(f"\n{'N clicks':<12s} {'Shift mag':>12s} {'cos(next,now)':>15s} {'Session ent':>12s} {'N':>10s}")
print("-" * 65)
for nc in [1, 2, 3, 4, 5]:
    if nc < 5:
        sub = df_sess[df_sess['n_clicks'] == nc]
        label = str(nc)
    else:
        sub = df_sess[df_sess['n_clicks'] >= nc]
        label = f"{nc}+"
    if len(sub) > 0:
        print(f"  {label:<10s} {sub['shift_mag'].mean():12.4f} {sub['sim_next'].mean():15.4f} "
              f"{sub['session_entropy'].mean():12.4f} {len(sub):10,}")

# Correlation: session entropy → shift magnitude
rho_ent, p_ent = stats.spearmanr(df_sess['session_entropy'], df_sess['shift_mag'])
rho_nc, p_nc = stats.spearmanr(df_sess['n_clicks'], df_sess['shift_mag'])
print(f"\nCorrelations with taste shift magnitude:")
print(f"  Session entropy:  ρ = {rho_ent:.4f}, p = {p_ent:.2e}")
print(f"  N clicks:         ρ = {rho_nc:.4f}, p = {p_nc:.2e}")


# ============================================================
# ANALYSIS 6: ENTRY/EXIT ASYMMETRY
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 6: Category Stickiness — Traps vs. Revolving Doors")
print("=" * 70)

cat_entry = {}
cat_exit = {}

for ci in range(n_cats):
    cname = categories[ci]

    # Entry rate: P1 frac = 0 → P3 frac > 0
    n_eligible_entry = 0
    n_entered = 0
    for uid in eligible:
        v = user_vecs[uid]
        if v['p1_raw'][ci] == 0:
            n_eligible_entry += 1
            if v['p3_raw'][ci] > 0:
                n_entered += 1
    entry_rate = n_entered / n_eligible_entry if n_eligible_entry > 100 else np.nan
    cat_entry[cname] = {'rate': entry_rate, 'entered': n_entered, 'eligible': n_eligible_entry}

    # Exit rate: P1 dominant = ci → P3 dominant ≠ ci
    n_eligible_exit = 0
    n_exited = 0
    for uid in eligible:
        v = user_vecs[uid]
        if np.argmax(v['p1_vec']) == ci:
            n_eligible_exit += 1
            if np.argmax(v['p3_vec']) != ci:
                n_exited += 1
    exit_rate = n_exited / n_eligible_exit if n_eligible_exit > 100 else np.nan
    cat_exit[cname] = {'rate': exit_rate, 'exited': n_exited, 'eligible': n_eligible_exit}

print(f"\n{'Category':<18s} {'Entry rate':>12s} {'Exit rate':>12s} {'Retention':>12s} {'Type':<20s}")
print("-" * 78)
entry_exit_data = []
for cname in categories:
    er = cat_entry[cname]['rate']
    xr = cat_exit[cname]['rate']
    if np.isnan(er) or np.isnan(xr):
        continue
    ret = 1 - xr
    if er > np.nanmedian([cat_entry[c]['rate'] for c in categories if not np.isnan(cat_entry[c]['rate'])]):
        if ret > np.nanmedian([1 - cat_exit[c]['rate'] for c in categories if not np.isnan(cat_exit[c]['rate'])]):
            ctype = "TRAP (high/high)"
        else:
            ctype = "Revolving door"
    else:
        if ret > np.nanmedian([1 - cat_exit[c]['rate'] for c in categories if not np.isnan(cat_exit[c]['rate'])]):
            ctype = "Fortress (low/high)"
        else:
            ctype = "Leaky (low/low)"
    print(f"  {cname:<18s} {er:10.4f} {xr:12.4f} {ret:12.4f}   {ctype}")
    entry_exit_data.append({'cat': cname, 'entry': er, 'retention': ret, 'type': ctype})


# ============================================================
# ANALYSIS 7: REBOUND EFFECT
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 7: Rebound Effect — Are Taste Shifts Permanent or Transient?")
print("=" * 70)

overshoot_data = []
for uid in eligible:
    v = user_vecs[uid]
    for ci in range(n_cats):
        p1_f = v['p1_vec'][ci]
        p2_f = v['p2_click_vec'][ci]
        p3_f = v['p3_vec'][ci]
        overshoot = p2_f - p1_f
        if overshoot > 0.01:  # user was "pulled forward" in this category
            rebound = p3_f - p2_f
            momentum = p3_f - p1_f  # net shift from baseline
            overshoot_data.append({
                'overshoot': overshoot, 'rebound': rebound,
                'momentum': momentum, 'p1_f': p1_f, 'ci': ci
            })

df_rebound = pd.DataFrame(overshoot_data)
print(f"\n(User, category) pairs with P2 overshoot > 0.01: {len(df_rebound):,}")

mean_rebound = df_rebound['rebound'].mean()
mean_momentum = df_rebound['momentum'].mean()
t_reb, p_reb = stats.ttest_1samp(df_rebound['rebound'], 0)
t_mom, p_mom = stats.ttest_1samp(df_rebound['momentum'], 0)

print(f"\nMean P3 rebound (P3 - P2): {mean_rebound:+.6f}  (t = {t_reb:.1f}, p = {p_reb:.2e})")
print(f"Mean net momentum (P3 - P1): {mean_momentum:+.6f}  (t = {t_mom:.1f}, p = {p_mom:.2e})")

if mean_rebound < 0 and p_reb < 0.05:
    print("\n  → REVERSION: Users partially revert after P2 overshoot.")
    if mean_momentum > 0:
        print("    But net momentum is still positive — the shift is partially permanent.")
    else:
        print("    And net momentum is negative — the shift is fully transient.")
elif mean_rebound >= 0:
    print("\n  → PERSISTENCE/MOMENTUM: No reversion detected. Taste shifts are self-sustaining.")

# Is rebound proportional to overshoot? (Ornstein-Uhlenbeck test)
rho_ou, p_ou = stats.spearmanr(df_rebound['overshoot'], df_rebound['rebound'])
print(f"\nRebound vs overshoot: ρ = {rho_ou:.4f}, p = {p_ou:.2e}")
if rho_ou < 0:
    print("  → Larger overshoots produce larger reversions (mean-reverting / Ornstein-Uhlenbeck)")
else:
    print("  → No proportional reversion (random walk)")

# Binned analysis
print(f"\n{'Overshoot bin':<18s} {'Mean overshoot':>16s} {'Mean rebound':>14s} {'Mean momentum':>15s} {'N':>10s}")
print("-" * 77)
try:
    df_rebound['ov_bin'] = pd.qcut(df_rebound['overshoot'], 5, labels=False, duplicates='drop')
    for b in sorted(df_rebound['ov_bin'].unique()):
        sub = df_rebound[df_rebound['ov_bin'] == b]
        print(f"  Q{b:<15d} {sub['overshoot'].mean():16.4f} {sub['rebound'].mean():+14.4f} "
              f"{sub['momentum'].mean():+15.4f} {len(sub):10,}")
except Exception:
    pass


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY OF DEEP BEHAVIORAL INSIGHTS")
print("=" * 70)

print(f"""
1. POLARIZATION vs DIVERSIFICATION
   {pct_narrow:.1f}% of users NARROW over 14 days (taste becomes more concentrated).
   Narrowers have {"lower" if df_ent.loc[df_ent['narrows']==1, 'e1'].mean() < df_ent.loc[df_ent['narrows']==0, 'e1'].mean() else "higher"} initial entropy.

2. SATIATION CURVES
   Reinforcement {"increases" if rho > 0 else "decreases"} with cumulative dosage (ρ = {rho:.4f}).

3. POSITION × DISTANCE
   Interaction coefficient = {beta_pos[3]:.6f}.
   {"Placement CAN compensate for distance." if beta_pos[3] > 0 else "Placement CANNOT compensate for distance."}

4. GATEWAY CATEGORIES
   Top gateway pairs identified (see above).

5. ENGAGEMENT INTENSITY
   Session entropy ↔ shift: ρ = {rho_ent:.4f}
   N clicks ↔ shift: ρ = {rho_nc:.4f}

6. ENTRY/EXIT ASYMMETRY
   Categories classified as traps, revolving doors, fortresses, or leaky.

7. REBOUND EFFECT
   Mean rebound = {mean_rebound:+.6f} ({"reversion" if mean_rebound < 0 else "persistence"}).
   Mean net momentum = {mean_momentum:+.6f} ({"partially permanent" if mean_momentum > 0 else "fully transient"}).
   Overshoot ↔ rebound: ρ = {rho_ou:.4f} ({"mean-reverting" if rho_ou < 0 else "random walk"}).
""")


# ============================================================
# FIGURES
# ============================================================
print("Generating figures...")
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Deep Behavioral Analysis — MINDlarge", fontsize=14, fontweight='bold')

# 1. Entropy change distribution
ax = axes[0, 0]
ax.hist(df_ent['delta_ent'], bins=100, color='#2196F3', alpha=0.7, edgecolor='black', linewidth=0.3)
ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='No change')
ax.axvline(df_ent['delta_ent'].mean(), color='orange', linestyle='--', linewidth=1.5, label=f'Mean={df_ent["delta_ent"].mean():.3f}')
ax.set_xlabel('ΔEntropy (P3 - P1)')
ax.set_ylabel('Users')
ax.set_title(f'(1) Diversification vs Narrowing\n{pct_narrow:.1f}% narrow')
ax.legend(fontsize=7)

# 2. Satiation curve
ax = axes[0, 1]
ax.plot(dosage_keys, dosage_rates, 'o-', color='#E91E63', markersize=5)
ax.set_xlabel('Cumulative same-category clicks (k)')
ax.set_ylabel('P(click same cat next session)')
ax.set_title(f'(2) Satiation Curve\nρ = {rho:.3f}')

# 3. Position × Distance heatmap
ax = axes[0, 2]
try:
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto', origin='lower')
    ax.set_xlabel('Distance quintile')
    ax.set_ylabel('Position quintile')
    ax.set_title(f'(3) CTR: Position × Distance\nInteraction = {beta_pos[3]:.4f}')
    plt.colorbar(im, ax=ax, label='CTR')
except Exception:
    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)

# 4. Gateway matrix (top categories)
ax = axes[0, 3]
top_cats_gw = [c for c in ['news', 'sports', 'finance', 'lifestyle', 'health',
                            'entertainment', 'foodanddrink', 'travel'] if c in cat_to_idx]
gw_idx = [cat_to_idx[c] for c in top_cats_gw]
gw_sub = gateway_matrix[np.ix_(gw_idx, gw_idx)]
np.fill_diagonal(gw_sub, 1.0)  # neutral diagonal
im2 = ax.imshow(gw_sub, cmap='RdBu_r', vmin=0.5, vmax=2.0, aspect='auto')
ax.set_xticks(range(len(top_cats_gw)))
ax.set_xticklabels([c[:6] for c in top_cats_gw], rotation=45, fontsize=7)
ax.set_yticks(range(len(top_cats_gw)))
ax.set_yticklabels([c[:6] for c in top_cats_gw], fontsize=7)
ax.set_title('(4) Gateway Matrix\n(source → target, overrep. ratio)')
plt.colorbar(im2, ax=ax, label='Ratio')

# 5. Engagement intensity
ax = axes[1, 0]
nc_bins = df_sess.groupby(df_sess['n_clicks'].clip(upper=6))['shift_mag'].mean()
ax.bar(nc_bins.index, nc_bins.values, color='#FF9800', alpha=0.8, edgecolor='black')
ax.set_xlabel('Clicks per session')
ax.set_ylabel('Mean taste shift magnitude')
ax.set_title(f'(5) Engagement Intensity\nρ(clicks, shift) = {rho_nc:.3f}')

# 6. Entry vs Retention scatter
ax = axes[1, 1]
df_ee = pd.DataFrame(entry_exit_data)
if len(df_ee) > 0:
    colors_ee = {'TRAP (high/high)': '#E91E63', 'Revolving door': '#FF9800',
                 'Fortress (low/high)': '#2196F3', 'Leaky (low/low)': '#9E9E9E'}
    for _, row in df_ee.iterrows():
        ax.scatter(row['entry'], row['retention'], c=colors_ee.get(row['type'], 'gray'),
                   s=80, edgecolors='black', linewidth=0.5, zorder=3)
        ax.annotate(row['cat'][:6], (row['entry'], row['retention']),
                    fontsize=6, ha='center', va='bottom')
    ax.axhline(df_ee['retention'].median(), color='gray', linestyle=':', alpha=0.5)
    ax.axvline(df_ee['entry'].median(), color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Entry rate')
ax.set_ylabel('Retention rate')
ax.set_title('(6) Stickiness: Entry vs Retention')

# 7. Rebound by overshoot quintile
ax = axes[1, 2]
try:
    ov_grouped = df_rebound.groupby('ov_bin').agg({'overshoot': 'mean', 'rebound': 'mean', 'momentum': 'mean'}).reset_index()
    ax.bar(ov_grouped['ov_bin'] - 0.15, ov_grouped['rebound'], 0.3, label='Rebound (P3-P2)', color='#E91E63', alpha=0.8)
    ax.bar(ov_grouped['ov_bin'] + 0.15, ov_grouped['momentum'], 0.3, label='Net shift (P3-P1)', color='#2196F3', alpha=0.8)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Overshoot quintile')
    ax.set_ylabel('Mean shift')
    ax.legend(fontsize=7)
except Exception:
    pass
ax.set_title(f'(7) Rebound Effect\nρ(overshoot, rebound) = {rho_ou:.3f}')

# 8. Summary text panel
ax = axes[1, 3]
ax.axis('off')
summary_text = (
    f"Key Findings:\n\n"
    f"1. {pct_narrow:.0f}% narrow vs {100-pct_narrow:.0f}% broaden\n"
    f"2. Reinforcement {'saturates' if rho < 0 else 'escalates'} (ρ={rho:.3f})\n"
    f"3. Position {'helps' if beta_pos[3] > 0 else 'does not help'} distant articles\n"
    f"4. Gateway paths identified\n"
    f"5. Diverse sessions → {'larger' if rho_ent > 0 else 'smaller'} shifts\n"
    f"6. Categories classified by stickiness\n"
    f"7. Taste shifts {'partially revert' if mean_rebound < 0 else 'persist'}\n"
    f"   (net momentum = {mean_momentum:+.4f})"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "deep_behavioral.png"), dpi=150, bbox_inches='tight')
fig.savefig(os.path.join(OUT, "deep_behavioral.pdf"), bbox_inches='tight')
print(f"Saved: {OUT}/deep_behavioral.png")
print("Done.")
