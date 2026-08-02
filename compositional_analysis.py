"""
Compositional Analysis: Disentangling Simplex Mechanics from Genuine Backfire
==============================================================================
Investigates Points 4+6 from external review:
  - Point 4: OLS β3(shown+NOT clicked) = +0.067 (positive) vs raw mean Δθ = -0.024 (negative)
  - Point 6: Compositional constraint — simplex means clicked gaining → others mechanically lose

Key questions:
  1. How much share do clicked categories gain? What's the mechanical loss imposed on others?
  2. Do shown-not-clicked categories lose MORE than neither-shown-nor-clicked? (genuine backfire)
  3. Why is OLS β3 positive while raw mean Δθ is negative? (Simpson's paradox diagnosis)
  4. Log-ratio approach to escape the simplex constraint

Author: Hossein Piri / Claude
Date: 2026-04-20
"""

import os, json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import functools
print = functools.partial(print, flush=True)

BASE = "/Users/piri/Desktop/Recommendation Systems/Mind-Data-Large"
OUT = "/Users/piri/Desktop/Recommendation Systems/code/output_full"
os.makedirs(OUT, exist_ok=True)

print("=" * 70)
print("COMPOSITIONAL ANALYSIS: Simplex Mechanics vs Genuine Backfire")
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


# ============================================================
# 1. LOAD AND PARSE BEHAVIORS
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

# Per-user aggregates
print("Building per-user aggregates...")
idx_max_hist = beh_labeled.groupby('user_id')['n_history'].idxmax()
user_hist_df = beh_labeled.loc[idx_max_hist, ['user_id', 'hist_list']].set_index('user_id')
user_hist = user_hist_df['hist_list'].to_dict()
del user_hist_df, idx_max_hist

beh_p1p2 = beh_labeled[beh_labeled['period'].isin(['P1', 'P2'])].copy()

print("  Aggregating P1 clicks, P2 clicks, P2 shown...")
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

# Load test set for P3 inference
print("Loading test behaviors...")
beh_test = pd.read_csv(os.path.join(BASE, "MINDlarge_test", "behaviors.tsv"),
                        sep='\t', header=None, names=beh_cols)

beh_test['hist_list'] = beh_test['click_history'].apply(parse_history)
beh_test['n_history'] = beh_test['hist_list'].apply(len)

idx_max_test = beh_test.groupby('user_id')['n_history'].idxmax()
user_test_hist = beh_test.loc[idx_max_test, ['user_id', 'hist_list']].set_index('user_id')['hist_list'].to_dict()
del beh_test, idx_max_test

# P3 new clicks
print("  Inferring P3 clicks from test history growth...")
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
# 2. COMPUTE CATEGORY VECTORS & CLASSIFY CATEGORIES PER USER
# ============================================================
print("\nComputing category vectors and classifying categories per user...")

# For each user, classify each category k into one of three groups:
#   CLICKED: category k appeared in P2 shown AND was clicked at least once
#   SHOWN_NOT_CLICKED: category k appeared in P2 shown but was NEVER clicked
#   NEITHER: category k was NOT shown at all in P2

user_records = []
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

    p1_vec = normalize(p1_raw)
    p3_vec = normalize(p3_raw)
    delta_vec = p3_vec - p1_vec  # Change in taste share per category

    # Classify each category
    # p2_click_raw[k] > 0 means user clicked category k during P2
    # p2_shown_raw[k] > 0 means category k was shown during P2
    clicked_cats = set(np.where(p2_click_raw > 0)[0])
    shown_cats = set(np.where(p2_shown_raw > 0)[0])
    shown_notclicked_cats = shown_cats - clicked_cats
    neither_cats = set(range(n_cats)) - shown_cats

    # Mean delta per group
    if clicked_cats:
        delta_clicked = np.mean([delta_vec[k] for k in clicked_cats])
    else:
        delta_clicked = np.nan

    if shown_notclicked_cats:
        delta_shown_notclicked = np.mean([delta_vec[k] for k in shown_notclicked_cats])
    else:
        delta_shown_notclicked = np.nan

    if neither_cats:
        delta_neither = np.mean([delta_vec[k] for k in neither_cats])
    else:
        delta_neither = np.nan

    # Also compute total share gained by clicked categories
    share_gain_clicked = sum(delta_vec[k] for k in clicked_cats) if clicked_cats else 0.0

    # P1 baseline taste for the shown-not-clicked categories
    if shown_notclicked_cats:
        p1_baseline_snc = np.mean([p1_vec[k] for k in shown_notclicked_cats])
    else:
        p1_baseline_snc = np.nan

    if neither_cats:
        p1_baseline_neither = np.mean([p1_vec[k] for k in neither_cats])
    else:
        p1_baseline_neither = np.nan

    user_records.append({
        'uid': uid,
        'n_clicked_cats': len(clicked_cats),
        'n_shown_notclicked_cats': len(shown_notclicked_cats),
        'n_neither_cats': len(neither_cats),
        'delta_clicked': delta_clicked,
        'delta_shown_notclicked': delta_shown_notclicked,
        'delta_neither': delta_neither,
        'share_gain_clicked': share_gain_clicked,
        'p1_baseline_snc': p1_baseline_snc,
        'p1_baseline_neither': p1_baseline_neither,
    })

df = pd.DataFrame(user_records)
print(f"Eligible users: {n_eligible:,}")
print(f"Records with all three groups: {df.dropna(subset=['delta_clicked','delta_shown_notclicked','delta_neither']).shape[0]:,}")

results = {}

# ============================================================
# 3. TEST 1: RAW MEAN DELTAS BY GROUP
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: Mean Δθ by category group (raw means)")
print("=" * 70)

df_full = df.dropna(subset=['delta_clicked', 'delta_shown_notclicked', 'delta_neither'])

mean_clicked = df_full['delta_clicked'].mean()
mean_snc = df_full['delta_shown_notclicked'].mean()
mean_neither = df_full['delta_neither'].mean()

se_clicked = df_full['delta_clicked'].std() / np.sqrt(len(df_full))
se_snc = df_full['delta_shown_notclicked'].std() / np.sqrt(len(df_full))
se_neither = df_full['delta_neither'].std() / np.sqrt(len(df_full))

print(f"\n  Clicked categories:           Δθ = {mean_clicked:+.6f} (SE={se_clicked:.6f})")
print(f"  Shown-not-clicked categories: Δθ = {mean_snc:+.6f} (SE={se_snc:.6f})")
print(f"  Neither (not shown):          Δθ = {mean_neither:+.6f} (SE={se_neither:.6f})")

# Key comparison: is shown-not-clicked MORE negative than neither?
diff_snc_vs_neither = df_full['delta_shown_notclicked'] - df_full['delta_neither']
t_stat, p_val = stats.ttest_1samp(diff_snc_vs_neither, 0)
mean_diff = diff_snc_vs_neither.mean()
se_diff = diff_snc_vs_neither.std() / np.sqrt(len(diff_snc_vs_neither))

print(f"\n  BACKFIRE TEST: Δ(shown-not-clicked) - Δ(neither) = {mean_diff:+.6f} (SE={se_diff:.6f})")
print(f"    t = {t_stat:.2f}, p = {p_val:.2e}")
print(f"    If negative and significant: genuine backfire beyond compositional mechanics")
print(f"    If ≈ 0: all loss is mechanical (simplex constraint only)")

results['test1_raw_means'] = {
    'n_users': int(len(df_full)),
    'mean_delta_clicked': float(mean_clicked),
    'mean_delta_shown_notclicked': float(mean_snc),
    'mean_delta_neither': float(mean_neither),
    'se_clicked': float(se_clicked),
    'se_snc': float(se_snc),
    'se_neither': float(se_neither),
    'backfire_diff_snc_minus_neither': float(mean_diff),
    'backfire_se': float(se_diff),
    'backfire_t': float(t_stat),
    'backfire_p': float(p_val),
}


# ============================================================
# 4. TEST 2: MECHANICAL DECOMPOSITION
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Mechanical decomposition of the simplex constraint")
print("=" * 70)

# If clicked categories gain G in total share, the remaining (K - n_clicked) categories
# must lose exactly G in total. The UNIFORM mechanical redistribution would be:
# mechanical_loss_per_category = -G / (K - n_clicked)
# Actual loss beyond this = genuine effect

df_full = df_full.copy()
df_full['expected_mechanical_loss'] = -df_full['share_gain_clicked'] / (n_cats - df_full['n_clicked_cats'])

# For shown-not-clicked specifically:
df_full['residual_snc'] = df_full['delta_shown_notclicked'] - df_full['expected_mechanical_loss']
# For neither specifically:
df_full['residual_neither'] = df_full['delta_neither'] - df_full['expected_mechanical_loss']

mean_mech_loss = df_full['expected_mechanical_loss'].mean()
mean_residual_snc = df_full['residual_snc'].mean()
mean_residual_neither = df_full['residual_neither'].mean()

print(f"\n  Mean share gained by clicked cats (total): {df_full['share_gain_clicked'].mean():+.6f}")
print(f"  Mean mechanical loss per non-clicked cat:  {mean_mech_loss:+.6f}")
print(f"  Residual for shown-not-clicked (beyond mechanical): {mean_residual_snc:+.6f}")
print(f"  Residual for neither (beyond mechanical):           {mean_residual_neither:+.6f}")

# Test: is the residual for shown-not-clicked different from zero?
t_res_snc, p_res_snc = stats.ttest_1samp(df_full['residual_snc'].dropna(), 0)
# Test: is the residual for neither different from zero?
t_res_neither, p_res_neither = stats.ttest_1samp(df_full['residual_neither'].dropna(), 0)

print(f"\n  Residual SNC vs 0: t={t_res_snc:.2f}, p={p_res_snc:.2e}")
print(f"  Residual Neither vs 0: t={t_res_neither:.2f}, p={p_res_neither:.2e}")

# More importantly: is residual_snc more negative than residual_neither?
diff_residuals = df_full['residual_snc'] - df_full['residual_neither']
t_res_diff, p_res_diff = stats.ttest_1samp(diff_residuals.dropna(), 0)
mean_res_diff = diff_residuals.mean()

print(f"\n  Residual SNC - Residual Neither = {mean_res_diff:+.6f}")
print(f"  t = {t_res_diff:.2f}, p = {p_res_diff:.2e}")
print(f"  (This is same as Test 1 backfire test — just reconfirms)")

results['test2_mechanical'] = {
    'mean_share_gain_clicked': float(df_full['share_gain_clicked'].mean()),
    'mean_mechanical_loss_per_cat': float(mean_mech_loss),
    'mean_residual_snc': float(mean_residual_snc),
    'mean_residual_neither': float(mean_residual_neither),
    'residual_snc_t': float(t_res_snc),
    'residual_snc_p': float(p_res_snc),
    'residual_neither_t': float(t_res_neither),
    'residual_neither_p': float(p_res_neither),
    'diff_residuals_mean': float(mean_res_diff),
    'diff_residuals_t': float(t_res_diff),
    'diff_residuals_p': float(p_res_diff),
}


# ============================================================
# 5. TEST 3: SIMPSON'S PARADOX DIAGNOSIS
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Simpson's paradox — why OLS β3>0 while raw mean Δθ<0")
print("=" * 70)

# The OLS controls for P1 taste (β1). Shown-not-clicked categories tend to be ones
# the user already had HIGH P1 taste for (that's why they were shown!).
# High P1 → mean reversion → negative Δθ mechanically.
# After controlling for P1, the residual effect of being shown might be positive.

# Evidence: Compare P1 baselines across groups
print(f"\n  Mean P1 taste per category:")
print(f"    Shown-not-clicked categories: {df_full['p1_baseline_snc'].mean():.6f}")
print(f"    Neither categories:           {df_full['p1_baseline_neither'].mean():.6f}")
print(f"    Ratio: {df_full['p1_baseline_snc'].mean() / df_full['p1_baseline_neither'].mean():.2f}x")

# Simpson's paradox test: within P1-taste strata, what happens?
# Create user-category level data for a proper within-analysis
print("\n  Building user×category panel for within-user analysis...")

# Rebuild with all user-category observations
panel_rows = []
for rec in user_records:
    uid = rec['uid']
    if uid not in user_data:
        continue
    ud = user_data[uid]

    p1_raw = nids_to_cat_vec(ud['P1_click'])
    p2_click_raw = nids_to_cat_vec(ud['P2_click'])
    p2_shown_raw = nids_to_cat_vec(ud['P2_shown'])

    n_p1 = int(p1_raw.sum())
    if uid not in user_p3_clicks:
        continue
    p3_raw = nids_to_cat_vec(user_p3_clicks[uid])
    n_p3 = int(p3_raw.sum())

    if n_p1 < 2 or n_p3 < 2:
        continue

    p1_vec = normalize(p1_raw)
    p3_vec = normalize(p3_raw)

    clicked_cats = set(np.where(p2_click_raw > 0)[0])
    shown_cats = set(np.where(p2_shown_raw > 0)[0])
    shown_notclicked_cats = shown_cats - clicked_cats
    neither_cats = set(range(n_cats)) - shown_cats

    for k in range(n_cats):
        if k in clicked_cats:
            group = 'clicked'
        elif k in shown_notclicked_cats:
            group = 'shown_notclicked'
        else:
            group = 'neither'

        panel_rows.append({
            'uid': uid,
            'cat': k,
            'p1_theta': p1_vec[k],
            'p3_theta': p3_vec[k],
            'delta_theta': p3_vec[k] - p1_vec[k],
            'group': group,
        })

panel = pd.DataFrame(panel_rows)
print(f"  Panel size: {len(panel):,} (user × category observations)")

# Mean Δθ by group, unconditional
group_means = panel.groupby('group')['delta_theta'].agg(['mean', 'std', 'count'])
print(f"\n  Unconditional mean Δθ by group (user×category level):")
for grp in ['clicked', 'shown_notclicked', 'neither']:
    row = group_means.loc[grp]
    se = row['std'] / np.sqrt(row['count'])
    print(f"    {grp:20s}: Δθ = {row['mean']:+.6f} (SE={se:.6f}, N={int(row['count']):,})")

# Now condition on P1 taste quintile
panel['p1_quintile'] = pd.qcut(panel['p1_theta'], q=5, labels=False, duplicates='drop')
conditional = panel.groupby(['p1_quintile', 'group'])['delta_theta'].mean().unstack()

print(f"\n  Mean Δθ by P1 taste quintile and group (Simpson's paradox check):")
print(f"  {'Quintile':<10s} {'Clicked':>12s} {'Shown-NC':>12s} {'Neither':>12s} {'SNC-Neither':>12s}")
print("  " + "-" * 60)
for q in sorted(conditional.index):
    c = conditional.loc[q, 'clicked'] if 'clicked' in conditional.columns else np.nan
    s = conditional.loc[q, 'shown_notclicked'] if 'shown_notclicked' in conditional.columns else np.nan
    n = conditional.loc[q, 'neither'] if 'neither' in conditional.columns else np.nan
    diff = s - n if not np.isnan(s) and not np.isnan(n) else np.nan
    print(f"  {q:<10d} {c:+12.6f} {s:+12.6f} {n:+12.6f} {diff:+12.6f}")

# Formal test: OLS with user fixed effects (demeaning)
# Δθ_ik = α_i + β1*p1_ik + β2*I(clicked) + β3*I(shown_nc) + ε
# where neither is the omitted group
print("\n  Running OLS: Δθ_ik = α + β1*p1_ik + β2*I(clicked) + β3*I(shown_nc) + ε")
panel['is_clicked'] = (panel['group'] == 'clicked').astype(float)
panel['is_snc'] = (panel['group'] == 'shown_notclicked').astype(float)

y_panel = panel['delta_theta'].values
X_panel = np.column_stack([
    np.ones(len(panel)),
    panel['p1_theta'].values,
    panel['is_clicked'].values,
    panel['is_snc'].values,
])
cl_panel = panel['uid'].astype('category').cat.codes.values

res_panel = ols_clustered(y_panel, X_panel, cl_panel)
var_names_panel = ['Intercept', 'P1 taste', 'I(clicked)', 'I(shown-not-clicked)']

print(f"\n  N obs = {res_panel['n_obs']:,}   N clusters = {res_panel['n_clusters']:,}   R² = {res_panel['r2']:.4f}")
print(f"\n  {'Variable':<25s} {'Coef':>10s} {'SE(clust)':>12s} {'t':>8s} {'p':>12s}")
print("  " + "-" * 70)
for i, name in enumerate(var_names_panel):
    print(f"  {name:<25s} {res_panel['beta'][i]:10.6f} {res_panel['se_clust'][i]:12.6f} "
          f"{res_panel['t_clust'][i]:8.2f} {res_panel['p_clust'][i]:12.2e}")

print(f"\n  INTERPRETATION:")
print(f"    β(P1) = {res_panel['beta'][1]:.4f} → strong mean reversion (high P1 → negative Δθ)")
print(f"    β(clicked) = {res_panel['beta'][2]:+.4f} → clicked cats gain share BEYOND mean reversion")
print(f"    β(shown-NC) = {res_panel['beta'][3]:+.4f} → shown-not-clicked effect AFTER controlling P1")
if res_panel['beta'][3] > 0:
    print(f"    *** POSITIVE β(shown-NC): No backfire after controlling for mean reversion ***")
    print(f"    The negative raw mean was entirely driven by high P1 → mean reversion")
elif res_panel['beta'][3] < 0 and res_panel['p_clust'][3] < 0.05:
    print(f"    *** NEGATIVE β(shown-NC): Genuine backfire even after controlling P1 ***")
else:
    print(f"    β(shown-NC) not significantly different from zero")

results['test3_simpsons'] = {
    'p1_baseline_snc': float(df_full['p1_baseline_snc'].mean()),
    'p1_baseline_neither': float(df_full['p1_baseline_neither'].mean()),
    'p1_ratio': float(df_full['p1_baseline_snc'].mean() / df_full['p1_baseline_neither'].mean()),
    'panel_n_obs': int(res_panel['n_obs']),
    'panel_n_clusters': int(res_panel['n_clusters']),
    'panel_r2': float(res_panel['r2']),
    'panel_beta': [float(b) for b in res_panel['beta']],
    'panel_se': [float(s) for s in res_panel['se_clust']],
    'panel_t': [float(t) for t in res_panel['t_clust']],
    'panel_p': [float(p) for p in res_panel['p_clust']],
    'panel_var_names': var_names_panel,
    'conditional_means': {str(q): {
        'clicked': float(conditional.loc[q, 'clicked']) if 'clicked' in conditional.columns else None,
        'shown_notclicked': float(conditional.loc[q, 'shown_notclicked']) if 'shown_notclicked' in conditional.columns else None,
        'neither': float(conditional.loc[q, 'neither']) if 'neither' in conditional.columns else None,
    } for q in sorted(conditional.index)},
    'group_means_unconditional': {
        'clicked': float(group_means.loc['clicked', 'mean']),
        'shown_notclicked': float(group_means.loc['shown_notclicked', 'mean']),
        'neither': float(group_means.loc['neither', 'mean']),
    }
}


# ============================================================
# 6. TEST 4: MEAN REVERSION QUANTIFICATION
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Mean reversion magnitude")
print("=" * 70)

# For categories with HIGH P1 baseline, what's the natural mean reversion?
# Use "neither" categories as the clean counterfactual (no treatment)
neither_panel = panel[panel['group'] == 'neither'].copy()
neither_panel['p1_decile'] = pd.qcut(neither_panel['p1_theta'], q=10, labels=False, duplicates='drop')

reversion = neither_panel.groupby('p1_decile').agg(
    mean_p1=('p1_theta', 'mean'),
    mean_delta=('delta_theta', 'mean'),
    n=('delta_theta', 'count')
).reset_index()

print(f"\n  Mean reversion in NEITHER categories (clean counterfactual):")
print(f"  {'Decile':<8s} {'Mean P1':>10s} {'Mean Δθ':>12s} {'N':>10s}")
print("  " + "-" * 44)
for _, row in reversion.iterrows():
    print(f"  {int(row['p1_decile']):<8d} {row['mean_p1']:10.4f} {row['mean_delta']:+12.6f} {int(row['n']):>10,}")

results['test4_mean_reversion'] = {
    'neither_by_decile': reversion.to_dict('records')
}


# ============================================================
# 7. TEST 5: LOG-RATIO ANALYSIS (escaping the simplex)
# ============================================================
print("\n" + "=" * 70)
print("TEST 5: Additive log-ratio (ALR) analysis — escaping simplex constraint")
print("=" * 70)

# ALR: use the last category as reference, compute log(θ_k / θ_ref)
# This maps the simplex to unconstrained R^{K-1} where changes are independent

# Rebuild with ALR
ref_cat = n_cats - 1  # Use last category as reference
epsilon = 1e-8  # Smoothing for zeros

alr_records = []
for rec in user_records[:]:  # all users
    uid = rec['uid']
    if uid not in user_data:
        continue
    ud = user_data[uid]

    p1_raw = nids_to_cat_vec(ud['P1_click'])
    p2_click_raw = nids_to_cat_vec(ud['P2_click'])
    p2_shown_raw = nids_to_cat_vec(ud['P2_shown'])

    if uid not in user_p3_clicks:
        continue
    p3_raw = nids_to_cat_vec(user_p3_clicks[uid])

    n_p1 = int(p1_raw.sum())
    n_p3 = int(p3_raw.sum())
    if n_p1 < 5 or n_p3 < 5:  # Need enough for stable log-ratios
        continue

    # Smooth and normalize
    p1_smooth = (p1_raw + epsilon) / (p1_raw.sum() + n_cats * epsilon)
    p3_smooth = (p3_raw + epsilon) / (p3_raw.sum() + n_cats * epsilon)

    # ALR transform
    alr_p1 = np.log(p1_smooth[:-1] / p1_smooth[ref_cat])
    alr_p3 = np.log(p3_smooth[:-1] / p3_smooth[ref_cat])
    delta_alr = alr_p3 - alr_p1

    clicked_cats = set(np.where(p2_click_raw > 0)[0])
    shown_cats = set(np.where(p2_shown_raw > 0)[0])
    shown_notclicked_cats = shown_cats - clicked_cats
    neither_cats = set(range(n_cats)) - shown_cats

    # Exclude reference category from grouping
    clicked_cats_alr = clicked_cats - {ref_cat}
    snc_cats_alr = shown_notclicked_cats - {ref_cat}
    neither_cats_alr = neither_cats - {ref_cat}

    if clicked_cats_alr:
        alr_records.append({
            'uid': uid, 'group': 'clicked',
            'mean_delta_alr': np.mean([delta_alr[k] for k in clicked_cats_alr])
        })
    if snc_cats_alr:
        alr_records.append({
            'uid': uid, 'group': 'shown_notclicked',
            'mean_delta_alr': np.mean([delta_alr[k] for k in snc_cats_alr])
        })
    if neither_cats_alr:
        alr_records.append({
            'uid': uid, 'group': 'neither',
            'mean_delta_alr': np.mean([delta_alr[k] for k in neither_cats_alr])
        })

alr_df = pd.DataFrame(alr_records)
alr_means = alr_df.groupby('group')['mean_delta_alr'].agg(['mean', 'std', 'count'])

print(f"\n  Mean Δ(ALR) by group (composition-free):")
for grp in ['clicked', 'shown_notclicked', 'neither']:
    if grp in alr_means.index:
        row = alr_means.loc[grp]
        se = row['std'] / np.sqrt(row['count'])
        print(f"    {grp:20s}: Δ(ALR) = {row['mean']:+.4f} (SE={se:.4f}, N={int(row['count']):,})")

# ALR backfire test
alr_pivot = alr_df.pivot_table(index='uid', columns='group', values='mean_delta_alr')
if 'shown_notclicked' in alr_pivot.columns and 'neither' in alr_pivot.columns:
    alr_diff = alr_pivot['shown_notclicked'] - alr_pivot['neither']
    alr_diff_clean = alr_diff.dropna()
    t_alr, p_alr = stats.ttest_1samp(alr_diff_clean, 0)
    print(f"\n  ALR BACKFIRE TEST: Δ(ALR)_SNC - Δ(ALR)_neither = {alr_diff_clean.mean():+.4f}")
    print(f"    t = {t_alr:.2f}, p = {p_alr:.2e}, N = {len(alr_diff_clean):,}")
    if alr_diff_clean.mean() > 0:
        print(f"    *** POSITIVE: Shown-not-clicked categories GAIN relative to neither ***")
        print(f"    *** NO BACKFIRE even in composition-free coordinates ***")
    elif alr_diff_clean.mean() < 0 and p_alr < 0.05:
        print(f"    *** NEGATIVE: Genuine backfire in composition-free coordinates ***")
    else:
        print(f"    Not significantly different from zero")

    results['test5_alr'] = {
        'alr_means': {grp: float(alr_means.loc[grp, 'mean']) for grp in alr_means.index},
        'alr_backfire_diff': float(alr_diff_clean.mean()),
        'alr_backfire_t': float(t_alr),
        'alr_backfire_p': float(p_alr),
        'alr_n': int(len(alr_diff_clean)),
    }


# ============================================================
# 8. TEST 6: REFRAMING — WHAT THE PAPER SHOULD CLAIM
# ============================================================
print("\n" + "=" * 70)
print("TEST 6: Summary — What's happening and what to claim")
print("=" * 70)

print(f"""
  FINDINGS SUMMARY:
  ================

  1. Raw mean Δθ for shown-not-clicked IS negative ({mean_snc:+.6f})
     But raw mean Δθ for NEITHER is also negative ({mean_neither:+.6f})
     → Difference (SNC - Neither) = {mean_diff:+.6f} (t={t_stat:.1f})

  2. Compositional mechanics:
     Mean share gain by clicked cats: {df_full['share_gain_clicked'].mean():+.6f}
     Mechanical loss per non-clicked cat: {mean_mech_loss:+.6f}

  3. Simpson's paradox explanation:
     P1 baseline for SNC categories: {df_full['p1_baseline_snc'].mean():.4f}
     P1 baseline for Neither categories: {df_full['p1_baseline_neither'].mean():.4f}
     → SNC cats have {df_full['p1_baseline_snc'].mean() / df_full['p1_baseline_neither'].mean():.1f}x higher baseline → more mean reversion

  4. After controlling for P1 (OLS):
     β(shown-not-clicked) = {res_panel['beta'][3]:+.4f} (t={res_panel['t_clust'][3]:.1f})
     → {'POSITIVE: no backfire' if res_panel['beta'][3] > 0 else 'NEGATIVE: genuine backfire'}
""")

results['summary'] = {
    'raw_mean_snc_negative': float(mean_snc),
    'raw_mean_neither_negative': float(mean_neither),
    'difference_snc_minus_neither': float(mean_diff),
    'difference_t': float(t_stat),
    'simplex_mechanical_loss': float(mean_mech_loss),
    'p1_baseline_ratio': float(df_full['p1_baseline_snc'].mean() / df_full['p1_baseline_neither'].mean()),
    'ols_beta_snc_after_p1_control': float(res_panel['beta'][3]),
    'ols_t_snc': float(res_panel['t_clust'][3]),
}

# Save results
with open(os.path.join(OUT, "compositional_analysis_results.json"), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {os.path.join(OUT, 'compositional_analysis_results.json')}")
print("\nDone.")
