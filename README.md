# Persuasive Recommendation and Endogenous Preference Dynamics

Empirical analysis of how recommendation systems shape user preferences over time, using the Microsoft MIND (Microsoft News Dataset) news recommendation dataset.

## Dataset

All analyses use **MINDlarge** (training + development sets), which covers 14 days of user interaction logs (November 9--22, 2019) with approximately 1 million users. The one exception is `taste_drift_analysis.py`, which uses **MINDsmall** (7 days, ~94K users) as a robustness check to test whether taste evolution is detectable in a shorter observation window.

The MIND dataset files are not included in this repository due to their size. Download them from [Microsoft MIND](https://msnews.github.io/) and place them in the appropriate directory. Two BASE paths are used across the scripts and must be configured:

- `mind_full_analysis.py`, `taste_drift_large.py`, `taste_drift_analysis.py`, `deep_behavioral_analysis.py`: set `BASE` to the directory containing `MINDlarge_train/`, `MINDlarge_dev/`, and/or `MINDsmall_dev/`.
- `exposure_effect_full.py`, `exposure_decomposed.py`: set `BASE` to the directory containing the extracted MINDlarge data files directly.

## Analysis Scripts

| Script | Dataset | Description |
|--------|---------|-------------|
| `mind_full_analysis.py` | MINDlarge | Core analysis: click prediction (MNL, AUC = 0.707), inner-product utility estimation, similarity quintile analysis, persuasion radius, exploitation-exploration tradeoff, substitution and spillover effects |
| `taste_drift_large.py` | MINDlarge | Three-period taste shift analysis (P1: Nov 9--11, P2: Nov 12--15, P3: Nov 16--22): self-reinforcement (+243%), entropy change, category transition matrices, controlled OLS regression |
| `taste_drift_analysis.py` | MINDsmall | Robustness check of taste drift using 7-day window; tests whether shorter observation period can detect taste evolution (finding: no clear evidence, supporting the view that taste change is gradual) |
| `exposure_effect_full.py` | MINDlarge | Stacked OLS exposure regressions, cosine alignment of taste shift with recommendation direction, backfire effect (shown-not-clicked pushes taste away), three-way decomposition, dose-response, per-category and magnitude asymmetry analyses |
| `exposure_decomposed.py` | MINDlarge | Decomposed exposure analysis: separates clicked vs. shown-not-clicked recommendation effects |
| `exposure_effect_analysis.py` | MINDlarge | Earlier version of the exposure analysis |
| `deep_behavioral_analysis.py` | MINDlarge | Diversification patterns, satiation effects, position-by-distance interaction, gateway categories, engagement intensity, stickiness taxonomy, Ornstein-Uhlenbeck rebound dynamics |

## Output Directories

- `output/` -- Results from MINDsmall robustness analysis (`taste_drift_analysis.py`)
- `output_large/` -- Results from MINDlarge individual analyses (taste drift, exposure, deep behavioral)
- `output_full/` -- Consolidated results and the full empirical report (`revised_full_empirical_report_expanded.tex/.pdf`)

## Key Findings

- **Taste shift alignment**: recommendation-driven taste change aligns with recommendation direction (cosine = +0.360, z = 515)
- **Click as mechanism**: taste change occurs through consumption (clicking), not mere exposure; shown-not-clicked items push taste away (backfire, d = -0.75)
- **Self-reinforcement**: users who click recommendations see +243% amplification of initial taste tendencies
- **Gradual evolution**: 14-day window shows clear taste shift; 7-day window does not, indicating taste change is gradual
- **Inner-product utility**: MNL click model achieves AUC = 0.707, validating the preference representation

## Requirements

- Python 3.8+
- numpy, pandas, scipy, scikit-learn, matplotlib, statsmodels
