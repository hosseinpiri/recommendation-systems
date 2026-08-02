# Phase 1-2: Structural calibration of the recommendation-persuasion model on MIND

Deliverable for the Aug 5, 2026 meeting. Takes the group's model (Jiangze May 2026 note,
finite-horizon version in Meichun's `Finite_horizon_recommendation_learning_July2026`)
to the MINDlarge data, then runs the cultivation-blind vs cultivation-aware simulation
the group asked for on July 22.

Model as estimated (taste space = 18-category simplex, x_i = e_cat, m = 1):

- consumption: P(click | z, i) = sigma(alpha_cat(i) + kappa * z[cat(i)])
- transition on click: z <- (1 - rho) z + rho e_cat ; no update otherwise

## Scripts (run in order)

| Script | What it does |
|---|---|
| `p0_prepare.py` | Parses MINDlarge train (Nov 9-14) + dev (Nov 15) into intermediates (scratch dir; set `SCRATCH` env var) |
| `p1a_consumption.py` | ML logit of clicks on category intercepts + taste alignment; kappa, alpha_c, holdout AUC |
| `p1b_transition.py` | Profile likelihood for rho (temporal identification: train clicks build z, dev impressions evaluate); 2-D (rho, rho_ns) test of no-movement-without-consumption; running-mean nonlinearity check |
| `p2_simulation.py` | MIND-calibrated simulation: oracle/learning x cultivation-aware/blind, rho sweep, planner validated against exact DP on a small instance |
| `p3_figures.py` | All memo figures |

## Outputs

`output/p1a_consumption.json`, `output/p1b_transition.json`, `output/p2_simulation.json`,
figures `output/fig_*.pdf`, memo `memo_phase1_2.tex/.pdf`.

## Key identification choices

- z0 comes from the pre-period history field (clicks before Nov 9), so the transition
  estimate is out-of-sample in time: rho moves z with TRAIN-period clicks and is scored
  on DEV-period (next-day) click behavior. A running-average z would be mechanically
  correlated with rho; this design avoids that circularity.
- Impressions provide the take-it-or-leave-it outcome of the model: shown items are
  recommendations, clicks are consumption.
- MIND has no margins; the simulation uses m = 1 (uniform), so cultivation value is
  expressed in expected clicks. Heterogeneous margins are a one-line change.
