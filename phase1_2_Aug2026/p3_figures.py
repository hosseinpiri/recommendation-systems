"""Figures for the Phase 1-2 memo. Static matplotlib, CVD-safe palette
(Okabe-Ito subset validated with the dataviz six-checks script; the flagged
7.6-dE pair is relieved by direct labels + distinct line styles/markers)."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")

BLUE, ORANGE, GREEN, PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
INK, MUTED = "#1a1a1a", "#666666"

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.dpi": 150,
})


def fig_profile():
    with open(os.path.join(OUT, "p1b_transition.json")) as f:
        d = json.load(f)
    rho = np.array(d["rho_grid"])
    ll = np.array(d["profile_loglik"])
    ll = ll - ll.max()
    rh = d["rho_hat"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.9))
    ax.plot(rho, ll, color=BLUE, lw=2, marker="o", ms=3.5)
    ax.axvline(rh, color=MUTED, lw=1, ls=":")
    ax.annotate(rf"$\hat\rho={rh:.3f}$", (rh, ll.min() * 0.2), fontsize=9,
                color=INK, xytext=(rh + 0.04, ll.min() * 0.2))
    ax.set_xlabel(r"taste-update step $\rho$")
    ax.set_ylabel("profile log likelihood (rel. max)")
    ax.set_title("full grid", fontsize=9, color=MUTED)

    m = rho <= 0.0801
    ax2.plot(rho[m], ll[m], color=BLUE, lw=2, marker="o", ms=3.5)
    ax2.axhline(-1.92, color=MUTED, lw=1, ls="--")
    ax2.text(0.0805, -1.92, "95% LR cutoff", va="bottom", ha="right",
             fontsize=8, color=MUTED)
    ax2.axvline(rh, color=MUTED, lw=1, ls=":")
    ax2.set_xlabel(r"taste-update step $\rho$")
    ax2.set_ylim(-45, 3)
    ax2.set_title("near the peak", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_profile_rho.pdf"))
    plt.close(fig)


def fig_alpha():
    with open(os.path.join(OUT, "p1a_consumption.json")) as f:
        d = json.load(f)
    items = sorted(d["alpha_by_category"].items(), key=lambda kv: kv[1]["alpha"])
    names = [k for k, _ in items]
    a = np.array([v["alpha"] for _, v in items])
    se = np.array([v["se"] for _, v in items])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    y = np.arange(len(names))
    ax.errorbar(a, y, xerr=1.96 * se, fmt="o", color=BLUE, ms=5,
                ecolor=MUTED, elinewidth=1.2, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(r"item intercept $\hat\alpha_c$ (95% CI)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_alpha.pdf"))
    plt.close(fig)


def fig_sim():
    with open(os.path.join(OUT, "p2_simulation.json")) as f:
        d = json.load(f)
    res = d["results"]
    rhos = sorted(float(r) for r in res.keys())
    pols = ["oracle-aware", "learn-aware", "oracle-blind", "learn-blind"]
    colors = {"oracle-aware": BLUE, "learn-aware": GREEN,
              "oracle-blind": ORANGE, "learn-blind": PINK}
    styles = {"oracle-aware": "-", "learn-aware": "--",
              "oracle-blind": "-", "learn-blind": "--"}
    marks = {"oracle-aware": "o", "learn-aware": "s",
             "oracle-blind": "^", "learn-blind": "D"}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for p in pols:
        m = [res[str(r)][p]["mean_clicks"] for r in rhos]
        s = [res[str(r)][p]["se_over_seeds"] for r in rhos]
        ax.errorbar(rhos, m, yerr=[1.96 * x for x in s], color=colors[p],
                    ls=styles[p], marker=marks[p], ms=5, lw=1.8, capsize=2)
        ax.annotate(p, (rhos[-1], m[-1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=8.5,
                    color=INK, va="center")
    ax.axvline(d["config"]["rho_hat"], color=MUTED, lw=1, ls=":")
    ax.text(d["config"]["rho_hat"], ax.get_ylim()[0], r" $\hat\rho$",
            fontsize=9, color=MUTED, va="bottom")
    ax.set_xlabel(r"taste-update step $\rho$")
    ax.set_ylabel(f"expected clicks per user (H = {d['config']['H']})")
    ax.set_xlim(-0.01, max(rhos) * 1.30)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sim_clicks.pdf"))
    plt.close(fig)


def fig_rev():
    with open(os.path.join(OUT, "p2_simulation_hetero.json")) as f:
        d = json.load(f)
    res = d["results"]
    rhos = sorted(float(r) for r in res.keys())
    pols = ["oracle-aware", "learn-aware", "oracle-blind", "learn-blind"]
    colors = {"oracle-aware": BLUE, "learn-aware": GREEN,
              "oracle-blind": ORANGE, "learn-blind": PINK}
    styles = {"oracle-aware": "-", "learn-aware": "--",
              "oracle-blind": "-", "learn-blind": "--"}
    marks = {"oracle-aware": "o", "learn-aware": "s",
             "oracle-blind": "^", "learn-blind": "D"}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for p in pols:
        m = [res[str(r)][p]["mean_revenue"] for r in rhos]
        s = [res[str(r)][p]["se_over_seeds"] for r in rhos]
        ax.errorbar(rhos, m, yerr=[1.96 * x for x in s], color=colors[p],
                    ls=styles[p], marker=marks[p], ms=5, lw=1.8, capsize=2)
        ax.annotate(p, (rhos[-1], m[-1]), xytext=(5, 0),
                    textcoords="offset points", fontsize=8.5,
                    color=INK, va="center")
    ax.axvline(d["config"]["rho_hat"], color=MUTED, lw=1, ls=":")
    ax.text(d["config"]["rho_hat"], ax.get_ylim()[0], r" $\hat\rho$",
            fontsize=9, color=MUTED, va="bottom")
    ax.set_xlabel(r"taste-update step $\rho$")
    ax.set_ylabel(f"expected revenue per user (H = {d['config']['H']})")
    ax.set_xlim(-0.01, max(rhos) * 1.32)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sim_rev.pdf"))
    plt.close(fig)


def fig_gap():
    with open(os.path.join(OUT, "p2_simulation.json")) as f:
        d = json.load(f)
    res = d["results"]
    rhos = sorted(float(r) for r in res.keys())
    gap_or = [res[str(r)]["oracle-aware"]["mean_clicks"]
              - res[str(r)]["oracle-blind"]["mean_clicks"] for r in rhos]
    gap_ln = [res[str(r)]["learn-aware"]["mean_clicks"]
              - res[str(r)]["learn-blind"]["mean_clicks"] for r in rhos]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(rhos, gap_or, color=BLUE, marker="o", ms=5, lw=1.8)
    ax.plot(rhos, gap_ln, color=GREEN, ls="--", marker="s", ms=5, lw=1.8)
    ax.annotate("oracle", (rhos[-1], gap_or[-1]), xytext=(5, 0),
                textcoords="offset points", fontsize=8.5, va="center", color=INK)
    ax.annotate("learning", (rhos[-1], gap_ln[-1]), xytext=(5, 0),
                textcoords="offset points", fontsize=8.5, va="center", color=INK)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(d["config"]["rho_hat"], color=MUTED, lw=1, ls=":")
    ax.set_xlabel(r"taste-update step $\rho$")
    ax.set_ylabel("cultivation value (clicks per user)")
    ax.set_xlim(-0.01, max(rhos) * 1.30)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_sim_gap.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_profile()
    fig_alpha()
    for fn in (fig_sim, fig_gap, fig_rev):
        try:
            fn()
        except FileNotFoundError:
            print(f"{fn.__name__}: json not present yet; skipped")
    print("figures written to", OUT)
