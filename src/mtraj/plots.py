"""Figures (PNG, 150 dpi) in one consistent style."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import DEAD, DISCHARGED, STATES  # noqa: E402

# Validated categorical order (dataviz reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STATE_COLOR = dict(zip(STATES, SERIES))
METHOD_COLOR = {"plain": SERIES[0], "antithetic_naive": SERIES[1], "antithetic": SERIES[2],
                "control_variate": SERIES[3], "antithetic_cv": SERIES[4]}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2,
    "font.family": "sans-serif", "savefig.dpi": 150, "savefig.bbox": "tight",
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def occupancy(pi: np.ndarray, bands: np.ndarray, path: Path) -> Path:
    """Fig 3: stacked pi_t plus DISCHARGED / DEAD curves with posterior bands."""
    t = np.arange(len(pi))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    axes[0].stackplot(t, pi.T, colors=[STATE_COLOR[s] for s in STATES], labels=STATES,
                      edgecolor=SURFACE, linewidth=1)
    axes[0].set(xlim=(0, t[-1]), ylim=(0, 1), xlabel="Day since hospital admission",
                ylabel="Fraction of cohort", title="State occupancy $\\pi_t$")
    axes[0].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    for ax, s in zip(axes[1:], (DISCHARGED, DEAD)):
        c = STATE_COLOR[STATES[s]]
        ax.fill_between(t, bands[0, :, s], bands[2, :, s], color=c, alpha=0.25, linewidth=0,
                        label="95% posterior band")
        ax.plot(t, pi[:, s], color=c, label="point estimate")
        ax.set(xlim=(0, t[-1]), xlabel="Day since hospital admission",
               ylabel=f"P({STATES[s]} by day t)", title=STATES[s].capitalize())
        ax.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, path)


def se_vs_n(table: pd.DataFrame, path: Path) -> Path:
    """Fig 7: log-log SE vs n per estimator, with a slope -1/2 reference."""
    targets = {"y1": "Y1 = P(ICU days in 28 > 7)", "y2": "Y2 = P(LOS > 14)", "mort": "28-day mortality"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    for ax, (t, title) in zip(axes, targets.items()):
        d = table[table.target == t]
        for m, g in d.groupby("method", sort=False):
            ax.loglog(g.n, g.se, "o-", color=METHOD_COLOR[m], markersize=5, label=m.replace("_", " "))
        p = d[d.method == "plain"].sort_values("n")
        ax.loglog(p.n, p.se.iloc[0] * np.sqrt(p.n.iloc[0] / p.n), "--", color=INK2, linewidth=1,
                  label="slope −1/2")
        ax.set(xlabel="Paths per replication n", ylabel="Standard error", title=title)
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return _save(fig, path)


def tail_hist(plug: np.ndarray, post: np.ndarray, rm: pd.DataFrame, K: int, path: Path) -> Path:
    """Fig 8: cohort ICU bed-days, plug-in vs posterior-predictive, VaR95/99 marked."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.arange(min(plug.min(), post.min()), max(plug.max(), post.max()) + 5, 4) - 0.5
    for x, name, c in ((plug, "plug_in", SERIES[0]), (post, "posterior_predictive", SERIES[1])):
        ax.hist(x, bins=bins, density=True, histtype="step", color=c, linewidth=1.5,
                label=name.replace("_", "-"))
        for q, ls in (("VaR95", "--"), ("VaR99", ":")):
            ax.axvline(rm.loc[q, name], color=c, linestyle=ls, linewidth=1.2)
    ax.plot([], [], "--", color=INK2, linewidth=1.2, label="VaR95")
    ax.plot([], [], ":", color=INK2, linewidth=1.2, label="VaR99")
    ax.set(xlabel=f"Total ICU bed-days over 28 days, cohort of K={K}", ylabel="Density",
           title="Cohort ICU bed-day tail risk")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)
