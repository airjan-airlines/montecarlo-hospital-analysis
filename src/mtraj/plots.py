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


def tail_hist(panels: list[tuple[str, np.ndarray, np.ndarray]], K: int, path: Path) -> Path:
    """Fig 8: cohort ICU bed-days, plug-in vs posterior-predictive, VaR95/99 marked.

    One panel per (title, plug_in draws, posterior-predictive draws).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4), squeeze=False)
    for ax, (title, plug, post) in zip(axes[0], panels):
        bins = np.arange(min(plug.min(), post.min()), max(plug.max(), post.max()) + 5, 4) - 0.5
        for x, name, c in ((plug, "plug-in", SERIES[0]), (post, "posterior-predictive", SERIES[1])):
            ax.hist(x, bins=bins, density=True, histtype="step", color=c, linewidth=1.5, label=name)
            for q, ls in ((0.95, "--"), (0.99, ":")):
                ax.axvline(np.quantile(x, q), color=c, linestyle=ls, linewidth=1.2)
        ax.plot([], [], "--", color=INK2, linewidth=1.2, label="VaR95")
        ax.plot([], [], ":", color=INK2, linewidth=1.2, label="VaR99")
        ax.set(xlabel=f"Total ICU bed-days over 28 days, cohort of K={K}", ylabel="Density", title=title)
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, path)


BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NODE_XY = {0: (0.0, 1.0), 1: (1.0, 1.8), 2: (2.0, 1.0), 3: (0.5, 0.0), 4: (1.5, 0.0)}
SHORT = ["WARD", "ICU", "SUPPORT", "DISCH", "DEAD"]
# label offset direction and alignment per node: side nodes label outward
LABEL_AT = {0: (-1, 0, "right", "center"), 1: (0, 1, "center", "bottom"),
            2: (1, 0, "left", "center"), 3: (0, -1, "center", "top"), 4: (0, -1, "center", "top")}


def state_diagram(P: np.ndarray, path: Path, min_p: float = 0.005) -> Path:
    """Fig 1: states as nodes, edge width proportional to transition probability."""
    from matplotlib.patches import Circle, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(8, 6.5))
    r, rad = 0.17, 0.2
    for i, (x, y) in NODE_XY.items():
        ax.add_patch(Circle((x, y), r, facecolor=STATE_COLOR[STATES[i]], edgecolor=SURFACE,
                            linewidth=2, zorder=3))
        stay = f"\nstay {P[i, i]:.2f}" if i < 3 else ""
        dx, dy, ha, va = LABEL_AT[i]
        ax.text(x + dx * (r + 0.06), y + dy * (r + 0.08), STATES[i] + stay, ha=ha, va=va,
                fontsize=9, color=INK)
    for i in range(3):
        for j in range(5):
            if i == j or P[i, j] < min_p:
                continue
            (x0, y0), (x1, y1) = NODE_XY[i], NODE_XY[j]
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}",
                                         arrowstyle="-|>", mutation_scale=10, shrinkA=r * 72,
                                         shrinkB=r * 72, linewidth=0.5 + 12 * P[i, j],
                                         color=INK2, alpha=0.75, zorder=2))
            # point at u=0.4 on matplotlib's arc3 quadratic Bezier (control point
            # m + rad * (dy, -dx)); slightly toward the source keeps crossing labels apart
            cx = (x0 + x1) / 2 + rad * (y1 - y0)
            cy = (y0 + y1) / 2 - rad * (x1 - x0)
            u = 0.4
            mx = (1 - u) ** 2 * x0 + 2 * u * (1 - u) * cx + u**2 * x1
            my = (1 - u) ** 2 * y0 + 2 * u * (1 - u) * cy + u**2 * y1
            ax.text(mx, my, f"{P[i, j]:.3f}", fontsize=8,
                    ha="center", va="center", color=INK, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none"))
    ax.set(xlim=(-1.0, 3.0), ylim=(-0.5, 2.3), aspect="equal",
           title="Daily transition probabilities (pooled, shrunk)")
    ax.axis("off")
    return _save(fig, path)


def heatmaps(mats: dict[str, np.ndarray], path: Path) -> Path:
    """Fig 2: transient rows of P as annotated heatmaps, one panel per matrix."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    fig, axes = plt.subplots(1, len(mats), figsize=(4.2 * len(mats), 3.2), squeeze=False, sharey=True)
    for ax, (name, P) in zip(axes[0], mats.items()):
        M = P[:3]
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for (i, j), v in np.ndenumerate(M):
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.45 else INK)
        ax.set_xticks(range(5), SHORT, fontsize=8)
        ax.set_yticks(range(3), SHORT[:3], fontsize=8)
        ax.set(xlabel="to state", title=name)
        ax.grid(False)
    axes[0][0].set_ylabel("from state")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="P(next state)")
    return _save(fig, path)


def cv_curves(grid: np.ndarray, curves: dict[str, np.ndarray], alphas: dict[str, float],
              path: Path) -> Path:
    """Fig 4: subject K-fold log-loss vs shrinkage alpha, one panel per model."""
    fig, axes = plt.subplots(1, len(curves), figsize=(4.4 * len(curves), 3.4))
    for ax, (m, c), color in zip(np.atleast_1d(axes), curves.items(), SERIES):
        ax.semilogx(grid, c, "o-", color=color, markersize=4)
        ax.axvline(alphas[m], color=INK2, linestyle="--", linewidth=1)
        ax.text(alphas[m], c.max(), f" α = {alphas[m]:.3g}", color=INK, fontsize=9, va="top")
        ax.set(xlabel="Shrinkage strength α (pseudo-counts per row)", ylabel="CV log-loss per transition",
               title=m)
    fig.tight_layout()
    return _save(fig, path)


def model_comparison(tab: pd.DataFrame, path: Path) -> Path:
    """Fig 5: test log-loss (±1 SE) per model and split, plus paired difference vs mle1.

    Dots, not bars: the differences of interest are ~0.01 on a ~0.9 scale, which a
    zero-based bar would hide.
    """
    models = list(dict.fromkeys(tab.model))
    y = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for k, (split, color) in enumerate((("walk_forward", SERIES[0]), ("kfold", SERIES[1]))):
        d = tab[tab.split == split].set_index("model").loc[models]
        off = (k - 0.5) * 0.25
        axes[0].errorbar(d.log_loss, y + off, xerr=d.log_loss_se, fmt="o", color=color,
                         markersize=6, capsize=3, label=split.replace("_", "-"))
        axes[1].errorbar(d.dlog_loss_vs_mle1, y + off, xerr=d.dlog_loss_vs_mle1_se, fmt="o",
                         color=color, markersize=6, capsize=3)
    axes[0].set_yticks(y, models)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Test log-loss per transition (±1 subject-bootstrap SE)", title="Out-of-sample log-loss")
    axes[0].legend(loc="lower right")
    axes[1].axvline(0, color=INK2, linewidth=1)
    axes[1].set(xlabel="Log-loss minus mle1 (paired, ±1 SE); left is better", title="Paired difference vs first-order MLE")
    fig.tight_layout()
    return _save(fig, path)


def calibration(cal: dict, path: Path) -> Path:
    """Fig 6: observed (with bootstrap band) vs predicted occupancy on the test set."""
    H = len(cal["predicted"]) - 1
    t = np.arange(H + 1)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for s, name in enumerate(STATES):
        c = STATE_COLOR[name]
        ax.fill_between(t, cal["observed_lo"][:, s], cal["observed_hi"][:, s], color=c, alpha=0.2, linewidth=0)
        ax.plot(t, cal["observed"][:, s], color=c, label=name)
        ax.plot(t, cal["predicted"][:, s], color=c, linestyle="--", linewidth=1.5)
    ax.plot([], [], color=INK2, label="observed (95% band)")
    ax.plot([], [], color=INK2, linestyle="--", linewidth=1.5, label="predicted")
    ax.set(xlim=(0, H), ylim=(0, 1), xlabel="Day since hospital admission", ylabel="Fraction of test admissions",
           title=f"Calibration on test set: 28-day mortality {cal['mort28_observed']:.3f} observed vs "
                 f"{cal['mort28_predicted']:.3f} predicted; mean LOS {cal['los_observed']:.2f} vs {cal['los_predicted']:.2f} d")
    ax.title.set_fontsize(10)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return _save(fig, path)
