"""Figures 2 and 3 of the paper from cvar_policy_results.npz.

Run from the repository root, after experiment_cvar_policy.py:
    python experiments/plot_cvar_policy.py
"""

import shutil

import matplotlib.pyplot as plt
import numpy as np

USE_TEX = shutil.which("latex") is not None or shutil.which("pdflatex") is not None
TRUTH_COLOR = "#000000"
STYLES = {
    "ccme_dr": ("#0072B2", "CCME-DR (ours)", "-"),
    "one_step": ("#E69F00", "One-Step (CME/CoDiTE)", "--"),
    "cpme": ("#009E73", "CPME", "-."),
    "cate_dr": ("#D55E00", "CATE-DR", ":"),
}


def set_style():
    """Configure matplotlib rcParams for the figures of the paper."""
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "text.usetex": USE_TEX,
            "font.family": "serif",
            "font.serif": ["cm", "Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "lines.linewidth": 2.0,
            "axes.linewidth": 1.1,
            "grid.linewidth": 0.7,
            "xtick.major.width": 1.1,
            "ytick.major.width": 1.1,
        }
    )


def clean_axes(ax):
    """Add a light grid to the axes and remove the top and right spines.

    Args:
        ax: Matplotlib axes.
    """
    ax.grid(True, alpha=0.18, linestyle="--", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_band(ax, x, values, method, linewidth):
    """Plot the median and the 10th to 90th percentiles of the values against x.

    Args:
        ax: Matplotlib axes.
        x: Points on the horizontal axis, shape ``(L,)``.
        values: Values of shape ``(R, L)``, one row per replication.
        method: Key of STYLES that gives the color, the label and the line style.
        linewidth: Width of the median line.
    """
    color, label, linestyle = STYLES[method]
    ax.plot(
        x,
        np.median(values, axis=0),
        color=color,
        label=label,
        ls=linestyle,
        lw=linewidth,
        zorder=5,
    )
    ax.fill_between(
        x,
        np.percentile(values, 10, axis=0),
        np.percentile(values, 90, axis=0),
        color=color,
        alpha=0.15,
        lw=0,
        zorder=4,
    )


def save(fig, name):
    """Save a figure as a PDF file.

    Args:
        fig: Matplotlib figure.
        name: File name without the ``.pdf`` extension.
    """
    fig.savefig(f"{name}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {name}.pdf")


def standard_error(values):
    """Compute the standard error of the mean of the values.

    Args:
        values: Values of shape ``(R,)``.

    Returns:
        The standard deviation (with ``ddof=0``) divided by sqrt(R), as a float.
    """
    return values.std() / np.sqrt(values.size)


def plot_estimation(results):
    """Draw Figure 2 and save it as ``fig_benchmark_est.pdf``.

    Args:
        results: Results loaded from ``cvar_policy_results.npz``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    ax = axes[0]
    v_grid = results["v_grid"]
    ax.plot(
        v_grid,
        results["true_curve"],
        color=TRUTH_COLOR,
        lw=2.8,
        label="ground truth",
        zorder=7,
    )
    for method in ["ccme_dr", "one_step", "cpme"]:
        plot_band(ax, v_grid, results[f"curve_{method}"], method, 1.9)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("$v$")
    ax.set_ylabel(r"CVaR$_{%g}$-TE$(v)$" % float(results["curve_alpha"]))
    ax.set_title(
        r"\textbf{(a)} tail-risk effect recovery"
        if USE_TEX
        else "(a) tail-risk effect recovery"
    )
    clean_axes(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=14)

    ax = axes[1]
    methods = ["ccme_dr", "one_step", "cpme"]
    positions = np.arange(len(methods))
    width = 0.36
    colors = [STYLES[method][0] for method in methods]
    cvar_mae = [results[f"cvar_mae_{method}"] for method in methods]
    cate_mae = [results[f"cate_mae_{method}"] for method in methods]
    ax.bar(
        positions - width / 2,
        [values.mean() for values in cvar_mae],
        width,
        yerr=[standard_error(values) for values in cvar_mae],
        capsize=3,
        color=colors,
        edgecolor="k",
        lw=0.6,
        label="CVaR-TE MAE",
    )
    ax.bar(
        positions + width / 2,
        [values.mean() for values in cate_mae],
        width,
        yerr=[standard_error(values) for values in cate_mae],
        capsize=3,
        alpha=0.55,
        color=colors,
        edgecolor="k",
        lw=0.6,
        hatch="//",
        label="CATE MAE",
    )
    cate_dr_mae = results["cate_mae_cate_dr"]
    ax.bar(
        len(methods),
        cate_dr_mae.mean(),
        width,
        yerr=standard_error(cate_dr_mae),
        capsize=3,
        alpha=0.55,
        color=STYLES["cate_dr"][0],
        edgecolor="k",
        lw=0.6,
        hatch="//",
    )
    labels = ["CCME-DR", "One-Step\n(CME/CoDiTE)", "CPME", "CATE-DR"]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel("mean absolute error")
    ax.set_title(
        r"\textbf{(b)} estimation error" if USE_TEX else "(b) estimation error"
    )
    clean_axes(ax)
    ax.legend(frameon=False, fontsize=14, loc="upper left")

    fig.tight_layout()
    save(fig, "fig_benchmark_est")


def plot_policy(results):
    """Draw Figure 3 and save it as ``fig_benchmark_policy.pdf``.

    Args:
        results: Results loaded from ``cvar_policy_results.npz``.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.6))
    for method in ["ccme_dr", "one_step", "cpme", "cate_dr"]:
        plot_band(ax, results["alphas"], results[f"regret_{method}"], method, 2.1)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.invert_xaxis()
    ax.set_xlabel(r"tail level $\alpha$   (more risk-averse $\rightarrow$)")
    ax.set_ylabel("policy regret vs. oracle")
    ax.set_title("risk-sensitive policy regret")
    clean_axes(ax)
    ax.legend(loc="center left", frameon=False, fontsize=14)
    fig.tight_layout()
    save(fig, "fig_benchmark_policy")


def main():
    """Draw Figures 2 and 3 from ``cvar_policy_results.npz``."""
    set_style()
    results = np.load("cvar_policy_results.npz")
    plot_estimation(results)
    plot_policy(results)


if __name__ == "__main__":
    main()
