"""Density comparison plots.

Loads per-run density CSVs produced by :mod:`demo` and plots median
estimates with 50% and 90% confidence bands for DR vs One-Step modes
across all three methods (Ridge, Deep Feature, Neural-Kernel) in a
2x3 grid.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import conditional_density

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": "cm",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 2.0,
        "axes.linewidth": 1.2,
        "grid.linewidth": 0.8,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    }
)


def load_and_process_data(method, mode):
    """Load a density CSV and compute percentile statistics.

    Reads ``density_results_{method}_{mode}.csv`` and computes the
    median, 5th/25th/75th/95th percentiles across runs for both the
    "high" and "low" covariate profiles.

    Args:
        method: Estimation method name (``"ridge"``, ``"df"``, or ``"nk"``).
        mode: Estimation mode (``"dr"`` or ``"onestep"``).

    Returns:
        Dict with keys ``Y_grid``, ``{high,low}_{median,p05,p25,p75,p95}``,
        or ``None`` if the CSV is not found.
    """
    csv_path = f"density_results_{method}_{mode}.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found. Skipping...")
        return None

    Y_grid = df["Y_grid"].values

    high_cols = [col for col in df.columns if "high" in col]
    low_cols = [col for col in df.columns if "low" in col]

    pdfs_high = df[high_cols].values
    pdfs_low = df[low_cols].values

    stats = {
        "Y_grid": Y_grid,
        "high_median": np.median(pdfs_high, axis=1),
        "high_p05": np.percentile(pdfs_high, 5, axis=1),
        "high_p25": np.percentile(pdfs_high, 25, axis=1),
        "high_p75": np.percentile(pdfs_high, 75, axis=1),
        "high_p95": np.percentile(pdfs_high, 95, axis=1),
        "low_median": np.median(pdfs_low, axis=1),
        "low_p05": np.percentile(pdfs_low, 5, axis=1),
        "low_p25": np.percentile(pdfs_low, 25, axis=1),
        "low_p75": np.percentile(pdfs_low, 75, axis=1),
        "low_p95": np.percentile(pdfs_low, 95, axis=1),
    }

    return stats


def plot_density_comparison(
    ax,
    Y_grid,
    pdf_true,
    stats_dr,
    stats_onestep,
    density_type,
    method_name,
    row_label="",
):
    """Plot density estimates with confidence bands for DR and One-Step.

    Draws the ground-truth density, the median estimate, and shaded
    50%/90% confidence bands for each mode on a single axes.

    Args:
        ax: Matplotlib ``Axes`` to draw on.
        Y_grid: Outcome grid of shape ``(n_grid,)``.
        pdf_true: True density values of shape ``(n_grid,)``.
        stats_dr: Statistics dict for the DR mode (from
            :func:`load_and_process_data`), or ``None``.
        stats_onestep: Statistics dict for the One-Step mode, or ``None``.
        density_type: ``"high"`` or ``"low"`` (selects which percentile
            keys to plot).
        method_name: Display name for the method (used for the column
            title).
        row_label: Optional LaTeX label drawn to the left of the axes.
    """

    color_dr = "#2E86AB"  # blue
    color_onestep = "#E63946"  # red
    color_truth = "#06070E"  # black

    ax.plot(
        Y_grid,
        pdf_true,
        color=color_truth,
        label="Ground Truth",
        linewidth=2.5,
        zorder=10,
        linestyle="-",
    )

    if stats_dr is not None:
        median_key = f"{density_type}_median"
        p05_key = f"{density_type}_p05"
        p25_key = f"{density_type}_p25"
        p75_key = f"{density_type}_p75"
        p95_key = f"{density_type}_p95"

        ax.plot(
            Y_grid,
            stats_dr[median_key],
            color=color_dr,
            label="DR",
            linewidth=2.0,
            zorder=6,
            linestyle="-",
        )

        ax.fill_between(
            Y_grid,
            stats_dr[p25_key],
            stats_dr[p75_key],
            color=color_dr,
            alpha=0.35,
            label="DR 50% CI",
            edgecolor="none",
            zorder=4,
        )
        ax.fill_between(
            Y_grid,
            stats_dr[p05_key],
            stats_dr[p95_key],
            color=color_dr,
            alpha=0.15,
            label="DR 90% CI",
            edgecolor="none",
            zorder=3,
        )

    if stats_onestep is not None:
        median_key = f"{density_type}_median"
        p05_key = f"{density_type}_p05"
        p25_key = f"{density_type}_p25"
        p75_key = f"{density_type}_p75"
        p95_key = f"{density_type}_p95"

        ax.plot(
            Y_grid,
            stats_onestep[median_key],
            color=color_onestep,
            label="One-Step",
            linewidth=2.0,
            zorder=5,
            linestyle="-",
        )

        ax.fill_between(
            Y_grid,
            stats_onestep[p25_key],
            stats_onestep[p75_key],
            color=color_onestep,
            alpha=0.35,
            label="One-Step 50% CI",
            edgecolor="none",
            zorder=3,
        )
        ax.fill_between(
            Y_grid,
            stats_onestep[p05_key],
            stats_onestep[p95_key],
            color=color_onestep,
            alpha=0.15,
            label="One-Step 90% CI",
            edgecolor="none",
            zorder=2,
        )

    ax.grid(True, alpha=0.2, linestyle="--", linewidth=0.5, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    if row_label:
        ax.text(
            -0.15,
            0.5,
            row_label,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="center",
            ha="right",
            rotation=90,
        )


def visualize_all_methods():
    """Create a 2x3 density comparison figure for all methods.

    Loads density CSVs for Ridge, Deep Feature, and Neural-Kernel
    methods in both DR and One-Step modes, computes the true
    counterfactual densities for the "high" and "low" covariate
    profiles, and produces a figure saved as
    ``fig_densities.pdf`` and ``fig_densities.png``.
    """

    methods = ["ridge", "df", "nk"]
    modes = ["dr", "onestep"]
    method_names = {
        "ridge": "Ridge Regression",
        "df": "Deep Feature",
        "nk": "Neural-Kernel",
    }

    data = {}
    for method in methods:
        data[method] = {}
        for mode in modes:
            data[method][mode] = load_and_process_data(method, mode)

    Y_grid = None
    for method in methods:
        for mode in modes:
            if data[method][mode] is not None:
                Y_grid = data[method][mode]["Y_grid"]
                break
        if Y_grid is not None:
            break

    if Y_grid is None:
        print("Error: No data files found!")
        return

    pdf_true_high = conditional_density(
        np.array([2.2, -0.2, 2.2, -0.2, 2.2]),
        Y_grid,
        baseline=1.0,
        baseline_effect=2.0,
        mixture_prop=0.3,
        n_covariates=10,
    )
    pdf_true_low = conditional_density(
        np.array([-0.2, 2.2, -0.2, 2.2, -0.2]),
        Y_grid,
        baseline=1.0,
        baseline_effect=2.0,
        mixture_prop=0.3,
        n_covariates=10,
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, method in enumerate(methods):
        stats_dr = data[method]["dr"]
        stats_onestep = data[method]["onestep"]

        ax_high = axes[0, i]
        row_label = r"$P(Y^1 \mid V = v_1)$" if i == 0 else ""
        plot_density_comparison(
            ax_high,
            Y_grid,
            pdf_true_high,
            stats_dr,
            stats_onestep,
            "high",
            method_names[method],
            row_label=row_label,
        )
        ax_high.set_xlim(3, 38)
        ax_high.set_ylim(0, 0.105)

        ax_high.set_title(
            r"\textbf{" + method_names[method] + "}",
            fontsize=20,
            fontweight="bold",
            pad=10,
        )

        if i == 0:
            ax_high.set_ylabel("Density", fontsize=12, fontweight="bold")
            ax_high.tick_params(axis="y", which="both", length=4)
        else:
            ax_high.set_yticks([])

        ax_high.set_xticklabels([])

        ax_low = axes[1, i]
        row_label = r"$P(Y^1 \mid V = v_2)$" if i == 0 else ""
        plot_density_comparison(
            ax_low,
            Y_grid,
            pdf_true_low,
            stats_dr,
            stats_onestep,
            "low",
            method_names[method],
            row_label=row_label,
        )
        ax_low.set_xlim(-5, 30)
        ax_low.set_ylim(0, 0.075)

        ax_low.set_xlabel("$Y$", fontsize=12, fontweight="bold")
        ax_low.tick_params(axis="x", which="both", length=4)

        if i == 0:
            ax_low.set_ylabel("Density", fontsize=12, fontweight="bold")
            ax_low.tick_params(axis="y", which="both", length=4)
        else:
            ax_low.set_yticks([])

    plt.tight_layout(rect=[0, 0.08, 1, 0.98])

    handles, labels = axes[0, 0].get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=7,
        frameon=True,
        framealpha=0.98,
        edgecolor="black",
        fontsize=18,
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.5,
        handlelength=2.5,
        handleheight=1.5,
    )
    legend.get_frame().set_linewidth(1.2)

    output_pdf = "fig_densities.pdf"
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.0, dpi=300)
    print(f"PDF saved to {output_pdf}")

    output_png = "fig_densities.png"
    plt.savefig(output_png, format="png", bbox_inches="tight", pad_inches=0.0, dpi=300)
    print(f"PNG saved to {output_png}")

    plt.show()


if __name__ == "__main__":
    visualize_all_methods()
