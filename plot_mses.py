"""Publication-quality MSE convergence plots.

Reads the CSV produced by :mod:`experiment_robustness` and renders a
3x3 ``seaborn.FacetGrid`` (scenarios x methods) showing MSE vs sample
size on log-log axes for each estimation mode.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams


def plot_grid(csv_path, output_path="fig_mses.pdf"):
    """Render the robustness benchmark as a 3x3 FacetGrid.

    Rows correspond to misspecification scenarios (Both Correct,
    pi Wrong, mu Wrong) and columns to methods (Ridge, Deep Feature,
    Neural-Kernel).  Each panel shows MSE vs sample size with shaded
    standard-deviation bands for every estimation mode.

    Args:
        csv_path: Path to ``robustness_convergence_benchmark.csv``.
        output_path: Base output path; ``.pdf`` and ``.png`` variants
            are saved.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    scenario_map = {
        "Both Correct": "Both Correct",
        "Pi Wrong": r"$\hat{\pi}$ Wrong",
        "Mu Wrong": r"$\hat{\mu}_{Y|X}$ Wrong",
    }

    method_map = {
        "ridge": "Ridge Regression",
        "df": "Deep Feature",
        "nk": "Neural-Kernel",
    }

    mode_map = {
        "onestep": "One-Step",
        "dr": "DR",
        "ipw": "IPW",
        "pi": "PI",
    }

    df["Scenario"] = df["Scenario"].map(lambda x: scenario_map.get(x, x))
    df["Mode"] = df["Mode"].map(lambda x: mode_map.get(x, x))

    scenario_order = ["Both Correct", r"$\hat{\pi}$ Wrong", r"$\hat{\mu}_{Y|X}$ Wrong"]
    method_order = ["ridge", "df", "nk"]

    scenario_order = [s for s in scenario_order if s in df["Scenario"].unique()]
    method_order = [m for m in method_order if m in df["Method"].unique()]

    plt.style.use("seaborn-v0_8-paper")

    rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": "cm",
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10,
            "legend.title_fontsize": 10,
            # Line and marker settings
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            # Figure settings
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Use vector formats
            "pdf.fonttype": 42,  # TrueType fonts
            "ps.fonttype": 42,
            # Grid and axes
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "axes.axisbelow": True,
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            # Ticks
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )

    mode_colors = {
        "DR": "#E69F00",  # Orange
        "IPW": "#56B4E9",  # Sky blue
        "PI": "#009E73",  # Bluish green
        "One-Step": "#CC79A7",  # Reddish purple
    }

    mode_markers = {
        "DR": "o",  # Circle
        "IPW": "s",  # Square
        "PI": "^",  # Triangle up
        "One-Step": "D",  # Diamond
    }

    mode_styles = {
        "DR": "-",  # Solid
        "IPW": "--",  # Dashed
        "PI": "-.",  # Dash-dot
        "One-Step": ":",  # Dotted
    }

    fig_width = 6.5
    fig_height = 5.5

    g = sns.FacetGrid(
        df,
        row="Scenario",
        col="Method",
        row_order=scenario_order,
        col_order=method_order,
        hue="Mode",
        height=fig_height / 3,
        aspect=fig_width / fig_height,
        sharex=True,
        sharey="row",
        margin_titles=False,
        despine=True,
    )

    def plot_shaded(x, y, yerr, label, color, **kwargs):
        """Plot a line with a shaded +/- 1 std region.

        Called by ``FacetGrid.map`` for each mode.  Sorts data by *x*
        before plotting to ensure correct line ordering.
        """
        mode = label

        data = sorted(zip(x, y, yerr))
        xs, ys, yerrs = zip(*data) if data else ([], [], [])

        xs = np.array(xs)
        ys = np.array(ys)
        yerrs = np.array(yerrs)

        c = mode_colors[mode]
        m = mode_markers[mode]
        ls = mode_styles[mode]

        plt.fill_between(
            xs,
            ys - yerrs,
            ys + yerrs,
            color=c,
            alpha=0.15,
            linewidth=0,
            zorder=1,
        )

        plt.plot(
            xs,
            ys,
            label=mode,
            color=c,
            linestyle=ls,
            marker=m,
            markersize=4,
            linewidth=1.5,
            markeredgewidth=0.5,
            markeredgecolor="white",
            zorder=3 if mode == "DR" else 2,
            alpha=0.95,
        )

    g.map(plot_shaded, "N", "MSE", "Std")
    g.set(xscale="log", yscale="log")
    g.set_axis_labels("Sample Size ($n$)", "Mean Squared Error")

    for ax in g.axes.flat:
        title = ax.get_title()

        if "Scenario" in title or "Method" in title:
            ax.set_title("")

    for i, scenario in enumerate(scenario_order):
        g.axes[i, 0].set_ylabel(f"{scenario}\n\nMSE", fontsize=10)

    for j, method in enumerate(method_order):
        g.axes[0, j].set_title(
            r"\textbf{" + method_map[method] + "}", fontsize=13, pad=10
        )

    for ax in g.axes.flat:
        ax.grid(True, which="major", ls="--", alpha=0.3, linewidth=0.5)
        ax.grid(True, which="minor", ls=":", alpha=0.15, linewidth=0.3)
        ax.tick_params(which="both", direction="in")
        ax.set_xticks([1000, 10000])
        ax.set_xticklabels(["$10^3$", "$10^4$"])

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=mode_colors[m],
            linestyle=mode_styles[m],
            marker=mode_markers[m],
            markersize=5,
            linewidth=1.5,
            label=m,
            markeredgewidth=0.5,
            markeredgecolor="white",
        )
        for m in ["DR", "IPW", "PI", "One-Step"]
    ]

    legend = g.fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.42, 0.935),
        ncol=4,
        frameon=True,
        fancybox=False,
        shadow=False,
        edgecolor="black",
        facecolor="white",
        framealpha=0.95,
        columnspacing=1.5,
    )
    legend.get_frame().set_linewidth(0.8)

    plt.subplots_adjust(
        top=0.92, bottom=0.08, left=0.10, right=0.98, hspace=0.04, wspace=0.04
    )

    base_name = output_path.rsplit(".", 1)[0]

    pdf_path = f"{base_name}.pdf"
    plt.savefig(pdf_path, format="pdf", dpi=300, pad_inches=0.0, bbox_inches="tight")
    print(f"PDF saved to: {pdf_path}")

    png_path = f"{base_name}.png"
    plt.savefig(png_path, format="png", dpi=300, pad_inches=0.0, bbox_inches="tight")
    print(f"PNG saved to: {png_path}")

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate robustness comparison plots"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="robustness_convergence_benchmark.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="fig_mses.pdf",
        help="Output file path (will generate .pdf and .png)",
    )
    args = parser.parse_args()

    plot_grid(args.file, args.out)
