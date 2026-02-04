"""Publication-quality MNIST prototype comparison figure.

Loads the ``.npz`` file produced by :mod:`experiment_mnist` and
arranges the highest-density prototype images in a 3x10 grid
(Oracle / DR / One-Step x digits 0--9).
"""

import argparse

import matplotlib as mpl
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def setup_matplotlib_style():
    """Configure matplotlib rcParams for Computer Modern serif fonts."""
    rcParams["font.family"] = "serif"
    cmfont = font_manager.FontProperties(
        fname=mpl.get_data_path() + "/fonts/ttf/cmr10.ttf"
    )
    rcParams["font.serif"] = cmfont.get_name()
    rcParams["mathtext.fontset"] = "cm"
    rcParams["axes.formatter.use_mathtext"] = True
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42


def plot_modes(data_path):
    """
    Import images from the MNIST experiments and arrange them in 3 rows.

    Args:
        data_path: Path to .npz file containing keys 'clean', 'dr', 'one_step'
    """
    data = np.load(data_path)

    modes = [("clean", "Oracle"), ("dr", "DR"), ("one_step", "One-Step")]

    print(f"Loaded data from {data_path}. Found keys: {list(data.keys())}")

    setup_matplotlib_style()

    fig_width = 6.75
    fig_height = 2.025
    fig, axes = plt.subplots(3, 10, figsize=(fig_width, fig_height))

    for row_idx, (key, label) in enumerate(modes):
        if key not in data:
            print(f"Warning: Key '{key}' not found in .npz file. Skipping row.")
            continue

        images = data[key]

        for col_idx in range(10):
            ax = axes[row_idx, col_idx]
            ax.imshow(images[col_idx], cmap="gray", interpolation="nearest")
            ax.axis("off")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)

    pdf_path = "fig_mnist.pdf"
    plt.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"PDF saved to {pdf_path}")

    png_path = "fig_mnist.png"
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"PNG saved to {png_path}")

    plt.close()

    return pdf_path, png_path


def main():
    """CLI entry point for generating the MNIST figure."""
    parser = argparse.ArgumentParser(
        description="Create comparison ranking figures (Clean vs DR vs OneStep)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to .npz file containing experiment results",
        default="mnist_experiment_results.npz",
    )

    args = parser.parse_args()

    plot_modes(args.file)


if __name__ == "__main__":
    main()
