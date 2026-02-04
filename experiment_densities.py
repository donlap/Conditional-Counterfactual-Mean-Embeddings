"""Repeated density estimation experiment on simulated data.

Runs :class:`ccme.CCDEstimator` over multiple random seeds for two
covariate profiles ("high" and "low") and saves per-run density
estimates to a CSV file.  The CSV is consumed by
:func:`plot_densities.visualize_all_methods` to produce publication
figures.

Usage::

    python demo.py
"""

import numpy as np
import pandas as pd
from hydra import compose, initialize
from itertools import product
from omegaconf import DictConfig
from tqdm import tqdm

from ccme import CCDEstimator
from utils import generate_data

METHODS = ["ridge", "df", "nk"]
MODES = ["dr", "onestep"]


def run_experiment(cfg: DictConfig) -> None:
    """Run repeated estimation for a single method/mode and save density CSV.

    Args:
        cfg: Hydra configuration with ``method``, ``mode``, and
            nested ``test.num_bin``.
    """
    print(f"\nRunning Repeated Estimation Experiment")
    print(f"Method: {cfg.method}")
    print(f"Mode: {cfg.mode}")

    num_inputs = 5
    num_train = 20000
    num_runs = 30

    # 1. Define Evaluation Points (V)

    # High features profile
    V_high = np.array([[2.2, -0.2, 2.2, -0.2, 2.2]])
    # Low features profile
    V_low = np.array([[-0.2, 2.2, -0.2, 2.2, -0.2]])

    V_eval = np.concatenate([V_high, V_low], axis=0)

    # Define Y grid for density estimation
    y_min, y_max = -6.0, 40.0  # Fixed range to ensure columns align in CSV
    Y_grid = np.linspace(y_min, y_max, cfg.test.num_bin)[:, None]

    results_data = {"Y_grid": Y_grid.flatten()}

    # 2. Loop for Runs
    for i in tqdm(range(num_runs), desc=f"{cfg.method}/{cfg.mode}"):
        X, Y, A = generate_data(n_samples=num_train, seed=i)

        V_train = X[:, :num_inputs]

        # Estimation
        estimator = CCDEstimator(cfg)
        estimator.fit(X=X, V=V_train, Y=Y, A=A)
        pdf_est = estimator.predict(V_eval=V_eval, Y_grid=Y_grid)

        results_data[f"run_{i}_high"] = pdf_est[0]
        results_data[f"run_{i}_low"] = pdf_est[1]

    # 3. Save to CSV
    df = pd.DataFrame(results_data)
    csv_filename = f"density_results_{cfg.method}_{cfg.mode}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Results from {num_runs} runs saved to {csv_filename}")


def main() -> None:
    """Run repeated estimation across all method/mode combinations."""
    with initialize(config_path="configs/simulation", version_base=None):
        for method, mode in product(METHODS, MODES):
            cfg = compose(config_name="config", overrides=[f"method={method}", f"mode={mode}"])
            run_experiment(cfg)


if __name__ == "__main__":
    main()
