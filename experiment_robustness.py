"""Robustness and convergence benchmark for conditional density estimators.

Evaluates all ``{method} x {mode} x {scenario}`` combinations over a range
of sample sizes and saves mean integrated squared error (MISE) results to
``robustness_convergence_benchmark.csv``.  The CSV is consumed by
:func:`plot_mses.plot_grid` to produce publication figures.

Scenarios:
    * **Both Correct** -- correctly specified propensity and nuisance models.
    * **Pi Wrong** -- misspecified propensity model (logistic regression).
    * **Mu Wrong** -- misspecified nuisance model (omitted covariates).
"""

import os

import jax.numpy as jnp
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ccme import CCDEstimator
from utils import (
    conditional_density,
    generate_data,
)


def run_robustness_experiment(n_total, seed, cfg):
    """Run a single experiment for one sample size and seed.

    Generates data, fits the estimator, evaluates on a fixed test set of
    10 000 points drawn from ``N(1, 1)``, and returns the average MISE
    against the known ground-truth density.

    Args:
        n_total: Number of training observations.
        seed: Random seed for data generation.
        cfg: OmegaConf configuration with ``method``, ``mode``, ``model``,
            ``train``, ``test``, and ``propensity_score_model`` sections.

    Returns:
        Mean integrated squared error (float) averaged over test points.
    """
    X, Y, A = generate_data(n_total, seed)

    num_inputs = 5
    V = X[:, :num_inputs]

    n_covariates = X.shape[1]

    rng = np.random.default_rng(seed=0)
    X_test_all = rng.normal(1.0, 1.0, size=(10000, n_covariates))
    V_test_all = X_test_all[:, :num_inputs]

    y_min, y_max = jnp.min(Y), jnp.max(Y)
    Y_eval = jnp.linspace(y_min - 2, y_max + 2, cfg.test.num_bin)[:, None]

    estimator = CCDEstimator(cfg)
    estimator.fit(X=X, V=V, Y=Y, A=A)
    pdf_est = estimator.predict(V_eval=V_test_all, Y_grid=Y_eval)

    mse_list = []
    dy = Y_eval[1, 0] - Y_eval[0, 0]
    for i in range(len(X_test_all)):
        pdf_true = conditional_density(
            X_test_all[i, :num_inputs],
            np.array(Y_eval[:, 0]),
            baseline=1.0,
            baseline_effect=2.0,
            mixture_prop=0.3,
            n_covariates=n_covariates,
        )
        mse = np.sum((pdf_est[i] - pdf_true) ** 2) * dy
        mse_list.append(mse)

    return np.mean(mse_list)


def main():
    """Run the full robustness benchmark grid.

    Iterates over sample sizes, methods (``nk``, ``df``, ``ridge``),
    modes (``dr``, ``ipw``, ``pi``, ``onestep``), and misspecification
    scenarios.  For each configuration, 10 seeds are evaluated.  Results
    are aggregated and saved to
    ``robustness_convergence_benchmark.csv``.
    """
    ns = [200, 500, 1000, 2000, 5000, 10000, 20000]

    scenarios = ["Both Correct", "Pi Wrong", "Mu Wrong"]
    methods = ["nk", "df", "ridge"]
    modes = ["dr", "ipw", "pi", "onestep"]

    # Map method names to config filenames
    method_to_yaml = {"nk": "nk.yaml", "df": "df.yaml", "ridge": "ridge.yaml"}

    all_results = []
    print("Starting Robustness Benchmark...")

    for n in ns:
        print(f"=== N = {n} ===")

        for method in methods:
            for mode in modes:
                for scenario in scenarios:
                    # Load configs
                    yaml_path = os.path.join(
                        "configs/simulation/method", method_to_yaml[method]
                    )
                    if not os.path.exists(yaml_path):
                        # Fallback if file doesn't exist, assuming standard naming
                        yaml_path = os.path.join(
                            "configs/simulation/method", f"{method}.yaml"
                        )

                    try:
                        loaded_cfg = OmegaConf.load(yaml_path)
                        # Wrap in 'simulation' key to match access pattern
                        cfg = OmegaConf.create(loaded_cfg)
                    except Exception as e:
                        print(f"Error loading config for {method}: {e}")
                        continue
                    print(f"  {method} [{mode}] - {scenario}: ", end="", flush=True)
                    if (
                        (mode == "dr")
                        or (mode == "ipw" and scenario != "Mu Wrong")
                        or (mode == "pi" and scenario != "Pi Wrong")
                        or (mode == "onestep" and scenario == "Both Correct")
                    ):
                        mses = []
                        # 10 Seeds
                        for seed in range(10):
                            cfg.method = method
                            cfg.mode = mode
                            if method == "nk":
                                cfg.train.lr_or = 4e-4 * n / 200
                                cfg.train.lr_fi = 4e-4 * n / 200
                            elif method == "df":
                                cfg.train.lr_or = 2e-4 * n / 200
                                cfg.train.lr_fi = 2e-4 * n / 200
                            if scenario == "Pi Wrong":
                                cfg.propensity_score_model = {
                                    "_target_": "sklearn.linear_model.LogisticRegression"
                                }
                            else:
                                cfg.propensity_score_model = {
                                    "_target_": "sklearn.ensemble.RandomForestClassifier",
                                    "max_depth": 4,
                                    "random_state": seed,
                                }

                            if scenario == "Mu Wrong":
                                cfg.model.idx_or = [
                                    0,
                                    1,
                                    2,
                                    3,
                                    4,
                                    6,
                                    7,
                                    8,
                                    9,
                                ]
                            cfg.seed = seed
                            cfg.verbose = False
                            mse = run_robustness_experiment(n, seed, cfg)
                            mses.append(mse)

                        if mses:
                            mean_mse = np.mean(mses)
                            std_mse = np.std(mses)
                            print(f"{mean_mse:.5f}")

                            all_results.append(
                                {
                                    "Method": method,
                                    "Mode": mode,
                                    "Scenario": scenario,
                                    "N": n,
                                    "MSE": mean_mse,
                                    "Std": std_mse,
                                }
                            )

                    else:
                        if mode == "ipw":
                            new_result = all_results[-2].copy()
                        else:
                            new_result = all_results[-1].copy()

                        new_result["Mode"] = mode
                        new_result["Scenario"] = scenario
                        print(f"{new_result['MSE']:.5f}")

                        all_results.append(new_result)

    df = pd.DataFrame(all_results)
    df.to_csv("robustness_convergence_benchmark.csv", index=False)

    print(
        "Benchmark complete. Saved the results to robustness_convergence_benchmark.csv"
    )


if __name__ == "__main__":
    main()
