"""Tasks 1 and 3 of the downstream causal tasks, for Figures 2 and 3 of the paper.

Run from the repository root:
    python experiments/experiment_cvar_policy.py
"""

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from ccme import CCDEstimator
from downstream import (
    PROPENSITY_CLIP,
    cate_pseudo_outcomes,
    fit_propensity,
    gaussian_gram,
    generate_data,
    linear_smoother,
    select_bandwidths,
    split_units,
    true_cvar,
)

N_SAMPLES = 2000
N_REPLICATIONS = 100
FIRST_SEED = 650000
RIDGE_LAMBDA = 4.0
V_GRID = np.linspace(-2.0, 2.0, 100)
Y_GRID = np.linspace(-20.0, 20.0, 1000)
Y_STEP = float(Y_GRID[1] - Y_GRID[0])
ALPHAS = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
CURVE_ALPHA = 0.10
TRUE_CATE = 1.5
DENSITY_METHODS = ["ccme_dr", "one_step", "cpme"]


def ccme_config(mode, seed, sigma):
    """Build the configuration of the ridge estimator of CCDEstimator.

    Args:
        mode: ``"dr"`` for CCME-DR or ``"onestep"`` for One-Step.
        seed: Seed of CCDEstimator.
        sigma: Bandwidth of the Gaussian kernels.

    Returns:
        OmegaConf configuration for :class:`ccme.CCDEstimator`.
    """
    return OmegaConf.create(
        {
            "method": "ridge",
            "mode": mode,
            "seed": seed,
            "verbose": False,
            "n_folds": None,
            "pi_clip": PROPENSITY_CLIP,
            "model": {
                "output_dim": 25,
                "hidden_dim": [32, 32],
                "sigma_init": sigma,
                "lamb": RIDGE_LAMBDA,
                "learn_sigma": False,
                "idx_pi": [0, 1],
                "idx_or": [2, 3],
            },
            "propensity_score_model": {
                "_target_": "sklearn.linear_model.LogisticRegression",
                "C": float("inf"),
                "max_iter": 1000,
            },
        }
    )


def ccme_density(config, x_input, v_input, y, treatment, v_eval):
    """Fit CCDEstimator and estimate the density of a potential outcome on Y_GRID.

    Args:
        config: Configuration from :func:`ccme_config`.
        x_input: Covariates of shape ``(N, D_x)``.
        v_input: Conditioning variable of shape ``(N, D_v)``.
        y: Outcomes of shape ``(N,)``.
        treatment: Binary indicators of the group whose potential outcome is estimated,
            shape ``(N,)``.
        v_eval: Evaluation points of V, shape ``(L, D_v)``.

    Returns:
        Density estimates of shape ``(L, len(Y_GRID))``.
    """
    estimator = CCDEstimator(config).fit(X=x_input, V=v_input, Y=y, A=treatment)
    return estimator.predict(V_eval=v_eval, Y_grid=Y_GRID)


def ccme_densities(x, v, y, a, mode, seed, bandwidths):
    """Estimate the densities of Y^0 and Y^1 given V on V_GRID with CCDEstimator.

    CCDEstimator with the ridge method uses the bandwidth on Y for every kernel. Scaling
    X by sigma_y / sigma_x and V by sigma_y / sigma_v gives Gaussian kernels with
    bandwidths sigma_x and sigma_v on the original scales. The propensity score is
    fitted on the unscaled copy of X.

    Args:
        x: Covariates of shape ``(N, 2)``.
        v: Conditioning variable of shape ``(N, 1)``.
        y: Outcomes of shape ``(N,)``.
        a: Binary treatment indicators of shape ``(N,)``.
        mode: ``"dr"`` for CCME-DR or ``"onestep"`` for One-Step.
        seed: Seed of CCDEstimator.
        bandwidths: Bandwidths from :func:`downstream.select_bandwidths`.

    Returns:
        Tuple ``(density0, density1)`` of the densities of Y^0 and Y^1, each of shape
        ``(len(V_GRID), len(Y_GRID))``.
    """
    scale_x = bandwidths["y"] / bandwidths["x"]
    scale_v = bandwidths["y"] / bandwidths["v"]
    x_input = np.hstack([x, x * scale_x])
    v_input = v * scale_v
    v_eval = V_GRID[:, None] * scale_v
    config = ccme_config(mode, seed, bandwidths["y"])
    density1 = ccme_density(config, x_input, v_input, y, a, v_eval)
    density0 = ccme_density(config, x_input, v_input, y, 1 - a, v_eval)
    return density0, density1


def cpme_density(x, y, treated, pi, bandwidths):
    """Estimate the density of Y^1 on Y_GRID with CPME.

    CPME averages the augmented inverse propensity weighted kernel densities over all
    units, so the estimate does not depend on V. Passing ``1 - a`` as ``treated`` and
    ``1 - pi`` as ``pi`` gives the density of Y^0.

    Args:
        x: Covariates of shape ``(N, D_x)``.
        y: Outcomes of shape ``(N,)``.
        treated: Binary treatment indicators of shape ``(N,)``.
        pi: Propensity scores P(treated = 1 | X), shape ``(N,)``.
        bandwidths: Bandwidths from :func:`downstream.select_bandwidths`.

    Returns:
        Density estimate of shape ``(len(Y_GRID),)``, normalized to integrate to one.
    """
    sigma_x, sigma_y = bandwidths["x"], bandwidths["y"]
    units = np.where(treated == 1)[0]
    gram = gaussian_gram(x[units], x[units], sigma_x) + RIDGE_LAMBDA * np.eye(
        units.size
    )
    weights = np.linalg.solve(gram, gaussian_gram(x[units], x, sigma_x))
    scale = np.sqrt(2 * np.pi * sigma_y**2)
    kernel = gaussian_gram(y, Y_GRID, sigma_y) / scale
    outcome_model = weights.T @ (gaussian_gram(y[units], Y_GRID, sigma_y) / scale)
    ipw = (treated / pi)[:, None]
    density = np.clip(
        (ipw * kernel + (1.0 - ipw) * outcome_model).mean(axis=0), 0, None
    )
    return density / (density.sum() * Y_STEP)


def cpme_densities(x, y, a, bandwidths):
    """Estimate the densities of Y^0 and Y^1 with CPME.

    Args:
        x: Covariates of shape ``(N, D_x)``.
        y: Outcomes of shape ``(N,)``.
        a: Binary treatment indicators of shape ``(N,)``.
        bandwidths: Bandwidths from :func:`downstream.select_bandwidths`.

    Returns:
        Tuple ``(density0, density1)``, each of shape ``(len(V_GRID), len(Y_GRID))``.
        All rows are equal, since CPME does not depend on V.
    """
    pi = fit_propensity(x, a, x)
    density1 = cpme_density(x, y, a, pi, bandwidths)
    density0 = cpme_density(x, y, 1 - a, 1 - pi, bandwidths)
    return np.tile(density0, (V_GRID.size, 1)), np.tile(density1, (V_GRID.size, 1))


def grid_mean(density):
    """Compute the mean of a density given on Y_GRID.

    Args:
        density: Density values on Y_GRID.

    Returns:
        The mean as a float.
    """
    return float(np.sum(density * Y_GRID) * Y_STEP)


def grid_cvar(density, alpha):
    """Compute the mean of the lower alpha-tail of a density given on Y_GRID.

    Args:
        density: Density values on Y_GRID.
        alpha: Tail level in (0, 1).

    Returns:
        The CVaR at level alpha as a float.
    """
    cdf = np.clip(np.cumsum(density) * Y_STEP, 0.0, 1.0)
    mask = Y_GRID <= float(np.interp(alpha, cdf, Y_GRID))
    mass = np.sum(density[mask]) * Y_STEP
    return float(np.sum(Y_GRID[mask] * density[mask]) * Y_STEP / mass)


def policy_values(effects, true_cvar0, true_cvar1):
    """Compute the true value of the plug-in policy at each tail level alpha.

    At level ALPHAS[k], the policy treats the points of V_GRID where row k of
    ``effects`` is positive. The value is the average over V_GRID of the true CVaR of
    Y^1 at the treated points and of Y^0 at the other points.

    Args:
        effects: Estimated effects of shape ``(len(ALPHAS), len(V_GRID))``.
        true_cvar0: True CVaR of Y^0, of the same shape as ``effects``.
        true_cvar1: True CVaR of Y^1, of the same shape as ``effects``.

    Returns:
        Policy values of shape ``(len(ALPHAS),)``.
    """
    return np.where(effects > 0, true_cvar1, true_cvar0).mean(axis=1)


def main():
    """Run Tasks 1 and 3 and save the results to ``cvar_policy_results.npz``."""
    true_cvar0, true_cvar1 = (
        np.array([[true_cvar(v, alpha, treated) for v in V_GRID] for alpha in ALPHAS])
        for treated in (False, True)
    )
    true_effects = true_cvar1 - true_cvar0
    oracle_values = policy_values(true_effects, true_cvar0, true_cvar1)
    curve_index = list(ALPHAS).index(CURVE_ALPHA)

    results = {}
    for method in DENSITY_METHODS:
        results[f"curve_{method}"] = np.zeros((N_REPLICATIONS, V_GRID.size))
        results[f"cvar_mae_{method}"] = np.zeros(N_REPLICATIONS)
        results[f"cate_mae_{method}"] = np.zeros(N_REPLICATIONS)
        results[f"regret_{method}"] = np.zeros((N_REPLICATIONS, ALPHAS.size))
    results["cate_mae_cate_dr"] = np.zeros(N_REPLICATIONS)
    results["regret_cate_dr"] = np.zeros((N_REPLICATIONS, ALPHAS.size))

    for replication in tqdm(range(N_REPLICATIONS)):
        seed = FIRST_SEED + replication
        x, y, a, v = generate_data(N_SAMPLES, seed)
        bandwidths = select_bandwidths(x, v, y, a, RIDGE_LAMBDA, seed)
        densities = {
            "ccme_dr": ccme_densities(x, v, y, a, "dr", replication, bandwidths),
            "one_step": ccme_densities(x, v, y, a, "onestep", replication, bandwidths),
            "cpme": cpme_densities(x, y, a, bandwidths),
        }
        for method, (density0, density1) in densities.items():
            pairs = list(zip(density0, density1))
            cate = np.array([grid_mean(p1) - grid_mean(p0) for p0, p1 in pairs])
            effects = np.array(
                [
                    [grid_cvar(p1, alpha) - grid_cvar(p0, alpha) for p0, p1 in pairs]
                    for alpha in ALPHAS
                ]
            )
            curve_error = effects[curve_index] - true_effects[curve_index]
            results[f"curve_{method}"][replication] = effects[curve_index]
            results[f"cvar_mae_{method}"][replication] = np.mean(np.abs(curve_error))
            results[f"cate_mae_{method}"][replication] = np.mean(
                np.abs(cate - TRUE_CATE)
            )
            values = policy_values(effects, true_cvar0, true_cvar1)
            results[f"regret_{method}"][replication] = oracle_values - values

        nuisance, target, _, _ = split_units(N_SAMPLES, seed)
        pseudo_outcomes = cate_pseudo_outcomes(x, y, a, nuisance, target)
        cate = linear_smoother(x[target, 0], V_GRID) @ pseudo_outcomes
        values = policy_values(np.tile(cate, (ALPHAS.size, 1)), true_cvar0, true_cvar1)
        results["cate_mae_cate_dr"][replication] = np.mean(np.abs(cate - TRUE_CATE))
        results["regret_cate_dr"][replication] = oracle_values - values

    np.savez(
        "cvar_policy_results.npz",
        v_grid=V_GRID,
        alphas=ALPHAS,
        curve_alpha=CURVE_ALPHA,
        true_curve=true_effects[curve_index],
        **results,
    )
    print("Saved cvar_policy_results.npz")
    for method in DENSITY_METHODS:
        cvar_mae = results[f"cvar_mae_{method}"].mean()
        cate_mae = results[f"cate_mae_{method}"].mean()
        print(f"{method}: CVaR MAE {cvar_mae:.3f}, CATE MAE {cate_mae:.3f}")
    print(f"cate_dr: CATE MAE {results['cate_mae_cate_dr'].mean():.3f}")


if __name__ == "__main__":
    main()
