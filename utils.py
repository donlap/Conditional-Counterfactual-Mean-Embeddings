"""Utility functions for data generation, ground-truth computation, and helpers.

Provides:

* :func:`generate_data` -- synthetic data generation with confounding, heteroscedastic
  noise, and a Gaussian mixture outcome.
* :func:`conditional_density` -- analytical ground-truth density P(Y^1 | V) for the
  synthetic DGP, computed by marginalizing over unobserved covariates.
* :func:`train_nuisance_models` -- fit a scikit-learn classifier for propensity scores.
* :func:`get_median_heuristic` -- median heuristic for kernel bandwidth selection.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats
from scipy.special import expit


def generate_data(n_samples, seed):
    """Generate synthetic observational data with confounded treatment.

    The data-generating process features:

    * 10-dimensional covariates X ~ N(1, I).
    * Non-random treatment assignment depending on X[0] and X[5].
    * Heterogeneous treatment effects depending on X.
    * Heteroscedastic noise depending on X[0] and X[4].
    * A Gaussian mixture component (shift of +15) with probability
      depending on X[0].

    Args:
        n_samples: Number of observations to generate.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of ``(X, Y, A)`` where:
            - X: Covariates of shape ``(n_samples, 10)``.
            - Y: Outcomes of shape ``(n_samples,)``.
            - A: Binary treatment indicators of shape ``(n_samples,)``.
    """
    rng = np.random.default_rng(seed)
    n_covariates = 10

    baseline = 1.0
    baseline_effect = 2.0
    covariate_coeffs = np.array([1.0, -0.5, 0.8, -0.7, 0.6, 1.0, 0.3, -0.2, 0.1, -0.3])
    interaction_covariates = np.array(
        [0.8, 0.0, 0.0, 0.6, 0.0, 2.0, 0.4, 0.0, 0.0, 0.2]
    )
    base_std = 0.5
    mixture_prop = 0.3

    X = rng.normal(loc=1, scale=1, size=(n_samples, n_covariates))

    is_in_box = 1.0 * ((X[:, 0] >= 0.0) & (X[:, 0] <= 2.0)) * (X[:, 5] >= 1.5)
    prob_t = 0.1 * (1.0 - is_in_box) + 0.9 * (is_in_box)
    A = rng.binomial(1, prob_t)

    effect = baseline_effect + np.dot(X, interaction_covariates)
    mean_y = baseline + np.dot(X, covariate_coeffs) + A * effect

    std_y = base_std * (1 + 0.5 * np.abs(X[:, 0]) + 0.3 * np.abs(X[:, 4]))

    is_mixture = rng.binomial(1, expit(0.5 * X[:, 0]), size=n_samples)
    shift = is_mixture * 15.0

    noise = rng.standard_normal(size=n_samples) * std_y
    Y = mean_y + shift + noise

    return X, Y, A


def conditional_density(
    X_observed,
    y_range,
    baseline=1.0,
    covariate_coeffs=None,
    baseline_effect=2.0,
    interaction_covariates=None,
    base_std=0.5,
    mixture_prop=0.3,
    n_covariates=10,
):
    """Compute the analytical ground-truth density P(Y^1 | V = X_observed).

    Marginalizes over unobserved covariates (indices >= len(X_observed))
    assuming independent N(1, 1) priors and the data-generating process
    defined in :func:`generate_data`.

    Args:
        X_observed: Observed conditioning covariate values, shape ``(n_obs,)``.
        y_range: Outcome values at which to evaluate the density, shape ``(G,)``.
        baseline: Intercept term in the outcome model.
        covariate_coeffs: Coefficient vector for the main covariate effect.
            Defaults to the values used in :func:`generate_data`.
        baseline_effect: Additive treatment effect constant.
        interaction_covariates: Coefficient vector for treatment-covariate
            interactions.  Defaults to the values used in :func:`generate_data`.
        base_std: Base noise standard deviation.
        mixture_prop: Unused (mixture probability is computed from X_observed[0]).
        n_covariates: Total number of covariates in the DGP.

    Returns:
        Array of density values with the same shape as ``y_range``.
    """

    if covariate_coeffs is None:
        covariate_coeffs = np.array(
            [1.0, -0.5, 0.8, -0.7, 0.6, 1.0, 0.3, -0.2, 0.1, -0.3]
        )

    if interaction_covariates is None:
        interaction_covariates = np.array(
            [0.8, 0.0, 0.0, 0.6, 0.0, 2.0, 0.4, 0.0, 0.0, 0.2]
        )

    # Covariance matrix for X (identity matrix as in your code)
    # Sigma_X = np.eye(n_covariates)

    # Partition covariance matrix
    # X = [X_obs, X_unobs] where X_obs has first 5 components
    n_obs = len(X_observed)
    n_unobs = n_covariates - n_obs

    # Sigma_11 = Sigma_X[:n_obs, :n_obs]  # Cov(X_obs, X_obs)
    # Sigma_12 = Sigma_X[:n_obs, n_obs:]  # Cov(X_obs, X_unobs)
    # Sigma_21 = Sigma_X[n_obs:, :n_obs]  # Cov(X_unobs, X_obs)
    # Sigma_22 = Sigma_X[n_obs:, n_obs:]  # Cov(X_unobs, X_unobs)

    # Mean of unobserved variables is 1
    # mu_unobs = np.ones(n_unobs)

    # Conditional distribution: X_unobs | X_obs ~ N(mu_cond, Sigma_cond)
    # Sigma_22_inv = np.linalg.inv(Sigma_22)
    # mu_cond = mu_unobs + Sigma_21 @ np.linalg.inv(Sigma_11) @ (
    #     X_observed - np.zeros(n_obs)
    # )
    # Sigma_cond = Sigma_22 - Sigma_21 @ np.linalg.inv(Sigma_11) @ Sigma_12

    # Since Sigma_X = I, this simplifies to:
    # mu_cond = 0 (because Sigma_21 = 0 for identity matrix)
    # Sigma_cond = I (unobserved variables remain independent)
    mu_cond = np.ones(n_unobs)
    Sigma_cond = np.eye(n_unobs)

    # Now calculate the conditional distribution of Y^1
    # Y^1 = baseline + X^T * covariate_coeffs + baseline_effect + X^T * interaction_covariates + noise

    # Split coefficients for observed and unobserved parts
    beta_obs = covariate_coeffs[:n_obs]
    beta_unobs = covariate_coeffs[n_obs:]
    gamma_obs = interaction_covariates[:n_obs]
    gamma_unobs = interaction_covariates[n_obs:]

    # Fixed part (depends only on observed X)
    fixed_part = (
        baseline
        + baseline_effect
        + np.dot(X_observed, beta_obs)
        + np.dot(X_observed, gamma_obs)
    )

    # Random part (depends on unobserved X)
    # E[X_unobs^T * (beta_unobs + gamma_unobs) | X_obs] = mu_cond^T * (beta_unobs + gamma_unobs)
    combined_coeff = beta_unobs + gamma_unobs
    random_mean = np.dot(mu_cond, combined_coeff)

    # Var[X_unobs^T * (beta_unobs + gamma_unobs) | X_obs] = (beta_unobs + gamma_unobs)^T * Sigma_cond * (beta_unobs + gamma_unobs)
    random_var = np.dot(combined_coeff, Sigma_cond @ combined_coeff)

    # Total mean and variance for the linear part
    linear_mean = fixed_part + random_mean
    linear_var = random_var

    # Add noise variance (heteroscedastic)
    # std_y = base_std * (1 + 0.5 * |X[0]| + 0.3 * |X[4]|)
    # Both X[0] and X[4] are observed since we're conditioning on first 5 variables (indices 0-4)

    noise_std = base_std * (
        1 + 0.5 * np.abs(X_observed[0]) + 0.3 * np.abs(X_observed[4])
    )
    noise_var = noise_std**2

    # Total variance
    total_var = linear_var + noise_var
    total_std = np.sqrt(total_var)

    # Conditional distributions (first 5 variables only)
    mixture_prop = expit(0.5 * X_observed[0])
    pdf_cond = (1 - mixture_prop) * stats.norm.pdf(
        y_range, linear_mean, total_std
    ) + mixture_prop * stats.norm.pdf(y_range, linear_mean + 15.0, total_std)

    return pdf_cond


def train_nuisance_models(model, X, Y, A):
    """Fit a propensity score classifier P(A=1|X).

    Args:
        model: A scikit-learn classifier (e.g. RandomForestClassifier).
        X: Covariates of shape ``(N, D_x)``.
        Y: Outcomes (unused, kept for API compatibility).
        A: Binary treatment indicators of shape ``(N,)``.

    Returns:
        The fitted classifier.
    """
    model.fit(X, A)
    propensity_scores = model.predict_proba(X[A == 1])[:, 1]  # P(A=1|X)
    np.clip(propensity_scores, a_min=1e-16, a_max=1)
    return model


def get_median_heuristic(Y, n_subsample=2000, random_seed=None):
    """Compute the median heuristic for Gaussian kernel bandwidth selection.

    Calculates the median of all pairwise Euclidean distances in Y (or a
    random subsample thereof).

    Args:
        Y: Input data of shape ``(N,)`` or ``(N, 1)``.
        n_subsample: Maximum number of rows to use.  If ``None``, uses all rows.
        random_seed: Random seed for reproducible subsampling.

    Returns:
        The median pairwise distance as a float, or 1.0 if the median is
        below ``1e-8``.
    """
    # Ensure Y is (N, 1)
    Y = jnp.array(Y)
    if Y.ndim == 1:
        Y = Y[:, None]

    # Subsample if requested
    if n_subsample is not None and n_subsample < Y.shape[0]:
        key = jax.random.PRNGKey(random_seed if random_seed is not None else 0)
        indices = jax.random.choice(
            key, Y.shape[0], shape=(n_subsample,), replace=False
        )
        Y = Y[indices]

    n = Y.shape[0]

    Y_sq = jnp.sum(Y**2, axis=1, keepdims=True)
    D2 = Y_sq + Y_sq.T - 2 * jnp.dot(Y, Y.T)
    D2 = jnp.maximum(D2, 0.0)
    D = jnp.sqrt(D2)

    triu_idx = jnp.triu_indices(D.shape[0], k=1)
    pairwise_dists = D[triu_idx]

    sigma = jnp.median(pairwise_dists)

    return float(jax.lax.cond(sigma < 1e-8, lambda: 1.0, lambda: sigma))
