"""Inference (prediction) functions for conditional density estimation.

Each estimator has one or two prediction functions:

* Two-stage (DR / IPW / PI) functions use both a target model and a nuisance
  model with sample-splitting data.
* One-step functions use a single model trained on treated data only.

All functions return ``(density_matrix, sigma)`` where ``density_matrix`` has
shape ``(n_test, n_grid)`` and ``sigma`` is the kernel bandwidth used.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from kernels import gram_matrix


def estimate_ridge(
    model,
    state,
    V_test,
    yc,
    model_or,
    state_or,
    V_target,
    X_target,
    Y_target,
    A_target,
    pi_target,
    X_aux,
    Y_aux,
    mode="dr",
):
    """Kernel ridge regression density estimate (two-stage).

    Constructs kernel weights between auxiliary and target data, then combines
    IPW-weighted direct kernel evaluations with plug-in nuisance predictions.

    Args:
        model: Fitted RidgeModel containing ``lamb`` and ``sig_param``.
        state: Equinox state (unused for ridge, passed for API compat).
        V_test: Evaluation points of shape ``(n_test, D_v)``.
        yc: Outcome grid of shape ``(n_grid, D_y)``.
        model_or: Nuisance RidgeModel (unused, kept for API compat).
        state_or: Nuisance state (unused).
        V_target: Target-split conditioning variables, shape ``(n_target, D_v)``.
        X_target: Target-split full covariates, shape ``(n_target, D_x)``.
        Y_target: Target-split outcomes, shape ``(n_target, D_y)``.
        A_target: Target-split treatment indicators, shape ``(n_target,)``.
        pi_target: Target-split propensity scores, shape ``(n_target,)``.
        X_aux: Auxiliary-split covariates (treated only), shape ``(n_aux, D_x)``.
        Y_aux: Auxiliary-split outcomes (treated only), shape ``(n_aux, D_y)``.
        mode: One of ``"dr"``, ``"ipw"``, or ``"pi"``.

    Returns:
        Tuple of ``(density_matrix, sigma)`` where ``density_matrix`` has
        shape ``(n_test, n_grid)``.
    """
    lamb = model.lamb
    sig = jax.nn.softplus(model.sig_param)

    # 1. Nuisance Matrix B (Aux[X] -> Target[X])
    n_aux = X_aux.shape[0]
    K_aux = gram_matrix(X_aux, X_aux, sig, scaled=False)
    K_cross_X = gram_matrix(X_aux, X_target, sig, scaled=False)
    B = jnp.linalg.solve(K_aux + lamb * jnp.eye(n_aux), K_cross_X)

    # 2. Target Weights w (Target[V] -> Test[V])
    n_target = V_target.shape[0]
    K_V = gram_matrix(V_target, V_target, sig, scaled=False)
    K_V_cross = gram_matrix(V_target, V_test, sig, scaled=False)
    w = jnp.linalg.solve(K_V + lamb * jnp.eye(n_target), K_V_cross)

    # 3. Kernel Evals
    K_Y_test = gram_matrix(Y_target, yc, sig, scaled=False)
    K_Yaux_test = gram_matrix(Y_aux, yc, sig, scaled=False)

    w_ipw = (A_target / pi_target)[:, None]
    w_res = 1.0 - w_ipw

    if mode == "ipw":
        w_res = jnp.zeros_like(w_ipw)
    elif mode == "pi":
        w_ipw = jnp.zeros_like(w_ipw)
        w_res = jnp.ones_like(w_ipw)

    term1 = (w * w_ipw).T @ K_Y_test
    term2 = (w * w_res).T @ (B.T @ K_Yaux_test)

    return term1 + term2, sig


def estimate_nk(model, state, V_test, yc, *args, **kwargs):
    """Neural-Kernel density estimate.

    Applies the trained NeuralKernelNet to evaluation points and computes
    density as ``g(V_test) @ K(ypcl, yc)``.

    Args:
        model: Trained NeuralKernelNet (target model).
        state: Equinox state.
        V_test: Evaluation points of shape ``(n_test, D_v)``.
        yc: Outcome grid of shape ``(n_grid, D_y)``.
        *args: Additional arguments (ignored).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Tuple of ``(density_matrix, sigma)`` where ``density_matrix`` has
        shape ``(n_test, n_grid)``.
    """
    inference_model = eqx.nn.inference_mode(model)
    batch_model = jax.vmap(
        inference_model, in_axes=(0, None), out_axes=(0, None, None, None)
    )

    # Target model uses V_test directly
    g, _, ypcl, sig = batch_model(V_test, state)
    # g = g / jnp.sum(g, axis=1, keepdims=True)
    K_grid_y = gram_matrix(ypcl, yc, sig, scaled=True)
    return g @ K_grid_y, sig


def estimate_df(
    model,
    state,
    V_test,
    yc,
    model_or,
    state_or,
    V_target,
    X_target,
    Y_target,
    A_target,
    pi_target,
    X_aux,
    Y_aux,
    mode="dr",
):
    """Deep Feature density estimate (two-stage).

    Projects test points into feature space, constructs kernel regression
    weights, and combines IPW and plug-in nuisance predictions.

    Args:
        model: Trained target DeepFeatureNet.
        state: Equinox state for the target model.
        V_test: Evaluation points of shape ``(n_test, D_v)``.
        yc: Outcome grid of shape ``(n_grid, D_y)``.
        model_or: Trained nuisance DeepFeatureNet.
        state_or: Equinox state for the nuisance model.
        V_target: Target-split conditioning variables, shape ``(n_target, D_v)``.
        X_target: Target-split full covariates, shape ``(n_target, D_x)``.
        Y_target: Target-split outcomes, shape ``(n_target, D_y)``.
        A_target: Target-split treatment indicators, shape ``(n_target,)``.
        pi_target: Target-split propensity scores, shape ``(n_target,)``.
        X_aux: Auxiliary-split covariates (treated only), shape ``(n_aux, D_x)``.
        Y_aux: Auxiliary-split outcomes (treated only), shape ``(n_aux, D_y)``.
        mode: One of ``"dr"``, ``"ipw"``, or ``"pi"``.

    Returns:
        Tuple of ``(density_matrix, sigma)`` where ``density_matrix`` has
        shape ``(n_test, n_grid)``.
    """
    inference_model = eqx.nn.inference_mode(model)
    lamb = model.lamb

    # 1. Features
    batch_target = jax.vmap(
        inference_model, in_axes=(0, None), out_axes=(0, None, None, None)
    )
    psi_target, _, _, sig = batch_target(V_target, state)
    psi_test, _, _, _ = batch_target(V_test, state)

    batch_nuis = jax.vmap(model_or, in_axes=(0, None), out_axes=(0, None, None, None))
    psi0_aux, _, _, _ = batch_nuis(X_aux, state_or)
    psi0_target, _, _, _ = batch_nuis(X_target, state_or)

    # 2. Nuisance Smoothing
    d0 = psi0_aux.shape[1]
    Cov0 = psi0_aux.T @ psi0_aux + lamb * jnp.eye(d0)

    K_Yaux_test = gram_matrix(Y_aux, yc, sig, scaled=False)
    K_proj_nuis = jnp.linalg.solve(Cov0, psi0_aux.T @ K_Yaux_test)
    Preds_nuis = psi0_target @ K_proj_nuis

    # 3. Target Weights
    d = psi_target.shape[1]
    Cov = psi_target.T @ psi_target + lamb * jnp.eye(d)
    w_proj = jnp.linalg.solve(Cov, psi_test.T)

    w_ipw = (A_target / pi_target)[:, None]
    w_res = 1.0 - w_ipw

    if mode == "ipw":
        w_res = jnp.zeros_like(w_ipw)
    elif mode == "pi":
        w_ipw = jnp.zeros_like(w_ipw)
        w_res = jnp.ones_like(w_ipw)

    K_Y_test = gram_matrix(Y_target, yc, sig, scaled=False)

    A1 = w_proj.T @ (psi_target.T @ (w_ipw * K_Y_test))
    A2 = w_proj.T @ (psi_target.T @ (w_res * Preds_nuis))

    return A1 + A2, sig


def estimate_ridge_onestep(model, state, V_test, yc, V_train, Y_train, **kwargs):
    """Kernel ridge regression density estimate (one-step, treated data only).

    Standard kernel ridge regression of P(Y|V) using only treated
    observations, without sample splitting or propensity weighting.

    Args:
        model: Fitted RidgeModel containing ``lamb`` and ``sig_param``.
        state: Equinox state (unused).
        V_test: Evaluation points of shape ``(n_test, D_v)``.
        yc: Outcome grid of shape ``(n_grid, D_y)``.
        V_train: Training conditioning variables (treated only), shape ``(n_train, D_v)``.
        Y_train: Training outcomes (treated only), shape ``(n_train, D_y)``.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Tuple of ``(density_matrix, sigma)`` where ``density_matrix`` has
        shape ``(n_test, n_grid)``.
    """
    lamb = model.lamb
    sig = jax.nn.softplus(model.sig_param)

    # 1. Compute Weights w = (K_VV + n*lam*I)^-1 K_V_test
    # Note: ridge_solve adds the n*lam*I term
    n_train = V_train.shape[0]

    K_VV = gram_matrix(V_train, V_train, sig, scaled=False)
    K_V_cross = gram_matrix(V_train, V_test, sig, scaled=False)

    # w shape: (n_train, n_test)
    w = jnp.linalg.solve(K_VV + lamb * jnp.eye(n_train), K_V_cross)

    # 2. Kernel Evaluations
    K_Y_test = gram_matrix(Y_train, yc, sig, scaled=False)

    # 3. Density Estimate: w.T @ K_Y
    # Shape: (n_test, n_train) @ (n_train, n_grid) -> (n_test, n_grid)
    return w.T @ K_Y_test, sig


def estimate_df_onestep(model, state, V_test, yc, V_train, Y_train, **kwargs):
    """Deep Feature density estimate (one-step, treated data only).

    Standard feature-space ridge regression of P(Y|V) using only treated
    observations, without sample splitting or propensity weighting.

    Args:
        model: Trained DeepFeatureNet.
        state: Equinox state.
        V_test: Evaluation points of shape ``(n_test, D_v)``.
        yc: Outcome grid of shape ``(n_grid, D_y)``.
        V_train: Training conditioning variables (treated only), shape ``(n_train, D_v)``.
        Y_train: Training outcomes (treated only), shape ``(n_train, D_y)``.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Tuple of ``(density_matrix, sigma)`` where ``density_matrix`` has
        shape ``(n_test, n_grid)``.
    """
    inference_model = eqx.nn.inference_mode(model)
    batch_model = jax.vmap(
        inference_model, in_axes=(0, None), out_axes=(0, None, None, None)
    )

    # 1. Get Features
    # psi_train: (n_train, dim)
    psi_train, _, lamb, sig = batch_model(V_train, state)
    # psi_test: (n_test, dim)
    psi_test, _, _, _ = batch_model(V_test, state)

    dim = psi_train.shape[1]

    # 2. Solve Weights in Feature Space
    # Cov = Psi.T @ Psi + n*lam*I (matching DF training scaling)
    # Note: In training we removed 'n' from reg based on previous discussion,
    # but for ridge algebra on features, standard form is usually unscaled lambda
    # if the loss was mean-reduced.
    # Consistent with loss_df: reg = lamb * I

    Cov = psi_train.T @ psi_train + lamb * jnp.eye(dim)

    # w_proj = Cov^-1 @ Psi_test.T
    # shape: (dim, n_test)
    w_proj = jnp.linalg.solve(Cov, psi_test.T)

    # 3. Project back to sample weights
    # alpha = Psi_train @ w_proj
    # shape: (n_train, n_test)
    weights = psi_train @ w_proj

    # 4. Density Estimate
    K_Y_test = gram_matrix(Y_train, yc, sig, scaled=False)

    return weights.T @ K_Y_test, sig
