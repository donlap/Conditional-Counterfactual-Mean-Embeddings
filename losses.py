"""Loss functions for two-stage conditional density estimation.

Stage 1 (nuisance) losses train a model to estimate P(Y|X) using only
treated observations.  Stage 2 (target) losses incorporate propensity
scores and doubly robust corrections to estimate P(Y^1|V).

Each loss function returns ``(loss, state)`` and is compatible with
:func:`eqx.filter_value_and_grad`.
"""

import jax
import jax.numpy as jnp

from kernels import gram_matrix

# ==========================================
# Stage 1 Losses (Nuisance - Uses Full X)
# ==========================================


def loss_nk(model, state, X, Y):
    """Neural-Kernel nuisance loss (Stage 1).

    Minimizes the RKHS norm ||f - mu_{Y|X}||^2 using the grid-based
    coefficient representation from :class:`models.NeuralKernelNet`.

    Args:
        model: NeuralKernelNet model (un-vmapped, single-sample callable).
        state: Equinox state object.
        X: Covariates of shape ``(N, D_x)``.
        Y: Outcomes of shape ``(N, D_y)``.

    Returns:
        Tuple of ``(loss, state)``.
    """
    batch_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None, None, None))
    g, state, ypcl, sig = batch_model(X, state)

    K_MM = gram_matrix(ypcl, ypcl, sig, scaled=True)
    K_YM = gram_matrix(Y, ypcl, sig, scaled=True)

    term1 = jnp.sum((g @ K_MM) * g)
    term2 = -2 * jnp.sum(g * K_YM)

    loss = (term1 + term2) / X.shape[0]
    return loss, state


def loss_df(model, state, X, Y):
    """Deep Feature nuisance loss (Stage 1).

    Maximizes the trace criterion tr(Cov^{-1} Psi^T K_Y Psi) which
    corresponds to minimizing the RKHS regression error in feature space.

    Args:
        model: DeepFeatureNet model.
        state: Equinox state object.
        X: Covariates of shape ``(N, D_x)``.
        Y: Outcomes of shape ``(N, D_y)``.

    Returns:
        Tuple of ``(loss, state)``.
    """
    batch_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None, None, None))
    psi, state, lamb, sig = batch_model(X, state)

    n, m = X.shape[0], psi.shape[1]
    K_Y = gram_matrix(Y, Y, sig, scaled=False)
    reg = lamb * jnp.eye(m)

    Cov = psi.T @ psi + reg
    Proj = psi.T @ K_Y @ psi
    M_inv = jnp.linalg.inv(Cov)
    tr_val = jnp.sum(M_inv * Proj.T)

    return -tr_val / n, state


# ==========================================
# Stage 2 Losses (Target - Uses V and X)
# ==========================================


def loss_nk_dr(
    model, state, V, X, Y, A, model_or, state_or, pi, X_ref=None, Y_ref=None
):
    """Doubly robust Neural-Kernel target loss (Stage 2).

    Combines IPW-weighted direct observations with plug-in nuisance
    predictions to form the DR cross-term.

    Args:
        model: Target NeuralKernelNet model (uses V).
        state: Equinox state for the target model.
        V: Conditioning variables of shape ``(N, D_v)``.
        X: Full covariates of shape ``(N, D_x)`` (for nuisance model).
        Y: Outcomes of shape ``(N, D_y)``.
        A: Treatment indicators of shape ``(N,)``.
        model_or: Trained nuisance NeuralKernelNet model.
        state_or: Equinox state for the nuisance model.
        pi: Propensity scores of shape ``(N,)``.
        X_ref: Reference covariates (unused for NK, accepted for API compat).
        Y_ref: Reference outcomes (unused for NK, accepted for API compat).

    Returns:
        Tuple of ``(loss, state)``.
    """
    # 1. Target Model (uses V)
    batch_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None, None, None))
    g, state, ypcl, sig = batch_model(V, state)

    # 2. Nuisance Model (uses full X)
    batch_model_or = jax.vmap(
        model_or, in_axes=(0, None), out_axes=(0, None, None, None)
    )
    f, _, _, _ = batch_model_or(X, state_or)

    K_MM = gram_matrix(ypcl, ypcl, sig)
    K_YM = gram_matrix(Y, ypcl, sig)

    term_norm = jnp.sum((g @ K_MM) * g)
    w_ipw = A / pi
    w_res = 1.0 - w_ipw

    dot_Y = jnp.sum(g * K_YM, axis=1)
    dot_nuis = jnp.sum((g @ K_MM) * f, axis=1)

    cross_term = w_ipw * dot_Y + w_res * dot_nuis

    loss = (term_norm - 2 * jnp.sum(cross_term)) / X.shape[0]
    return loss, state


def loss_df_dr(model, state, V, X, Y, A, model_or, state_or, pi, X_ref, Y_ref):
    """Doubly robust Deep Feature target loss (Stage 2).

    Constructs the DR projected kernel matrix from four terms (P1--P4) that
    combine IPW-weighted and plug-in components in feature space.

    Args:
        model: Target DeepFeatureNet model (uses V).
        state: Equinox state for the target model.
        V: Conditioning variables of shape ``(N, D_v)``.
        X: Full covariates of shape ``(N, D_x)``.
        Y: Outcomes of shape ``(N, D_y)``.
        A: Treatment indicators of shape ``(N,)``.
        model_or: Trained nuisance DeepFeatureNet model.
        state_or: Equinox state for the nuisance model.
        pi: Propensity scores of shape ``(N,)``.
        X_ref: Reference covariates from the nuisance split, shape ``(N_ref, D_x)``.
        Y_ref: Reference outcomes from the nuisance split, shape ``(N_ref, D_y)``.

    Returns:
        Tuple of ``(loss, state)``.
    """
    n = V.shape[0]

    # 1. Target Features (uses V)
    batch_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None, None, None))
    psi, state, lamb, sig = batch_model(V, state)

    # 2. Nuisance Features (uses full X)
    batch_model_or = jax.vmap(
        model_or, in_axes=(0, None), out_axes=(0, None, None, None)
    )
    psi0_ref, _, _, _ = batch_model_or(X_ref, state_or)
    psi0_target, _, _, _ = batch_model_or(X, state_or)

    # Nuisance Projection
    n_ref, dim0 = X_ref.shape[0], psi0_ref.shape[1]
    Cov0 = psi0_ref.T @ psi0_ref
    reg0 = lamb * jnp.eye(dim0)  # Unscaled lambda

    Cov0_inv = jnp.linalg.inv(Cov0 + reg0)

    K_Y = gram_matrix(Y, Y, sig, scaled=False)
    K_ref_target = gram_matrix(Y_ref, Y, sig, scaled=False)
    M_cross = psi0_target @ Cov0_inv @ (psi0_ref.T @ K_ref_target)

    w_vec = (A / pi).squeeze()
    D_w = w_vec[:, None]
    D_res = (1 - w_vec)[:, None]

    psi_w = psi * D_w
    P1 = psi_w.T @ K_Y @ psi_w

    psi_res = psi * D_res
    P2 = psi_w.T @ M_cross.T @ psi_res
    P3 = P2.T

    K_ref_ref = gram_matrix(Y_ref, Y_ref, sig, scaled=False)
    Core_inner = psi0_ref.T @ K_ref_ref @ psi0_ref
    Core = Cov0_inv @ Core_inner @ Cov0_inv

    Z = psi_res.T @ psi0_target
    P4 = Z @ Core @ Z.T

    K_proj = P1 + P2 + P3 + P4

    dim = psi.shape[1]
    reg = lamb * jnp.eye(dim)  # Unscaled lambda
    M_inv = jnp.linalg.inv(psi.T @ psi + reg)

    loss = -jnp.sum(K_proj * M_inv.T) / n
    return loss, state


def loss_nk_ipw(
    model, state, V, X, Y, A, model_or, state_or, pi, X_ref=None, Y_ref=None
):
    """Inverse probability weighted Neural-Kernel target loss (Stage 2).

    Uses only the IPW term (no nuisance plug-in correction).

    Args:
        model: Target NeuralKernelNet model.
        state: Equinox state for the target model.
        V: Conditioning variables of shape ``(N, D_v)``.
        X: Full covariates (unused, accepted for API compat).
        Y: Outcomes of shape ``(N, D_y)``.
        A: Treatment indicators of shape ``(N,)``.
        model_or: Nuisance model (unused).
        state_or: Nuisance state (unused).
        pi: Propensity scores of shape ``(N,)``.
        X_ref: Unused.
        Y_ref: Unused.

    Returns:
        Tuple of ``(loss, state)``.
    """
    batch_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None, None, None))
    g, state, ypcl, sig = batch_model(V, state)

    K_MM = gram_matrix(ypcl, ypcl, sig)
    K_YM = gram_matrix(Y, ypcl, sig)

    term_norm = jnp.sum((g @ K_MM) * g)
    w_ipw = A / pi
    dot_Y = jnp.sum(g * K_YM, axis=1)

    loss = (term_norm - 2 * jnp.sum(w_ipw * dot_Y)) / V.shape[0]
    return loss, state


def loss_nk_pi(
    model, state, V, X, Y, A, model_or, state_or, pi, X_ref=None, Y_ref=None
):
    """Plug-in Neural-Kernel target loss (Stage 2).

    Uses only the nuisance model predictions (no IPW correction).

    Args:
        model: Target NeuralKernelNet model.
        state: Equinox state for the target model.
        V: Conditioning variables of shape ``(N, D_v)``.
        X: Full covariates of shape ``(N, D_x)`` (for nuisance model).
        Y: Outcomes (unused in plug-in mode).
        A: Treatment indicators (unused).
        model_or: Trained nuisance NeuralKernelNet model.
        state_or: Equinox state for the nuisance model.
        pi: Propensity scores (unused).
        X_ref: Unused.
        Y_ref: Unused.

    Returns:
        Tuple of ``(loss, state)``.
    """
    batch_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None, None, None))
    g, state, ypcl, sig = batch_model(V, state)

    batch_model_or = jax.vmap(
        model_or, in_axes=(0, None), out_axes=(0, None, None, None)
    )
    f, _, _, _ = batch_model_or(X, state_or)

    K_MM = gram_matrix(ypcl, ypcl, sig)
    term_norm = jnp.sum((g @ K_MM) * g)
    dot_nuis = jnp.sum((g @ K_MM) * f, axis=1)

    loss = (term_norm - 2 * jnp.sum(dot_nuis)) / V.shape[0]
    return loss, state


def loss_df_ipw(model, state, V, X, Y, A, model_or, state_or, pi, X_ref, Y_ref):
    """Inverse probability weighted Deep Feature target loss (Stage 2).

    Args:
        model: Target DeepFeatureNet model.
        state: Equinox state for the target model.
        V: Conditioning variables of shape ``(N, D_v)``.
        X: Full covariates (unused).
        Y: Outcomes of shape ``(N, D_y)``.
        A: Treatment indicators of shape ``(N,)``.
        model_or: Nuisance model (unused).
        state_or: Nuisance state (unused).
        pi: Propensity scores of shape ``(N,)``.
        X_ref: Unused.
        Y_ref: Unused.

    Returns:
        Tuple of ``(loss, state)``.
    """
    n = V.shape[0]
    batch_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None, None, None))
    psi, state, lamb, sig = batch_model(V, state)

    K_Y = gram_matrix(Y, Y, sig, scaled=False)
    w_vec = (A / pi).squeeze()
    D_w = w_vec[:, None]

    psi_w = psi * D_w
    K_proj = psi_w.T @ K_Y @ psi_w

    dim = psi.shape[1]
    reg = lamb * jnp.eye(dim)
    M_inv = jnp.linalg.inv(psi.T @ psi + reg)
    return -jnp.sum(K_proj * M_inv.T) / n, state


def loss_df_pi(model, state, V, X, Y, A, model_or, state_or, pi, X_ref, Y_ref):
    """Plug-in Deep Feature target loss (Stage 2).

    Args:
        model: Target DeepFeatureNet model.
        state: Equinox state for the target model.
        V: Conditioning variables of shape ``(N, D_v)``.
        X: Full covariates of shape ``(N, D_x)``.
        Y: Outcomes (unused in plug-in mode).
        A: Treatment indicators (unused).
        model_or: Trained nuisance DeepFeatureNet model.
        state_or: Equinox state for the nuisance model.
        pi: Propensity scores (unused).
        X_ref: Reference covariates from the nuisance split.
        Y_ref: Reference outcomes from the nuisance split.

    Returns:
        Tuple of ``(loss, state)``.
    """
    n = V.shape[0]
    batch_model = jax.vmap(model, in_axes=(0, None), out_axes=(0, None, None, None))
    psi, state, lamb, sig = batch_model(V, state)

    batch_model_or = jax.vmap(
        model_or, in_axes=(0, None), out_axes=(0, None, None, None)
    )
    psi0_ref, _, _, _ = batch_model_or(X_ref, state_or)
    psi0_target, _, _, _ = batch_model_or(X, state_or)

    n_ref, dim0 = X_ref.shape[0], psi0_ref.shape[1]
    reg0 = lamb * jnp.eye(dim0)
    Cov0_inv = jnp.linalg.inv(psi0_ref.T @ psi0_ref + reg0)
    K_ref_ref = gram_matrix(Y_ref, Y_ref, sig, scaled=False)

    Core = Cov0_inv @ (psi0_ref.T @ K_ref_ref @ psi0_ref) @ Cov0_inv
    Z = psi.T @ psi0_target
    K_proj = Z @ Core @ Z.T

    dim = psi.shape[1]
    reg = lamb * jnp.eye(dim)
    M_inv = jnp.linalg.inv(psi.T @ psi + reg)

    return -jnp.sum(K_proj * M_inv.T) / n, state


def loss_pass(model, state, *args, **kwargs):
    """No-op loss that returns zero. Used when no training is needed (e.g. ridge)."""
    return jnp.array(0.0), state
