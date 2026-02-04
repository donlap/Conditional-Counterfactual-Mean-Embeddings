"""Gaussian RBF kernel functions for kernel mean embedding estimators."""

import jax.numpy as jnp


def gram_matrix(X, Y, sigma, scaled=False):
    """Compute the Gaussian RBF Gram matrix between two sets of points.

    Computes K_{ij} = exp(-||X_i - Y_j||^2 / (2 * sigma^2)).

    Args:
        X: First set of points, shape (N, D) or (N,).
        Y: Second set of points, shape (M, D) or (M,).
        sigma: Kernel bandwidth parameter (scalar).
        scaled: If True, normalizes by 1 / (sigma * sqrt(2 * pi)) to produce
            a density-like kernel evaluation.

    Returns:
        Gram matrix of shape (N, M).
    """
    # Ensure 2D (N, D)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]

    X_sq = jnp.sum(X**2, axis=1, keepdims=True)
    Y_sq = jnp.sum(Y**2, axis=1, keepdims=True)
    D2 = X_sq + Y_sq.T - 2 * jnp.dot(X, Y.T)

    K = jnp.exp(-D2 / (2 * sigma**2))

    if scaled:
        scale = jnp.sqrt(2 * jnp.pi * sigma**2)
        K = K / scale

    return K
