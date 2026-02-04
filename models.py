"""Neural network architectures for conditional density estimation.

Defines three model classes used by :class:`ccme.CCDEstimator`:

* :class:`NeuralKernelNet` -- grid-based neural kernel estimator (method ``"nk"``).
* :class:`DeepFeatureNet` -- learned feature map estimator (method ``"df"``).
* :class:`RidgeModel` -- parameter container for kernel ridge regression (method ``"ridge"``).

All models follow a common ``__call__`` signature::

    (features, state, lamb_or_ypcl, sigma) = model(x, state)

so they can be used interchangeably by the training and inference routines.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


class BaseMLP(eqx.Module):
    """Feedforward multi-layer perceptron with ReLU activations.

    Args:
        key: JAX PRNG key for weight initialization.
        dims: List of layer dimensions, e.g. ``[in_dim, 64, 64, out_dim]``.
    """

    layers: list

    def __init__(self, key, dims):
        keys = jax.random.split(key, len(dims) - 1)
        self.layers = []
        for i in range(len(dims) - 1):
            self.layers.append(eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i]))

    def __call__(self, x):
        """Forward pass. Applies ReLU after every layer except the last.

        Args:
            x: Input tensor of shape ``(in_dim,)``.

        Returns:
            Output tensor of shape ``(out_dim,)``.
        """
        for layer in self.layers[:-1]:
            x = jax.nn.relu(layer(x))
        return self.layers[-1](x)


class DeepFeatureNet(eqx.Module):
    """Deep Feature model that learns a feature map for ridge regression in
    feature space.

    The network maps input covariates to a learned feature vector ``psi(x)``.
    Density estimation is performed by solving a ridge regression problem in the
    feature space spanned by ``psi``.

    Args:
        key: JAX PRNG key.
        in_dim: Input dimensionality.
        hidden_dim: List of hidden layer widths, e.g. ``[64, 64]``.
        out_dim: Output feature dimensionality.
        ypcl: Grid points (unused, accepted for API compatibility).
        sigma_init: Initial kernel bandwidth (stored, not learned).
        lamb: Ridge regularization parameter.
        learn_sig: Unused (bandwidth is not learned for this model).
    """

    net: BaseMLP
    lamb: float
    sig_param: float
    learn_sig: bool

    def __init__(
        self, key, in_dim, hidden_dim, out_dim, ypcl, sigma_init, lamb, learn_sig=False
    ):
        self.net = BaseMLP(key, [in_dim] + hidden_dim + [out_dim])
        self.lamb = lamb
        self.sig_param = sigma_init
        self.learn_sig = learn_sig

    def __call__(self, x, state):
        """Compute feature representation.

        Args:
            x: Input of shape ``(in_dim,)``.
            state: Equinox state object (passed through unchanged).

        Returns:
            Tuple of ``(features, state, lamb, sigma)`` where ``lamb`` and
            ``sigma`` are stop-gradiented scalars.
        """
        features = self.net(x)

        return (
            features,
            state,
            jax.lax.stop_gradient(self.lamb),
            jax.lax.stop_gradient(self.sig_param),
        )


class NeuralKernelNet(eqx.Module):
    """Neural Kernel model that outputs coefficients over a fixed grid of
    outcome points.

    The network maps input covariates to a coefficient vector ``g(x)`` of length
    ``num_grid``.  The density estimate at outcome ``y`` is formed as
    ``sum_m g_m * K(y_m, y)`` where ``y_m`` are fixed grid (inducing) points.

    Args:
        key: JAX PRNG key.
        in_dim: Input dimensionality.
        hidden_dim: List of hidden layer widths.
        num_grid: Number of grid (inducing) points (output dimension).
        ypcl: Grid points of shape ``(num_grid, D_y)``.
        sigma_init: Initial kernel bandwidth. Stored via the inverse-softplus
            transform so that ``softplus(sig_param) == sigma_init``.
        lamb: Unused (accepted for API compatibility).
        learn_sig: If True, ``sigma`` is trainable; otherwise it is
            stop-gradiented during training.
    """

    net: BaseMLP
    ypcl: jax.Array
    sig_param: jax.Array
    learn_sig: bool

    def __init__(
        self,
        key,
        in_dim,
        hidden_dim,
        num_grid,
        ypcl,
        sigma_init,
        lamb=None,
        learn_sig=True,
    ):
        self.net = BaseMLP(key, [in_dim] + hidden_dim + [num_grid])
        self.ypcl = ypcl
        self.sig_param = jnp.log(jnp.expm1(sigma_init))
        self.learn_sig = learn_sig

    def __call__(self, x, state):
        """Compute grid coefficients.

        Args:
            x: Input of shape ``(in_dim,)``.
            state: Equinox state object (passed through unchanged).

        Returns:
            Tuple of ``(coefficients, state, ypcl, sigma)`` where ``ypcl`` is
            the stop-gradiented grid and ``sigma`` is optionally trainable.
        """
        out = self.net(x)

        sig_val = jax.nn.softplus(self.sig_param)
        if not self.learn_sig:
            sig_val = jax.lax.stop_gradient(sig_val)

        return out, state, jax.lax.stop_gradient(self.ypcl), sig_val


class RidgeModel(eqx.Module):
    """Parameter container for kernel ridge regression.

    This model holds only the regularization and bandwidth parameters.  Its
    ``__call__`` returns the input unchanged (identity mapping), so that the
    ridge regression estimator operates directly in the input space.

    Args:
        key: JAX PRNG key (unused, accepted for API compatibility).
        in_dim: Input dimensionality (unused).
        hidden_dim: Hidden layer widths (unused).
        out_dim: Output dimensionality (unused).
        ypcl: Grid points (unused).
        sigma_init: Initial kernel bandwidth.
        lamb: Ridge regularization parameter.
        learn_sig: Unused.
    """

    lamb: jax.Array
    sig_param: jax.Array

    def __init__(
        self, key, in_dim, hidden_dim, out_dim, ypcl, sigma_init, lamb, learn_sig=False
    ):
        self.lamb = lamb
        self.sig_param = jnp.log(jnp.expm1(sigma_init))

    def __call__(self, x, state):
        """Identity mapping that passes through bandwidth and regularization.

        Args:
            x: Input of shape ``(in_dim,)``.
            state: Equinox state object (passed through unchanged).

        Returns:
            Tuple of ``(x, state, lamb, sigma)``.
        """
        return x, state, self.lamb, jax.nn.softplus(self.sig_param)
