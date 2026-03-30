"""High-dimensional MNIST denoising experiment.

Demonstrates conditional density estimation on 784-dimensional image data.
A confounded treatment (denoising) is applied with probability depending on
image intensity.  Three estimators are compared:

1. **Oracle** -- trained on clean images with all units treated.
2. **DR** -- doubly robust estimator on the confounded data.
3. **One-Step** -- naive estimator using treated observations only.

For each digit (0--9), the empirical mode is selected and
saved to ``mnist_experiment_results.npz``.

Usage::

    python experiment_mnist.py method=nk
"""

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from sklearn.datasets import fetch_openml
from sklearn.metrics import pairwise_distances

from ccme import CCDEstimator


def median_heuristic(D):
    """Compute the median heuristic bandwidth for a dataset.

    Uses pairwise Euclidean distances on up to 1000 samples to estimate
    a kernel bandwidth.

    Args:
        D: Array of shape ``(N, D)`` with data points.

    Returns:
        Median of pairwise distances (float), or ``1.0`` if the median
        is zero.
    """
    n = min(len(D), 1000)
    subset = np.array(D[:n])
    dists = pairwise_distances(subset)
    sigma = np.median(dists)
    return float(sigma) if sigma > 0 else 1.0


def calculate_features(images, labels):
    """Compute conditioning features from MNIST images.

    Concatenates mean pixel intensity with a one-hot encoding of the
    digit label.

    Args:
        images: Array of shape ``(N, 28, 28)`` with pixel values.
        labels: Integer labels of shape ``(N,)``.

    Returns:
        Feature array of shape ``(N, 11)`` where column 0 is mean
        intensity and columns 1--10 are the one-hot label.
    """
    intensity = images.mean(axis=(1, 2)).reshape(-1, 1)

    one_hot = np.eye(10)[labels.astype(int)]

    return np.hstack([intensity, one_hot])


def get_mnist_data(n_samples=60000, seed=42):
    """Load MNIST and construct the confounded denoising dataset.

    Loads MNIST via OpenML, adds Gaussian noise to the images, and
    assigns a binary treatment (denoising) with probability determined
    by each image's mean intensity.  Treated images are replaced with
    their clean originals.

    Args:
        n_samples: Number of images to use (sampled without replacement).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of JAX arrays ``(V, Y, T, targets, Y_clean)`` where:

        - ``V``: Conditioning features of shape ``(n_samples, 11)``.
        - ``Y``: Observed (potentially noisy) images, shape ``(n_samples, 784)``.
        - ``T``: Binary treatment indicators, shape ``(n_samples,)``.
        - ``targets``: Digit labels, shape ``(n_samples,)``.
        - ``Y_clean``: Clean normalized images, shape ``(n_samples, 784)``.
    """
    print("Loading MNIST data...")
    mnist = fetch_openml(
        "mnist_784", version=1, cache=True, as_frame=False, parser="auto"
    )

    X_all = mnist.data.astype(np.float32)
    y_all = mnist.target.astype(int)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X_all))[:n_samples]
    X_raw_flat = X_all[indices]
    targets = y_all[indices]

    Y_clean = X_raw_flat / 255.0

    noise = rng.normal(0, 0.4, Y_clean.shape)
    Y_noisy = Y_clean + noise  # np.clip(Y_clean + noise, 0, 1)

    Y_noisy_images = Y_noisy.reshape(-1, 28, 28)
    V = calculate_features(Y_noisy_images, targets)

    intensity = V[:, 0]

    v_mean = np.quantile(intensity, 0.9)
    v_std = intensity.std()

    logits = 3.0 * (intensity - v_mean) / (v_std + 1e-6)
    prob_t = 1.0 / (1.0 + np.exp(-logits))
    print("Proportion of images with denoise probability > 0.5:", np.mean(prob_t > 0.5))
    prob_t = np.clip(prob_t, 0.01, 0.99)
    T = rng.binomial(1, prob_t)

    Y = Y_noisy.copy()
    Y[T == 1] = Y_clean[T == 1]

    return (
        jnp.array(V),
        jnp.array(Y),
        jnp.array(T),
        jnp.array(targets),
        jnp.array(Y_clean),
    )


def get_best_images(estimator, Y_clean):
    """Select the empirical mode image for each digit.

    For each digit 0--9, constructs a one-hot conditioning vector and
    evaluates the estimated density at every clean image.  The image
    with the highest density is selected.

    Args:
        estimator: A fitted :class:`~ccme.CCDEstimator`.
        Y_clean: Clean images of shape ``(N, 784)`` used as the
            evaluation grid.

    Returns:
        Array of shape ``(10, 28, 28)`` containing the best image for
        each digit.
    """
    best_images = []
    for digit in range(10):
        one_hot = [0.0] * 10
        one_hot[digit] = 1.0
        V_test = jnp.array(one_hot)

        pdf_matrix = estimator.predict(V_test, Y_clean, batch_size=10000)
        scores = pdf_matrix[0]
        max_idx = np.argmax(scores)

        best_images.append(Y_clean[max_idx].reshape(28, 28))

    return np.array(best_images)


@hydra.main(config_name="config", version_base=None, config_path="configs/mnist")
def run_mnist_experiment(cfg: DictConfig) -> None:
    """Run the full MNIST denoising experiment.

    Trains three estimators (oracle, DR, one-step) and saves the
    empirical mode for each digit to
    ``mnist_experiment_results.npz``.

    Args:
        cfg: Hydra configuration containing ``method``, ``seed``, and
            nested ``model`` / ``train`` / ``test`` sections.
    """
    print("Preparing Data...")
    X, Y, T, targets, Y_clean = get_mnist_data(n_samples=60000)

    V_context = X[:, 1:]

    rng = np.random.default_rng(cfg.seed)
    grid_idx = rng.choice(len(Y), cfg.model.output_dim, replace=False)
    y_grid = Y[grid_idx]

    final_results = {}

    print("--- Phase 1: Training on Clean Data (Oracle) ---")
    cfg.mode = "onestep"
    estimator_clean = CCDEstimator(cfg)
    T_all_ones = jnp.ones_like(T)
    estimator_clean.fit(X=X, V=V_context, Y=Y_clean, A=T_all_ones, grid_points=y_grid)
    final_results["clean"] = get_best_images(estimator_clean, Y_clean)

    print("--- Phase 2: Training with mode = dr ---")
    cfg.mode = "dr"
    estimator_dr = CCDEstimator(cfg)
    estimator_dr.fit(X=X, V=V_context, Y=Y, A=T, grid_points=y_grid)
    final_results["dr"] = get_best_images(estimator_dr, Y_clean)

    print("--- Phase 3: Training with mode = onestep ---")
    cfg.mode = "onestep"
    estimator_os = CCDEstimator(cfg)
    estimator_os.fit(X=X, V=V_context, Y=Y, A=T, grid_points=y_grid)
    final_results["one_step"] = get_best_images(estimator_os, Y_clean)

    save_path = "mnist_experiment_results.npz"
    np.savez(save_path, **final_results)
    print(f"All experiments complete. Results saved to {save_path}")


if __name__ == "__main__":
    run_mnist_experiment()
