"""Shared code for the downstream causal tasks of the paper."""

import numpy as np
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

PROPENSITY_CLIP = 0.05
OUTCOME_SD = np.sqrt(0.8**2 + 0.7**2)
LOG2_BANDWIDTH_GRIDS = {
    "x": np.arange(-4.0, 2.0),
    "v": np.arange(-5.0, 6.0),
    "y": np.arange(-5.0, 2.0),
}
SPLIT_SALT = 7919


def propensity(v, u):
    """Compute the true propensity score P(A = 1 | X).

    Args:
        v: Values of V, shape ``(N,)``.
        u: Values of U, shape ``(N,)``.

    Returns:
        Propensity scores sigmoid(0.5 V + 1.1 U), truncated to [0.05, 0.95], shape
        ``(N,)``.
    """
    return np.clip(expit(0.5 * v + 1.1 * u), PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)


def delta(v):
    """Compute the shift of the two components of Y^1 given V = v.

    The shift is 2.2 + 0.9 v.

    Args:
        v: Values of V.

    Returns:
        Array of the same shape as ``v``.
    """
    return 2.2 + 0.9 * v


def generate_data(n_samples, seed, scenario=None):
    """Draw a sample from the model of Tasks 1 and 3 or from a scenario of Task 2.

    Args:
        n_samples: Sample size N.
        seed: Seed of the NumPy random generator.
        scenario: ``None`` for the model of Tasks 1 and 3, or 1, 2, 3 or 4 for the
            scenarios of Task 2.

    Returns:
        Tuple ``(X, Y, A, V)`` where:

        - ``X``: Covariates (V, U) of shape ``(N, 2)``.
        - ``Y``: Observed outcomes of shape ``(N,)``.
        - ``A``: Binary treatment indicators of shape ``(N,)``.
        - ``V``: Conditioning variable of shape ``(N, 1)``.
    """
    rng = np.random.default_rng(seed)
    v = rng.uniform(-2.0, 2.0, size=n_samples)
    u = rng.standard_normal(n_samples)
    a = rng.binomial(1, propensity(v, u))
    if scenario == 2:
        y1 = v + 0.8 * u + 0.7 * rng.standard_normal(n_samples)
        y0 = v - 0.8 * u + 0.7 * rng.standard_normal(n_samples)
    else:
        s = rng.choice(np.array([-1.0, 1.0]), size=n_samples)
        noise0 = rng.standard_normal(n_samples)
        noise1 = rng.standard_normal(n_samples)
        if scenario == 3:
            w = expit(2.0 * v)
            y0 = 0.8 * u + s * 3.0 * w + 0.6 * noise0
            y1 = 0.8 * u + s * 3.0 * (1.0 - w) + 0.6 * noise1
        else:
            base = v + 0.8 * u
            y0 = base + 0.7 * noise0
            if scenario is None:
                y1 = base + 1.5 + s * delta(v) + 0.7 * noise1
            elif scenario == 1:
                y1 = base + 0.7 * noise1
            else:
                y1 = base + s * delta(v) + 0.7 * noise1
    y = a * y1 + (1 - a) * y0
    return np.stack([v, u], axis=1), y, a, v[:, None]


def true_density(v, y, treated):
    """Compute the true conditional density of Y^1 or Y^0 given V = v.

    Args:
        v: Value of V (scalar).
        y: Points at which to evaluate the density, shape ``(G,)``.
        treated: If True, use Y^1. Otherwise, use Y^0.

    Returns:
        Density values of shape ``(G,)``.
    """
    if treated:
        center = v + 1.5
        lower = stats.norm.pdf(y, center - delta(v), OUTCOME_SD)
        return 0.5 * lower + 0.5 * stats.norm.pdf(y, center + delta(v), OUTCOME_SD)
    return stats.norm.pdf(y, v, OUTCOME_SD)


def true_cdf(v, y, treated):
    """Compute the true conditional distribution function of Y^1 or Y^0 given V = v.

    Args:
        v: Value of V (scalar).
        y: Points at which to evaluate the distribution function, shape ``(G,)``.
        treated: If True, use Y^1. Otherwise, use Y^0.

    Returns:
        Values of the distribution function, shape ``(G,)``.
    """
    if treated:
        center = v + 1.5
        lower = stats.norm.cdf(y, center - delta(v), OUTCOME_SD)
        return 0.5 * lower + 0.5 * stats.norm.cdf(y, center + delta(v), OUTCOME_SD)
    return stats.norm.cdf(y, v, OUTCOME_SD)


def true_cvar(v, alpha, treated):
    """Compute the true conditional value-at-risk (CVaR) of Y^1 or Y^0 given V = v.

    The CVaR at level alpha is the mean of the lower alpha-tail of the conditional
    distribution. It is computed on a grid of 200001 points in [-20, 25].

    Args:
        v: Value of V (scalar).
        alpha: Tail level in (0, 1).
        treated: If True, use Y^1. Otherwise, use Y^0.

    Returns:
        The CVaR as a float.
    """
    grid = np.linspace(-20.0, 25.0, 200001)
    quantile = np.interp(alpha, true_cdf(v, grid, treated), grid)
    mask = grid <= quantile
    total = np.sum(grid[mask] * true_density(v, grid, treated)[mask])
    return float(total * (grid[1] - grid[0]) / alpha)


def squared_distances(a, b):
    """Compute the squared Euclidean distances between two sets of points.

    Args:
        a: First set of points, shape ``(N, D)`` or ``(N,)``.
        b: Second set of points, shape ``(M, D)`` or ``(M,)``.

    Returns:
        Matrix of squared distances of shape ``(N, M)``.
    """
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    return np.sum((a[:, None, :] - b[None, :, :]) ** 2, -1)


def gaussian_gram(a, b, sigma):
    """Compute the Gaussian Gram matrix between two sets of points.

    Computes K_{ij} = exp(-||a_i - b_j||^2 / (2 * sigma^2)).

    Args:
        a: First set of points, shape ``(N, D)`` or ``(N,)``.
        b: Second set of points, shape ``(M, D)`` or ``(M,)``.
        sigma: Kernel bandwidth.

    Returns:
        Gram matrix of shape ``(N, M)``.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a[:, None] if a.ndim == 1 else a
    b = b[:, None] if b.ndim == 1 else b
    sq = np.sum(a**2, 1)[:, None] + np.sum(b**2, 1)[None, :] - 2 * a @ b.T
    return np.exp(-np.maximum(sq, 0.0) / (2 * sigma**2))


def median_distance(z, rng, max_points=1500):
    """Compute the median of the pairwise distances between the rows of z.

    Args:
        z: Points of shape ``(N, D)`` or ``(N,)``.
        rng: NumPy random generator, used to subsample the rows when N > max_points.
        max_points: Maximum number of rows to use.

    Returns:
        The median pairwise distance as a float, or 1.0 if the median is at most
        ``1e-6``.
    """
    z = np.asarray(z, float)
    z = z[:, None] if z.ndim == 1 else z
    if z.shape[0] > max_points:
        z = z[rng.choice(z.shape[0], max_points, replace=False)]
    sq = squared_distances(z, z)[np.triu_indices(z.shape[0], 1)]
    median = np.median(np.sqrt(np.maximum(sq, 0.0)))
    return float(median) if median > 1e-6 else 1.0


def ridge_weights(sq_train, sq_cross, sigma, lamb):
    """Compute kernel ridge regression weights from squared distances.

    Args:
        sq_train: Squared distances between the N training points, shape ``(N, N)``.
        sq_cross: Squared distances between the training points and the M evaluation
            points, shape ``(N, M)``.
        sigma: Bandwidth of the Gaussian kernel.
        lamb: Regularization added to the diagonal of the Gram matrix.

    Returns:
        Weights of shape ``(N, M)``. Column j holds the weights of the training points
        for the j-th evaluation point.
    """
    gram = np.exp(-sq_train / (2 * sigma**2))
    gram[np.diag_indices_from(gram)] += lamb
    return np.linalg.solve(gram, np.exp(-sq_cross / (2 * sigma**2)))


def density_loss(weights, sq_y_train, sq_y_valid, sigma_y):
    """Compute the held-out squared L2 error of the density of Y given V.

    At each validation point, the density estimate is the normalized weighted sum of
    Gaussian kernels with bandwidth sigma_y, centered at the training outcomes. The
    error is computed up to an additive constant that does not depend on the
    bandwidths.

    Args:
        weights: Ridge weights on V of the training points, shape ``(N, M)``.
        sq_y_train: Squared distances between the training outcomes, shape ``(N, N)``.
        sq_y_valid: Squared distances between the training and validation outcomes,
            shape ``(N, M)``.
        sigma_y: Bandwidth on Y.

    Returns:
        The error averaged over the M validation points, as a float.
    """
    sums = weights.sum(0)
    sums = np.where(np.abs(sums) < 1e-8, 1e-8, sums)
    normalized = weights / sums
    overlap = np.exp(-sq_y_train / (4 * sigma_y**2)) / np.sqrt(4 * np.pi * sigma_y**2)
    kernel = np.exp(-sq_y_valid / (2 * sigma_y**2)) / np.sqrt(2 * np.pi * sigma_y**2)
    squared_norms = np.sum((normalized.T @ overlap) * normalized.T, 1)
    return float(np.mean(squared_norms - 2 * np.sum(normalized * kernel, 0)))


def rkhs_loss(weights, sq_y_train, sq_y_valid, sigma_y):
    """Compute the held-out squared RKHS error of the regression of phi(Y) on X.

    Args:
        weights: Ridge weights on X of the training points, shape ``(N, M)``.
        sq_y_train: Squared distances between the training outcomes, shape ``(N, N)``.
        sq_y_valid: Squared distances between the training and validation outcomes,
            shape ``(N, M)``.
        sigma_y: Bandwidth on Y.

    Returns:
        The error averaged over the M validation points, as a float.
    """
    gram = np.exp(-sq_y_train / (2 * sigma_y**2))
    cross = np.exp(-sq_y_valid / (2 * sigma_y**2))
    squared_norms = np.sum((weights.T @ gram) * weights.T, 1)
    return float(np.mean(1.0 - 2 * np.sum(weights * cross, 0) + squared_norms))


def select_bandwidths(x, v, y, a, lamb, seed):
    """Select the bandwidths on X, V and Y by minimizing held-out errors.

    The estimators are fitted on a random 80% of the units and evaluated on the other
    20%. Each bandwidth is chosen from the median pairwise distance times powers of two.
    The bandwidths on V and Y minimize the error of the density of Y given V. With the
    bandwidth on Y fixed, the bandwidth on X minimizes the error of the regression of
    phi(Y) on X. Both errors are summed over the treated and control groups.

    Args:
        x: Covariates of shape ``(N, D_x)``.
        v: Conditioning variable of shape ``(N, D_v)`` or ``(N,)``.
        y: Outcomes of shape ``(N,)``.
        a: Binary treatment indicators of shape ``(N,)``.
        lamb: Ridge regularization.
        seed: Seed of the random split and of the subsampling of the units.

    Returns:
        Dictionary with the bandwidths on X, V and Y under the keys ``"x"``, ``"v"``
        and ``"y"``.
    """
    x = np.asarray(x, float)
    v = np.asarray(v, float).reshape(len(x), -1)
    y = np.asarray(y, float).ravel()
    a = np.asarray(a, int).ravel()
    rng = np.random.default_rng([int(seed), SPLIT_SALT])
    order = rng.permutation(y.size)
    n_train = int(round(0.8 * y.size))
    train, valid = order[:n_train], order[n_train:]
    medians = {
        "x": median_distance(x, rng),
        "v": median_distance(v, rng),
        "y": median_distance(y, rng),
    }
    grids = {key: medians[key] * 2.0 ** LOG2_BANDWIDTH_GRIDS[key] for key in medians}
    groups = []
    for group in (0, 1):
        train_group, valid_group = train[a[train] == group], valid[a[valid] == group]
        sq_y_train = squared_distances(y[train_group], y[train_group])
        sq_y_valid = squared_distances(y[train_group], y[valid_group])
        groups.append((train_group, valid_group, sq_y_train, sq_y_valid))
    loss_vy = np.zeros((grids["v"].size, grids["y"].size))
    for train_group, valid_group, sq_y_train, sq_y_valid in groups:
        sq_v_train = squared_distances(v[train_group], v[train_group])
        sq_v_valid = squared_distances(v[train_group], v[valid_group])
        for i, sigma_v in enumerate(grids["v"]):
            weights = ridge_weights(sq_v_train, sq_v_valid, sigma_v, lamb)
            for j, sigma_y in enumerate(grids["y"]):
                loss_vy[i, j] += density_loss(weights, sq_y_train, sq_y_valid, sigma_y)
    i, j = np.unravel_index(int(np.argmin(loss_vy)), loss_vy.shape)
    sigma_v, sigma_y = float(grids["v"][i]), float(grids["y"][j])
    loss_x = np.zeros(grids["x"].size)
    for train_group, valid_group, sq_y_train, sq_y_valid in groups:
        sq_x_train = squared_distances(x[train_group], x[train_group])
        sq_x_valid = squared_distances(x[train_group], x[valid_group])
        for k, sigma_x in enumerate(grids["x"]):
            weights = ridge_weights(sq_x_train, sq_x_valid, sigma_x, lamb)
            loss_x[k] += rkhs_loss(weights, sq_y_train, sq_y_valid, sigma_y)
    return {"x": float(grids["x"][int(np.argmin(loss_x))]), "v": sigma_v, "y": sigma_y}


def split_units(n_samples, seed):
    """Randomly split the units into D_0 and D_1, and split D_1 into two halves.

    Args:
        n_samples: Number of units N.
        seed: Seed of the NumPy random generator.

    Returns:
        Tuple of index arrays ``(nuisance, target, first_half, second_half)``, where
        ``nuisance`` indexes D_0, ``target`` indexes D_1, and the last two arrays
        index the two halves of D_1.
    """
    order = np.random.default_rng(seed).permutation(n_samples)
    nuisance, target = order[: n_samples // 2], order[n_samples // 2 :]
    half = target.size // 2
    return nuisance, target, target[:half], target[half:]


def fit_propensity(x_fit, a_fit, x_eval):
    """Estimate the propensity score with an unpenalized logistic regression.

    Args:
        x_fit: Covariates used for fitting, shape ``(N, D_x)``.
        a_fit: Binary treatment indicators used for fitting, shape ``(N,)``.
        x_eval: Covariates at which to evaluate the propensity score, shape
            ``(M, D_x)``.

    Returns:
        Estimated propensity scores, truncated to [0.05, 0.95], shape ``(M,)``.
    """
    model = LogisticRegression(C=np.inf, max_iter=1000).fit(x_fit, a_fit)
    pi = model.predict_proba(x_eval)[:, 1]
    return np.clip(pi, PROPENSITY_CLIP, 1 - PROPENSITY_CLIP)


def cate_pseudo_outcomes(x, y, a, nuisance, target):
    """Compute the doubly robust pseudo-outcomes for Y^1 - Y^0 on the units of D_1.

    The propensity score and the linear regressions of Y on X in the treated and
    control groups are fitted on the units of D_0.

    Args:
        x: Covariates of shape ``(N, D_x)``.
        y: Outcomes of shape ``(N,)``.
        a: Binary treatment indicators of shape ``(N,)``.
        nuisance: Indices of the units of D_0.
        target: Indices of the units of D_1.

    Returns:
        Pseudo-outcomes of shape ``(len(target),)``, in the order of ``target``.
    """
    pi = fit_propensity(x[nuisance], a[nuisance], x[target])
    design = np.column_stack([np.ones(x.shape[0]), x])
    means = {}
    for group in (1, 0):
        units = nuisance[a[nuisance] == group]
        coefficients = np.linalg.lstsq(design[units], y[units], rcond=None)[0]
        means[group] = design[target] @ coefficients
    a_target, y_target = a[target], y[target]
    treated_term = a_target * (y_target - means[1]) / pi
    control_term = (1 - a_target) * (y_target - means[0]) / (1 - pi)
    return means[1] - means[0] + treated_term - control_term


def linear_smoother(v, v_grid):
    """Compute the matrix that maps responses at v to a least-squares line at v_grid.

    If W is the returned matrix and r holds responses at the points v, then W @ r holds
    the values at v_grid of the least-squares line fitted to the pairs (v, r).

    Args:
        v: Points of shape ``(N,)``.
        v_grid: Evaluation points of shape ``(L,)``.

    Returns:
        Matrix W of shape ``(L, N)``.
    """
    design = np.column_stack([np.ones(v.size), v])
    grid_design = np.column_stack([np.ones(v_grid.size), v_grid])
    return grid_design @ np.linalg.solve(design.T @ design, design.T)
