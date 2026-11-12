"""Task 2 of the downstream causal tasks, for Table 3 of the paper.

Run all four scenarios from the repository root:
    python experiments/experiment_two_sample_test.py
Run only some scenarios, for example Scenarios 1 and 3:
    python experiments/experiment_two_sample_test.py --scenarios 1 3
Print Table 3 from the saved results without running any scenario:
    python experiments/experiment_two_sample_test.py --scenarios
"""

import argparse
import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from downstream import (
    cate_pseudo_outcomes,
    fit_propensity,
    gaussian_gram,
    generate_data,
    linear_smoother,
    select_bandwidths,
    split_units,
)

N_SAMPLES = 2000
N_TRIALS = 2000
RIDGE_LAMBDA = 1.0
V_GRID = np.linspace(-2.0, 2.0, 100)[:, None]
FIRST_SEEDS = {1: 460000, 2: 466000, 3: 464000, 4: 462000}
TARGET_LEVELS = {1: 0.05, 2: 0.05, 3: 1.0, 4: 1.0}
TESTS = ["CCME-DR", "One-Step", "CPME", "CATE-DR"]
CRITICAL_VALUE = norm.ppf(0.95)


def studentized(terms):
    """Compute the studentized mean of the terms f_i.

    Args:
        terms: Terms of shape ``(n,)``.

    Returns:
        The statistic sqrt(n) * mean / standard deviation, as a float.
    """
    return np.sqrt(terms.size) * terms.mean() / terms.std()


def ridge_smoother(gram, cross, units):
    """Compute kernel ridge regression weights on a subset of the training points.

    Args:
        gram: Gram matrix of the training points, shape ``(N, N)``.
        cross: Gram matrix between the training points and the M evaluation points,
            shape ``(N, M)``.
        units: Indices of the training points to use.

    Returns:
        Weights of shape ``(len(units), M)``.
    """
    regularized = gram[np.ix_(units, units)] + RIDGE_LAMBDA * np.eye(units.size)
    return np.linalg.solve(regularized, cross[units])


def ridge_weights(v, units, sigma_v):
    """Compute the kernel ridge regression weights of the units on V.

    Args:
        v: Conditioning variable of shape ``(N, 1)``.
        units: Indices of the units.
        sigma_v: Bandwidth on V.

    Returns:
        Weights of shape ``(len(units), len(V_GRID))``. Column j holds the weights for
        the j-th point of V_GRID.
    """
    gram = gaussian_gram(v[units], v[units], sigma_v)
    cross = gaussian_gram(v[units], V_GRID, sigma_v)
    return np.linalg.solve(gram + RIDGE_LAMBDA * np.eye(units.size), cross)


def one_step_weights(v, a, units, sigma_v):
    """Compute the One-Step weights of the units on V.

    The weights of the treated units are their kernel ridge regression weights within
    the treated units. The weights of the control units are the negatives of their
    weights within the control units. Thus, the weighted sum of phi(Y) is the One-Step
    estimate of the difference of the two conditional mean embeddings.

    Args:
        v: Conditioning variable of shape ``(N, 1)``.
        a: Binary treatment indicators of shape ``(N,)``.
        units: Indices of the units.
        sigma_v: Bandwidth on V.

    Returns:
        Weights of shape ``(len(units), len(V_GRID))``.
    """
    weights = np.zeros((units.size, V_GRID.shape[0]))
    for group, sign in ((1, 1.0), (0, -1.0)):
        members = np.where(a[units] == group)[0]
        weights[members] = sign * ridge_weights(v, units[members], sigma_v)
    return weights


def kernel_statistics(x, v, y, a, nuisance, target, bandwidths):
    """Compute the statistics T of the CCME-DR, One-Step and CPME tests.

    The pseudo-outcome of each unit in D_1 is a linear combination of phi(Y_1), ...,
    phi(Y_n), with the weights of the i-th unit in column i of ``coefficients``. If K_Y
    is the Gram matrix of Y, the inner products of the pseudo-outcomes are the entries
    of ``coefficients.T @ K_Y @ coefficients``.

    Args:
        x: Covariates of shape ``(N, D_x)``.
        v: Conditioning variable of shape ``(N, 1)``.
        y: Outcomes of shape ``(N,)``.
        a: Binary treatment indicators of shape ``(N,)``.
        nuisance: Indices of the units of D_0.
        target: Indices of the units of D_1, whose two halves are I_1 and I_2.
        bandwidths: Bandwidths from :func:`downstream.select_bandwidths`.

    Returns:
        List of the three statistics, in the order CCME-DR, One-Step, CPME.
    """
    sigma_x, sigma_v, sigma_y = bandwidths["x"], bandwidths["v"], bandwidths["y"]
    half = target.size // 2
    first, second = target[:half], target[half:]
    pi = fit_propensity(x[nuisance], a[nuisance], x[target])
    weight1 = a[target] / pi
    weight0 = (1 - a[target]) / (1 - pi)
    gram_x = gaussian_gram(x[nuisance], x[nuisance], sigma_x)
    cross_x = gaussian_gram(x[nuisance], x[target], sigma_x)
    treated = np.where(a[nuisance] == 1)[0]
    control = np.where(a[nuisance] == 0)[0]
    coefficients = np.zeros((y.size, target.size))
    coefficients[target, np.arange(target.size)] = weight1 - weight0
    coefficients[nuisance[treated], :] += (
        ridge_smoother(gram_x, cross_x, treated) * (1 - weight1)[None, :]
    )
    coefficients[nuisance[control], :] -= (
        ridge_smoother(gram_x, cross_x, control) * (1 - weight0)[None, :]
    )
    gram_y = gaussian_gram(y, y, sigma_y)
    inner = (coefficients.T @ gram_y @ coefficients)[:half, half:]
    gram_y_halves = gram_y[np.ix_(target, target)][:half, half:]

    ccme_first = ridge_weights(v, first, sigma_v)
    ccme_second = ridge_weights(v, second, sigma_v)
    ccme_terms = np.mean(ccme_first * (inner @ ccme_second), axis=1)
    one_step_first = one_step_weights(v, a, first, sigma_v)
    one_step_second = one_step_weights(v, a, second, sigma_v)
    one_step_terms = np.mean(one_step_first * (gram_y_halves @ one_step_second), axis=1)
    cpme_terms = inner.mean(axis=1)
    return [studentized(terms) for terms in (ccme_terms, one_step_terms, cpme_terms)]


def cate_statistic(x, y, a, nuisance, target):
    """Compute the statistic T of the CATE-DR test.

    Args:
        x: Covariates of shape ``(N, D_x)``, whose first column is V.
        y: Outcomes of shape ``(N,)``.
        a: Binary treatment indicators of shape ``(N,)``.
        nuisance: Indices of the units of D_0.
        target: Indices of the units of D_1, whose two halves are I_1 and I_2.

    Returns:
        The statistic T as a float.
    """
    half = target.size // 2
    pseudo_outcomes = cate_pseudo_outcomes(x, y, a, nuisance, target)
    smoother_first = linear_smoother(x[target[:half], 0], V_GRID.ravel())
    smoother_second = linear_smoother(x[target[half:], 0], V_GRID.ravel())
    fit_second = smoother_second @ pseudo_outcomes[half:]
    products = smoother_first * pseudo_outcomes[None, :half] * fit_second[:, None]
    return studentized(products.mean(axis=0))


def result_path(scenario):
    """Return the name of the results file of a scenario.

    Args:
        scenario: Scenario number, 1, 2, 3 or 4.

    Returns:
        The file name ``two_sample_test_scenario_<scenario>.npz``.
    """
    return f"two_sample_test_scenario_{scenario}.npz"


def run_scenario(scenario, n_trials):
    """Run the four tests on n_trials samples of a scenario and save the statistics.

    Args:
        scenario: Scenario number, 1, 2, 3 or 4.
        n_trials: Number of samples.
    """
    statistics = np.zeros((n_trials, len(TESTS)))
    seeds = FIRST_SEEDS[scenario] + np.arange(n_trials)
    for trial in tqdm(range(n_trials), desc=f"Scenario {scenario}"):
        seed = int(seeds[trial])
        x, y, a, v = generate_data(N_SAMPLES, seed, scenario)
        nuisance, target, _, _ = split_units(N_SAMPLES, trial)
        bandwidths = select_bandwidths(
            x[nuisance], v[nuisance], y[nuisance], a[nuisance], RIDGE_LAMBDA, seed
        )
        statistics[trial, :3] = kernel_statistics(
            x, v, y, a, nuisance, target, bandwidths
        )
        statistics[trial, 3] = cate_statistic(x, y, a, nuisance, target)
    np.savez(result_path(scenario), statistics=statistics, tests=TESTS, seeds=seeds)
    print(f"Saved {result_path(scenario)}")


def format_cell(count, n_trials, target_level):
    """Format a rejection rate and its standard error as a LaTeX table cell.

    The rate is rounded half up to three decimals. The cell is bold if the 95%
    confidence interval of the rate covers the target level.

    Args:
        count: Number of rejections.
        n_trials: Number of trials.
        target_level: Target rejection rate of the scenario.

    Returns:
        The LaTeX cell as a string.
    """
    rate = count / n_trials
    standard_error = np.sqrt(rate * (1 - rate) / n_trials)
    rounded_rate = (Decimal(count) / Decimal(n_trials)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    )
    cell = f"{rounded_rate} \\pm {standard_error:.3f}"
    if abs(rate - target_level) <= 1.96 * standard_error:
        return f"$\\bm{{{cell}}}$"
    return f"${cell}$"


def print_table():
    """Print Table 3 of the paper in LaTeX from the saved results."""
    scenarios = sorted(FIRST_SEEDS)
    statistics = {k: np.load(result_path(k))["statistics"] for k in scenarios}
    header = " & ".join(rf"\textbf{{Scenario {k}}}" for k in scenarios)
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        rf"\textbf{{Test}} & {header} \\",
        r"\midrule",
    ]
    for j, test in enumerate(TESTS):
        cells = []
        for k in scenarios:
            rejections = statistics[k][:, j] >= CRITICAL_VALUE
            count = int(rejections.sum())
            cells.append(format_cell(count, rejections.size, TARGET_LEVELS[k]))
        lines.append(f"{test} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    print("\n".join(lines))


def main():
    """Run the scenarios given on the command line and print Table 3."""
    parser = argparse.ArgumentParser()
    scenarios = sorted(FIRST_SEEDS)
    parser.add_argument(
        "--scenarios", type=int, nargs="*", choices=scenarios, default=scenarios
    )
    parser.add_argument("--trials", type=int, default=N_TRIALS)
    args = parser.parse_args()
    for scenario in args.scenarios:
        run_scenario(scenario, args.trials)
    if all(os.path.exists(result_path(k)) for k in scenarios):
        print_table()


if __name__ == "__main__":
    main()
