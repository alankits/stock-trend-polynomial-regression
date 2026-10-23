"""
hypothesis_tests.py - STEP 7: Statistical validation of the model (Unit V, CO5).

WHAT: confidence interval, t-test, F-test, chi-square goodness of fit and
chi-square test of independence.
WHY: a model is only trustworthy if its errors behave the way the theory
assumes (centred on zero, Normal) and if it really beats simpler models.
"""
import numpy as np
from scipy import stats
import config


def print_result(name, h0, h1, statistic, p_value, if_reject, if_not_reject):
    """Print one test in the same format every time and return the decision.

    Rule: if p-value < ALPHA, the result is too unlikely under H0, so reject H0.
    """
    decision = "Reject H0" if p_value < config.ALPHA else "Fail to reject H0"
    print(f"\n--- {name} ---")
    print(f"H0: {h0}")
    print(f"H1: {h1}")
    print(f"Test statistic = {statistic:.4f}, p-value = {p_value:.4f}")
    print(f"Decision at alpha = {config.ALPHA}: {decision}")
    print(f"Conclusion: {if_reject if decision == 'Reject H0' else if_not_reject}")
    return decision


def confidence_interval_mean(returns):
    """95% confidence interval for the mean daily return: x̄ ± t × s/√n."""
    n, mean, std = len(returns), returns.mean(), returns.std(ddof=1)
    t_critical = stats.t.ppf(1 - config.ALPHA / 2, df=n - 1)
    margin = t_critical * std / np.sqrt(n)
    print(f"\n95% CI for mean daily return: {mean:.4f} ± {margin:.4f} "
          f"= ({mean - margin:.4f}%, {mean + margin:.4f}%)")
    t_stat, p_value = stats.ttest_1samp(returns, 0)   # same idea: is 0 inside the CI?
    return print_result("Confidence interval / test for mean daily return",
                        "mean daily return = 0", "mean daily return != 0", t_stat, p_value,
                        "0 is outside the CI, so the average daily return is not zero.",
                        "0 is inside the CI, so the average daily return could be zero.")


def t_test_residuals(residuals):
    """One-sample t-test: is the mean of the test residuals 0 (unbiased model)?"""
    t_stat, p_value = stats.ttest_1samp(residuals, 0)
    return print_result("One-sample t-test on test residuals",
                        "mean residual = 0 (model is unbiased)",
                        "mean residual != 0 (model is biased)", t_stat, p_value,
                        f"The model is biased: on average it is off by {residuals.mean():.2f} on unseen days.",
                        "No evidence of bias: errors are centred on zero.")


def f_test(linear_residuals, poly_residuals, poly_degree):
    """F-test: is the linear model's residual variance larger than the polynomial's?

    Residual variance = SSE / (n - number of coefficients). F = var1 / var2.
    """
    n = len(linear_residuals)
    var_linear = np.sum(linear_residuals ** 2) / (n - 2)
    var_poly = np.sum(poly_residuals ** 2) / (n - (poly_degree + 1))
    f_stat = var_linear / var_poly
    p_value = stats.f.sf(f_stat, n - 2, n - (poly_degree + 1))   # one-tailed (right tail)
    return print_result(f"F-test: linear vs degree {poly_degree} residual variance",
                        "both models have equal residual variance",
                        "linear model has larger residual variance", f_stat, p_value,
                        f"Degree {poly_degree} fits significantly better than a straight line.",
                        "The polynomial is not significantly better than a straight line.")


def chi_square_normality(residuals):
    """Chi-square goodness of fit: do the residuals follow a Normal distribution?"""
    mu, sigma = residuals.mean(), residuals.std(ddof=1)
    observed, edges = np.histogram(residuals, bins=7)
    edges[0], edges[-1] = -np.inf, np.inf            # outer bins cover both tails
    expected = len(residuals) * np.diff(stats.norm.cdf(edges, mu, sigma))

    obs, exp = [], []                                  # merge bins with expected count < 5
    for o, e in zip(observed, expected):
        if exp and exp[-1] < 5:
            obs[-1] += o
            exp[-1] += e
        else:
            obs.append(o)
            exp.append(e)
    if len(exp) > 1 and exp[-1] < 5:                   # last bin still too small
        obs[-2] += obs.pop()
        exp[-2] += exp.pop()

    obs, exp = np.array(obs), np.array(exp)
    chi2_stat = np.sum((obs - exp) ** 2 / exp)          # χ² = Σ (O - E)² / E
    dof = len(obs) - 1 - 2                              # minus 2 because we estimated mean and sd
    p_value = stats.chi2.sf(chi2_stat, dof)
    print(f"\nBins after merging: {len(obs)}, observed = {obs.tolist()}, "
          f"expected = {np.round(exp, 1).tolist()}, dof = {dof}")
    return print_result("Chi-square goodness of fit (residuals vs Normal)",
                        "residuals follow a Normal distribution",
                        "residuals do not follow a Normal distribution", chi2_stat, p_value,
                        "Residuals are not Normal, so the ±1.96σ band is only approximate.",
                        "Residuals look Normal, so the ±1.96σ band is justified.")


def chi_square_independence(y_test, test_pred):
    """Chi-square test of independence of attributes: actual vs predicted High/Low.

    High = price above the median of its own series, Low = below it.
    WHY not day-to-day up/down: a smooth polynomial curve moves in only one
    direction over the test period, so one column of the table would be all zeros.
    """
    actual_high = y_test > np.median(y_test)
    predicted_high = test_pred > np.median(test_pred)
    table = np.array([[np.sum(actual_high & predicted_high), np.sum(actual_high & ~predicted_high)],
                      [np.sum(~actual_high & predicted_high), np.sum(~actual_high & ~predicted_high)]])
    print("\nContingency table (rows: actual High/Low, columns: predicted High/Low):")
    print(table)
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(table)  # Yates correction for 2x2
    return print_result("Chi-square test of independence (actual vs predicted High/Low)",
                        "actual and predicted High/Low are independent",
                        "actual and predicted High/Low are related", chi2_stat, p_value,
                        "The model correctly separates high-price days from low-price days.",
                        "The model's High/Low calls are no better than chance.")


def run_tests(df, results):
    """Run all five tests and return their decisions for the summary."""
    # CO5: validate the data-driven model with statistical tests
    return {
        "CI mean return": confidence_interval_mean(df["Return"].dropna().to_numpy()),
        "t-test residuals": t_test_residuals(results["test_residuals"]),
        "F-test": f_test(results["linear_residuals"], results["poly_residuals"], results["poly_degree"]),
        "Chi-square normality": chi_square_normality(results["test_residuals"]),
        "Chi-square independence": chi_square_independence(results["y_test"], results["test_pred"]),
    }
