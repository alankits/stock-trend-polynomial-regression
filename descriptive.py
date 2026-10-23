"""
descriptive.py - STEPS 2-4: Descriptive statistics, trend and returns distribution.

WHAT: central tendency (Unit IV), time series components and trend measurement
(Unit IV), and the Normal distribution of daily returns (Unit III).
WHY: we must understand the data before we build a model on it.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import config


def central_tendency(df):
    """Print mean, median, mode, then mean and std by formula and by numpy.

    WHY both ways: shows that we understand the formula, not only the library.
    """
    for name, series, decimals in [("Closing price", df["Close"], 0),
                                   ("Daily return %", df["Return"].dropna(), 1)]:
        # Prices are continuous and almost never repeat exactly, so we round
        # them first; the mode is then the most common rounded value.
        mode = series.round(decimals).mode()[0]
        print(f"{name:15s}: mean = {series.mean():.3f}, median = {series.median():.3f}, "
              f"mode (rounded) = {mode}")

    returns = df["Return"].dropna().to_numpy()
    n = len(returns)
    manual_mean = sum(returns) / n                                         # x̄ = Σx / n
    manual_std = (sum((r - manual_mean) ** 2 for r in returns) / (n - 1)) ** 0.5  # s = √(Σ(x-x̄)²/(n-1))
    print(f"\nDaily return mean: formula = {manual_mean:.5f}, numpy = {np.mean(returns):.5f}")
    print(f"Daily return std : formula = {manual_std:.5f}, numpy = {np.std(returns, ddof=1):.5f}")


def explain_components():
    """Print the four components of a time series (theory, Unit IV)."""
    print("Trend (T)     : long-term upward or downward movement of the price.")
    print("Seasonal (S)  : pattern that repeats within a year (e.g. results season).")
    print("Cyclical (C)  : long up-and-down waves lasting more than a year (business cycles).")
    print("Irregular (I) : random, unpredictable changes (news, shocks).")


def freehand_trend(df):
    """Freehand / graphic method: draw a smooth line through the middle of the data.

    NOTE: the real freehand method is drawn by hand with a pencil, so it is
    subjective (two people draw different lines). Here we imitate it by joining
    the average price of each quarter, which passes through the middle of the data.
    """
    quarters = df.groupby(df["Date"].dt.to_period("Q"))
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], color="lightgray", label="Closing price")
    plt.plot(quarters["Date"].mean(), quarters["Close"].mean(), color="red",
             marker="o", label="Freehand trend line")
    config.save_plot("02_freehand_trend.png", f"{config.TICKER} – Trend by Freehand (Graphic) Method", "Date", "Price")


def moving_average_trend(df):
    """Moving average method: average of the last k days removes short-term noise."""
    df["MA5"] = df["Close"].rolling(window=5).mean()     # short-term trend
    df["MA20"] = df["Close"].rolling(window=20).mean()   # smoother, longer-term trend
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], color="lightgray", label="Closing price")
    plt.plot(df["Date"], df["MA5"], label="5-day moving average")
    plt.plot(df["Date"], df["MA20"], label="20-day moving average")
    config.save_plot("03_moving_average.png", f"{config.TICKER} – Trend by Moving Average Method (5 & 20 MA)", "Date", "Price")
    print(f"Latest 5-day MA = {df['MA5'].iloc[-1]:.2f}, latest 20-day MA = {df['MA20'].iloc[-1]:.2f}")


def returns_distribution(df):
    """Treat daily return as a continuous random variable and fit a Normal curve.

    WHY: if returns are roughly Normal, the Normal CDF gives probabilities
    of gains and losses, which is how we describe uncertainty (CO3).
    """
    returns = df["Return"].dropna()
    mu, sigma = returns.mean(), returns.std()
    x_values = np.linspace(returns.min(), returns.max(), 200)

    plt.figure(figsize=(8, 5))
    plt.hist(returns, bins=40, density=True, alpha=0.6, label="Daily returns")
    plt.plot(x_values, stats.norm.pdf(x_values, mu, sigma), color="red",
             label=f"Normal PDF (mean={mu:.2f}, sd={sigma:.2f})")
    config.save_plot("04_returns_normal.png", f"{config.TICKER} – Daily Returns vs Fitted Normal Distribution", "Daily return (%)", "Density")

    # CO3: probabilities from the Normal CDF, P(X <= a) = norm.cdf(a)
    print(f"Returns ~ Normal(mean = {mu:.3f}%, sd = {sigma:.3f}%)")
    print(f"P(return > 0)   = 1 - CDF(0)  = {1 - stats.norm.cdf(0, mu, sigma):.3f}")
    print(f"P(return < -2%) = CDF(-2)     = {stats.norm.cdf(-2, mu, sigma):.3f}")
    return mu, sigma
