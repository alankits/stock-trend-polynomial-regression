"""
model.py - STEPS 5-6: Polynomial regression, degree selection and forecast.

WHAT: fits polynomial regression of degrees 1..MAX_DEGREE, picks the best
degree, checks correlation, and forecasts the next few days with a band.
WHY: stock prices curve up and down, so a straight line (degree 1) may miss
the shape; a polynomial can bend to follow it (Unit IV regression, CO3, CO4).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import config


def split_data(df):
    """Time-based split: first 80% for training, last 20% for testing.

    WHY no shuffling: we must predict the FUTURE from the PAST. Shuffling would
    let the model see future days during training, which is cheating.
    """
    split_index = int(len(df) * (1 - config.TEST_SIZE))
    return df.iloc[:split_index], df.iloc[split_index:]


def fit_polynomial(x, y, degree):
    """Fit y = b0 + b1*x + b2*x^2 + ... + bd*x^d by least squares.

    PolynomialFeatures makes the columns x, x^2, ..., x^d; LinearRegression
    then finds the best coefficients for them.
    """
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(x.reshape(-1, 1), y)   # sklearn wants x as a column (2-D)
    return model


def select_degree(train, test):
    """Try every degree, print the error table and choose the best degree (CO4)."""
    x_train, y_train = train["x_scaled"].to_numpy(), train["Close"].to_numpy()
    x_test, y_test = test["x_scaled"].to_numpy(), test["Close"].to_numpy()
    rows = []
    print(f"{'Degree':>6} {'Train MSE':>12} {'Test MSE':>14} {'Train R2':>9} {'Test R2':>10}")
    for degree in range(1, config.MAX_DEGREE + 1):
        model = fit_polynomial(x_train, y_train, degree)
        train_pred = model.predict(x_train.reshape(-1, 1))
        test_pred = model.predict(x_test.reshape(-1, 1))
        row = (degree, mean_squared_error(y_train, train_pred), mean_squared_error(y_test, test_pred),
               r2_score(y_train, train_pred), r2_score(y_test, test_pred))
        rows.append(row)
        print(f"{row[0]:>6} {row[1]:>12.2f} {row[2]:>14.2f} {row[3]:>9.3f} {row[4]:>10.3f}")

    degrees = [r[0] for r in rows]
    test_mses = [r[2] for r in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, test_mses, marker="o")
    plt.yscale("log")   # high degrees can have huge errors, log scale keeps all visible
    config.save_plot("05_degree_selection.png", f"{config.TICKER} – Test MSE vs Polynomial Degree", "Degree", "Test MSE (log scale)", legend=False)

    # CO4: choose the degree with the lowest error on UNSEEN (test) data
    best = rows[int(np.argmin(test_mses))]
    print(f"\nChosen degree = {best[0]} (lowest test MSE = {best[2]:.2f}, test R2 = {best[4]:.3f})")
    print("Justification: a lower degree UNDERFITS (too simple to follow the curve), while")
    print("a higher degree OVERFITS: train MSE keeps falling, but it learns noise and test")
    print("MSE rises. The best degree balances both, judged on data the model never saw.")
    return best


def plot_fit(train, test, degree, compare_degree):
    """Plot actual prices, fitted train values and test predictions for two degrees."""
    plt.figure(figsize=(10, 5))
    plt.plot(train["Date"], train["Close"], color="gray", label="Actual (train)")
    plt.plot(test["Date"], test["Close"], color="black", label="Actual (test)")
    for d, color in [(degree, "red"), (compare_degree, "blue")]:
        model = fit_polynomial(train["x_scaled"].to_numpy(), train["Close"].to_numpy(), d)
        plt.plot(train["Date"], model.predict(train[["x_scaled"]].to_numpy()), color=color,
                 label=f"Degree {d} fitted")
        plt.plot(test["Date"], model.predict(test[["x_scaled"]].to_numpy()), color=color,
                 linestyle="--", label=f"Degree {d} test prediction")
    config.save_plot("06_model_fit.png", f"{config.TICKER} – Polynomial Regression (Degree {degree} vs {compare_degree})", "Date", "Price")


def correlations(actual, predicted):
    """Pearson (formula + scipy) and Spearman rank correlation (CO4).

    WHY: high correlation means predictions move together with actual prices.
    """
    # CO4: correlation shows how closely predictions follow actual prices
    a_mean, p_mean = actual.mean(), predicted.mean()
    # r = Σ(a - ā)(p - p̄) / √( Σ(a - ā)² × Σ(p - p̄)² )
    manual_r = np.sum((actual - a_mean) * (predicted - p_mean)) / np.sqrt(
        np.sum((actual - a_mean) ** 2) * np.sum((predicted - p_mean) ** 2))
    scipy_r = stats.pearsonr(actual, predicted)[0]
    spearman_rho = stats.spearmanr(actual, predicted)[0]   # correlation of the RANKS
    print(f"Pearson r  : formula = {manual_r:.4f}, scipy = {scipy_r:.4f}")
    print(f"Spearman rho (rank correlation) = {spearman_rho:.4f}")
    return scipy_r, spearman_rho


def forecast(df, degree):
    """Refit on all data and forecast the next FORECAST_DAYS trading days (CO3).

    WHY only a few days: a polynomial shoots up or down very fast outside the
    data range, so far-away forecasts become unreliable.
    """
    x_all, y_all = df["x_scaled"].to_numpy(), df["Close"].to_numpy()
    model = fit_polynomial(x_all, y_all, degree)
    sigma = np.std(y_all - model.predict(x_all.reshape(-1, 1)), ddof=degree + 1)  # residual std

    last_x = df["x"].iloc[-1]
    future_x = (last_x + np.arange(1, config.FORECAST_DAYS + 1)) / last_x   # same 0-1 scaling
    future_price = model.predict(future_x.reshape(-1, 1))
    future_dates = pd.bdate_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=config.FORECAST_DAYS)
    # CO3: if residuals are Normal, 95% of prices fall within ±1.96σ of the prediction
    lower, upper = future_price - 1.96 * sigma, future_price + 1.96 * sigma

    print(f"Residual std (sigma) = {sigma:.2f}, 95% band = prediction ± {1.96 * sigma:.2f}")
    print(f"{'Date':>12} {'Forecast':>10} {'Lower 95%':>10} {'Upper 95%':>10}")
    for date, price, low, high in zip(future_dates, future_price, lower, upper):
        print(f"{str(date.date()):>12} {price:>10.2f} {low:>10.2f} {high:>10.2f}")

    recent = df.tail(60)   # show only the last 60 days so the forecast is visible
    plt.figure(figsize=(10, 5))
    plt.plot(recent["Date"], recent["Close"], color="black", label="Actual (last 60 days)")
    plt.plot(recent["Date"], model.predict(recent[["x_scaled"]].to_numpy()), color="red", label="Fitted")
    plt.plot(future_dates, future_price, color="red", linestyle="--", marker="o", label="Forecast")
    plt.fill_between(future_dates, lower, upper, color="red", alpha=0.2, label="95% prediction band")
    config.save_plot("07_forecast.png", f"{config.TICKER} – {config.FORECAST_DAYS}-Day Forecast (Degree {degree})", "Date", "Price")

    trend = "UPWARD" if future_price[-1] > future_price[0] else "DOWNWARD"   # sign of the slope
    print(f"\nForecast trend: {trend} ({future_price[0]:.2f} -> {future_price[-1]:.2f})")
    return trend, future_price[-1], sigma


def run_model(df):
    """Run steps 5-6 and return what the hypothesis tests need."""
    train, test = split_data(df)
    print(f"Train: {len(train)} days, Test: {len(test)} days (no shuffling)\n")
    best = select_degree(train, test)
    degree = best[0]
    # If degree 1 wins, compare it with degree 2 so the comparison still means something
    compare_degree = 2 if degree == 1 else 1
    plot_fit(train, test, degree, compare_degree)

    x_train, y_train = train["x_scaled"].to_numpy(), train["Close"].to_numpy()
    x_test, y_test = test["x_scaled"].to_numpy(), test["Close"].to_numpy()
    chosen = fit_polynomial(x_train, y_train, degree)
    test_pred = chosen.predict(x_test.reshape(-1, 1))
    print()
    pearson_r, spearman_rho = correlations(y_test, test_pred)
    print()
    trend, last_forecast, sigma = forecast(df, degree)

    linear = fit_polynomial(x_train, y_train, 1)
    poly_degree = max(degree, 2)   # the F-test needs a polynomial with degree > 1
    poly = fit_polynomial(x_train, y_train, poly_degree)
    return {"degree": degree, "test_mse": best[2], "test_r2": best[4],
            "pearson_r": pearson_r, "spearman_rho": spearman_rho,
            "trend": trend, "last_forecast": last_forecast, "sigma": sigma,
            "y_test": y_test, "test_pred": test_pred, "test_residuals": y_test - test_pred,
            "linear_residuals": y_train - linear.predict(x_train.reshape(-1, 1)),
            "poly_residuals": y_train - poly.predict(x_train.reshape(-1, 1)),
            "poly_degree": poly_degree}
