# Stock Price Trend Prediction Using Polynomial Regression

Micro project for **25CS203 – Probability and Statistics for Computer Science**
Industry focus: **Data Analyst** · Stock: **TCS.NS** (Tata Consultancy Services)

The project downloads about 2 years of daily closing prices, describes them statistically,
fits polynomial regression models of degree 1 to 6, picks the best degree, forecasts the next
10 trading days with a 95% prediction band, and validates the model with hypothesis tests.

## How to install and run

```bash
pip install -r requirements.txt
```

### 1. Modern Web Application (Recommended for Demo & Analysis)

```bash
python web_app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser.
Features:
- **Preset Dataset Picker**: One-click selection for Indian and Global stocks (`TCS.NS`, `RELIANCE.NS`, `INFY.NS`, `AAPL`, `MSFT`, `AMZN`, `GOOGL`, `TSLA`).
- **Custom Ticker Input**: Analyze any global ticker supported by Yahoo Finance.
- **Dynamic Live Run**: Executes the complete 7-step statistical pipeline on demand.
- **Interactive Metric Cards**: Highlighting Best Polynomial Degree, Test $R^2$, MSE, Pearson $r$, Spearman $\rho$, and 10-day Forecast.
- **Full Plot Gallery**: Tabbed viewer for all 7 high-resolution charts.
- **Statistical Hypothesis Decisions**: Clear cards displaying results for Confidence Interval, $t$-test, $F$-test, and Chi-square tests.
- **Step-by-Step Execution Logs**: Expandable breakdown for each phase of analysis.

### 2. Command-Line Version

```bash
python main.py
```

All results print in the terminal and are saved in `outputs/results.txt`. All plots are saved in `outputs/`.

### 3. Classic Tkinter GUI

```bash
python gui.py
```

Type a ticker (default `TCS.NS`), click **Run Analysis**, read the results in the window, and
click any of the 7 buttons on the left to open a plot. The GUI uses `tkinter`, which comes with
Python on Windows and macOS (on Ubuntu/Debian install it with `sudo apt install python3-tk`).

**No internet?** Open Yahoo Finance, search `TCS.NS` → *Historical Data* → *Download*,
and save the file as `data/stock.csv`. The program loads that file automatically
when the download fails.

## Project structure

```
├── main.py              # runs all steps in order and writes outputs/results.txt
├── gui.py               # tkinter window: Run Analysis button, results and plot viewer
├── config.py            # settings (ticker, dates, test size, alpha, degrees) + save_plot helper
├── data_loader.py       # Step 1: download, clean, add returns and scaled day index
├── descriptive.py       # Steps 2–4: mean/median/mode, trend methods, Normal distribution
├── model.py             # Steps 5–6: polynomial regression, degree choice, forecast
├── hypothesis_tests.py  # Step 7: CI, t-test, F-test, chi-square tests
├── data/                # stock.csv is saved here
├── outputs/             # 7 PNG plots + results.txt
├── docs/                # Project_Report.docx and Presentation.pptx (fill the yellow placeholders)
├── requirements.txt     # libraries to install
└── README.md            # this file
```

## Course Outcome mapping

| CO | What it means here | File | Function |
|----|--------------------|------|----------|
| CO3 – Predict data with uncertainty | Normal probabilities of returns; 10-day forecast with 95% prediction band | `descriptive.py`, `model.py` | `returns_distribution()`, `forecast()` |
| CO4 – Justify model decisions statistically | Degree chosen by lowest test MSE; MSE, R², Pearson and Spearman correlation | `model.py` | `select_degree()`, `correlations()` |
| CO5 – Validate the model with statistical tests | CI, t-test, F-test, chi-square goodness of fit, chi-square independence | `hypothesis_tests.py` | `run_tests()` and the five test functions |

## Syllabus topic mapping

| Unit | Topic | File → Function |
|------|-------|-----------------|
| III | Normal distribution (returns) | `descriptive.py` → `returns_distribution()` |
| III | Normal distribution (residuals) | `hypothesis_tests.py` → `chi_square_normality()`, `model.py` → `forecast()` (±1.96σ) |
| IV | Mean, median, mode | `descriptive.py` → `central_tendency()` |
| IV | Components of a time series | `descriptive.py` → `explain_components()` |
| IV | Trend – freehand / graphic method | `descriptive.py` → `freehand_trend()` |
| IV | Trend – moving average method | `descriptive.py` → `moving_average_trend()` |
| IV | Karl Pearson correlation | `model.py` → `correlations()` (formula + scipy) |
| IV | Spearman rank correlation | `model.py` → `correlations()` |
| IV | Regression | `model.py` → `fit_polynomial()`, `select_degree()` |
| V | Confidence interval for a mean | `hypothesis_tests.py` → `confidence_interval_mean()` |
| V | Student t-test | `hypothesis_tests.py` → `t_test_residuals()` |
| V | F-test | `hypothesis_tests.py` → `f_test()` |
| V | Chi-square goodness of fit | `hypothesis_tests.py` → `chi_square_normality()` |
| V | Chi-square test of independence | `hypothesis_tests.py` → `chi_square_independence()` |

## Output plots

| File | What it shows |
|------|---------------|
| `01_closing_price.png` | Raw daily closing price over 2 years |
| `02_freehand_trend.png` | Freehand-style trend line through quarterly average prices |
| `03_moving_average.png` | 5-day and 20-day moving averages smoothing the price |
| `04_returns_normal.png` | Histogram of daily returns with the fitted Normal curve |
| `05_degree_selection.png` | Test MSE for degrees 1–6 (log scale), used to choose the degree |
| `06_model_fit.png` | Actual prices vs chosen-degree and comparison-degree fits and test predictions |
| `07_forecast.png` | Last 60 days, the 10-day forecast and its 95% prediction band |

## Notes on design choices

- **Scaled x (0–1):** with day numbers up to ~500, x⁶ would be around 10¹⁶; scaling keeps numbers small.
- **No shuffling:** the first 80% of days train the model and the last 20% test it, like real forecasting.
- **F-test comparison:** if degree 1 is chosen, it is compared with degree 2 instead (comparing a model with itself means nothing).
- **Chi-square independence uses High/Low:** a smooth polynomial moves in one direction over the
  test period, so day-to-day "predicted up/down" would be all one value. Instead each test day is
  labelled High or Low (above/below the median) for actual and predicted prices.
- **Residual tests use test-set residuals:** on training data the residual mean is always ≈ 0 by
  construction, so testing it would prove nothing.

## Viva preparation

1. **Why polynomial regression and not linear?**
   Prices curve up and down. A straight line can only show one constant slope; a polynomial
   (x, x², x³, …) can bend to follow the shape. We still compare against degree 1 to prove it helps.

2. **Is polynomial regression a linear model?**
   Yes, it is linear in the coefficients. We create new columns x, x², …, xᵈ and then run
   ordinary linear regression on them (that is what `PolynomialFeatures` + `LinearRegression` does).

3. **What do MSE and R² mean?**
   MSE = average of (actual − predicted)²; lower is better. R² = 1 − SSE/SST, the fraction of
   variation explained by the model; 1 is perfect, 0 is no better than the mean, negative is worse
   than just predicting the mean.

4. **Why is there no shuffling in the train/test split?**
   It is time-series data. We must predict the future from the past; shuffling would leak future
   prices into training and give falsely good results.

5. **What is overfitting and underfitting?**
   Underfitting: the model is too simple and misses the pattern (high train and test error).
   Overfitting: the model is too complex and learns noise (low train error, high test error).

6. **How did you choose the degree?**
   We fitted degrees 1–6 and chose the one with the lowest test MSE, i.e. the best error on data
   the model never saw.

7. **Why scale x to 0–1?**
   To avoid enormous values like 500⁶ ≈ 1.6×10¹⁶, which cause rounding errors in the computation.

8. **What is a p-value?**
   The probability of getting a result at least this extreme if H0 were true. If p < α (0.05),
   the result is too unlikely under H0, so we reject H0.

9. **Difference between the t-test and the F-test?**
   The t-test compares **means** (here: is the mean residual 0?). The F-test compares
   **variances** (here: is the linear model's error variance larger than the polynomial's?).

10. **Why use the Normal distribution?**
    Daily returns and model errors are sums of many small random effects, so they are roughly
    Normal (Central Limit Theorem). This lets us use the Normal CDF for probabilities and ±1.96σ for a 95% band.

11. **Why ±1.96σ for the 95% band?**
    For a Normal distribution, 95% of values lie within 1.96 standard deviations of the mean.

12. **Pearson vs Spearman correlation?**
    Pearson measures the strength of a **linear** relationship using the actual values.
    Spearman uses **ranks**, so it measures any monotonic (always increasing/decreasing) relationship and is less affected by outliers.

13. **What does the chi-square goodness-of-fit test check here, and why merge bins?**
    Whether the residuals follow a Normal distribution, by comparing observed bin counts with expected
    Normal counts: χ² = Σ(O − E)²/E. Bins with expected count < 5 are merged because the χ² approximation
    is unreliable for small expected counts. Degrees of freedom = bins − 1 − 2 (mean and sd were estimated).

14. **What are the components of a time series?**
    Trend (long-term direction), Seasonal (repeats within a year), Cyclical (waves longer than a year),
    Irregular (random shocks).

15. **Why forecast only 10 days?**
    Outside the data range a polynomial grows very fast (x⁶ explodes), so long forecasts become
    meaningless. Stock prices also depend on news that the model cannot see.

## Limitations

- Stock prices are noisy and driven by news, so no model predicts them exactly.
- Polynomials extrapolate badly; far-future forecasts can shoot up or down.
- The model uses only time (past prices), not volume, news or company results.
