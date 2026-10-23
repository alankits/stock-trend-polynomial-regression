"""
main.py - Runs the whole project step by step.

WHAT: calls every step in order and prints the results with headings.
WHY: one command (python main.py) reproduces all results, plots and
outputs/results.txt for the report.
"""
import os
import sys
import matplotlib
matplotlib.use("Agg")   # save plots to files only; no window is needed on any machine
import config
import data_loader
import descriptive
import model
import hypothesis_tests


class Tee:
    """Send everything that is printed to the terminal AND to results.txt."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def heading(title):
    """Print a clear section heading."""
    print(f"\n===== {title} =====")


def main():
    """Run steps 1-7 in order, then print a summary and limitations."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    results_file = open(os.path.join(config.OUTPUT_DIR, "results.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, results_file)

    heading("STEP 1: DATA COLLECTION AND PRE-PROCESSING")
    df = data_loader.preprocess(data_loader.load_data())
    data_loader.plot_closing_price(df)

    heading("STEP 2: DESCRIPTIVE STATISTICS (Unit IV)")
    descriptive.central_tendency(df)

    heading("STEP 3: TREND ANALYSIS (Unit IV)")
    descriptive.explain_components()
    descriptive.freehand_trend(df)
    descriptive.moving_average_trend(df)

    heading("STEP 4: DAILY RETURNS AND NORMAL DISTRIBUTION (Unit III, CO3)")
    descriptive.returns_distribution(df)   # CO3

    heading("STEP 5-6: POLYNOMIAL REGRESSION AND FORECAST (Unit IV, CO3, CO4)")
    results = model.run_model(df)          # CO3, CO4

    heading("STEP 7: HYPOTHESIS TESTS (Unit V, CO5)")
    decisions = hypothesis_tests.run_tests(df, results)   # CO5

    heading("SUMMARY")
    print(f"{'Stock':28s}: {config.TICKER}")
    print(f"{'Chosen polynomial degree':28s}: {results['degree']}")
    print(f"{'Test MSE / Test R2':28s}: {results['test_mse']:.2f} / {results['test_r2']:.3f}")
    print(f"{'Pearson r / Spearman rho':28s}: {results['pearson_r']:.3f} / {results['spearman_rho']:.3f}")
    print(f"{'Forecast trend':28s}: {results['trend']} (day {config.FORECAST_DAYS}: "
          f"{results['last_forecast']:.2f} ± {1.96 * results['sigma']:.2f})")
    for test_name, decision in decisions.items():
        print(f"{test_name:28s}: {decision}")

    heading("LIMITATIONS")
    print("1. Stock prices are noisy and affected by news, so no model predicts them exactly.")
    print("2. Polynomials extrapolate badly: far-future forecasts can shoot up or down.")
    print("3. The model uses only past prices (time), not volume, news or company results.")

    sys.stdout = sys.__stdout__
    results_file.close()
    print(f"\nAll plots and results.txt saved in: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
