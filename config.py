"""
config.py - All project settings in one place, plus one small plot helper.

WHY: if you want a different stock, dates or test settings, you change them
here only, instead of hunting through every file.
"""
import os
import matplotlib.pyplot as plt

TICKER = "AMZN"              # Amazon.com, Inc. on NASDAQ (USA)
START_DATE = "2024-10-22"    # about 2 years of daily data
END_DATE = "2026-10-22"
TEST_SIZE = 0.2              # last 20% of the days are kept for testing
ALPHA = 0.05                 # significance level used by every hypothesis test
MAX_DEGREE = 6               # try polynomial degrees 1, 2, ..., 6
FORECAST_DAYS = 10           # forecast only a short time into the future

# Folders are built from this file's location, so the project runs from anywhere
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_FILE = os.path.join(DATA_DIR, "stock.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def get_data_file(ticker=None):
    """Return specific CSV file path for a given stock ticker."""
    sym = (ticker or TICKER).strip().upper().replace("/", "_")
    return os.path.join(DATA_DIR, f"{sym}.csv")


def save_plot(filename, title, xlabel, ylabel, legend=True):
    """Add title, axis labels and legend, then save the plot as a PNG in outputs/.

    WHY: every plot needs the same finishing steps, so we write them only once.
    """
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=120)
    plt.close()   # free memory before the next plot
