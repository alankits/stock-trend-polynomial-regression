"""
data_loader.py - STEP 1: Data collection and pre-processing.

WHAT: downloads daily closing prices, cleans them, adds daily returns and a
scaled day index, and plots the closing price.
WHY: every later step (statistics, regression, tests) needs clean numeric data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import config


def load_data():
    """Download data with yfinance and save it to a dedicated data/<TICKER>.csv file.

    WHY per-ticker files: prevents one stock from overwriting another or falling back
    to another stock's data during network hiccups or rate limits.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)
    ticker_file = config.get_data_file(config.TICKER)
    df = None

    try:
        try:
            downloaded = yf.download(config.TICKER, start=config.START_DATE, end=config.END_DATE,
                                     progress=False, auto_adjust=True)
            if downloaded is not None and not downloaded.empty:
                df = downloaded
                df.columns = df.columns.get_level_values(0)   # ('Close', 'AAPL') -> 'Close'
                df = df.reset_index()                          # move Date from index to column
        except Exception:
            df = None

        if df is None or df.empty:
            # Fallback to yf.Ticker(...).history() which handles rate limits and sessions more reliably
            ticker_obj = yf.Ticker(config.TICKER)
            hist = ticker_obj.history(start=config.START_DATE, end=config.END_DATE, auto_adjust=True)
            if hist is not None and not hist.empty:
                df = hist.reset_index()

        if df is not None and not df.empty and "Close" in df.columns:
            # Save to the specific stock's CSV file
            df.to_csv(ticker_file, index=False)
            df.to_csv(config.DATA_FILE, index=False)
            print(f"Downloaded {len(df)} live rows for {config.TICKER}, saved to data/{os.path.basename(ticker_file)}")
            return df
        else:
            raise ValueError(f"No price data returned by Yahoo Finance for {config.TICKER}")

    except Exception as error:
        print(f"Live download for {config.TICKER} note: {error}")
        if os.path.exists(ticker_file):
            print(f"Loading cached real data for {config.TICKER} from {os.path.basename(ticker_file)}")
            return pd.read_csv(ticker_file)
        elif config.TICKER == "TCS.NS" and os.path.exists(config.DATA_FILE):
            print("Loading default dataset data/stock.csv for TCS.NS.")
            return pd.read_csv(config.DATA_FILE)
        else:
            print(f"ERROR: Could not download data for '{config.TICKER}' and no cached file exists at {ticker_file}.")
            raise SystemExit(1)


def preprocess(df):
    """Keep Date and Close, remove missing values, add Return and scaled x.

    WHY scale x to 0-1: with day numbers up to ~500, x^6 would be about 10^16,
    which causes huge numbers and rounding errors. On 0-1, x^6 stays small.
    """
    df = df[["Date", "Close"]].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    # Remove commas like "3,512.40" and turn text such as "null" into NaN
    df["Close"] = pd.to_numeric(df["Close"].astype(str).str.replace(",", ""), errors="coerce")
    print(f"Missing closing prices found: {df['Close'].isna().sum()} (rows dropped)")
    df = df.dropna().sort_values("Date").reset_index(drop=True)

    df["Return"] = df["Close"].pct_change() * 100   # daily % change; first row is NaN
    df["x"] = np.arange(len(df))                     # day index 0, 1, 2, ...
    df["x_scaled"] = df["x"] / df["x"].max()         # scaled to the range 0-1
    print(f"Clean data: {len(df)} trading days from {df['Date'].iloc[0].date()} "
          f"to {df['Date'].iloc[-1].date()}")
    print(df.head().to_string(index=False))
    return df


def plot_closing_price(df):
    """Save a line chart of the closing price (first look at the data)."""
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Close"], color="steelblue")
    config.save_plot("01_closing_price.png", f"{config.TICKER} – Daily Closing Price History", "Date", "Closing Price", legend=False)
