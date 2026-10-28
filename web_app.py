"""
web_app.py - Flask web server that wraps the stock analysis pipeline.

Exposes a REST API and serves a modern web interface for running
the stock price trend prediction analysis on any ticker.
"""
import os
import io
import sys
import json
import base64
import threading
import traceback

from flask import Flask, jsonify, request, send_from_directory

import matplotlib
matplotlib.use("Agg")

import config
import data_loader
import descriptive
import model
import hypothesis_tests

app = Flask(__name__, static_folder="static")

# Lock to prevent concurrent analysis runs (shared matplotlib state)
_analysis_lock = threading.Lock()

# ── Preset stocks ────────────────────────────────────────────────────
PRESET_STOCKS = [
    {"name": "Amazon",                "ticker": "AMZN",          "exchange": "NASDAQ", "currency": "USD"},
    {"name": "Apple",                 "ticker": "AAPL",          "exchange": "NASDAQ", "currency": "USD"},
    {"name": "Tesla",                 "ticker": "TSLA",          "exchange": "NASDAQ", "currency": "USD"},
    {"name": "Google / Alphabet",     "ticker": "GOOGL",         "exchange": "NASDAQ", "currency": "USD"},
    {"name": "Microsoft",             "ticker": "MSFT",          "exchange": "NASDAQ", "currency": "USD"},
    {"name": "Meta / Facebook",       "ticker": "META",          "exchange": "NASDAQ", "currency": "USD"},
    {"name": "Netflix",               "ticker": "NFLX",          "exchange": "NASDAQ", "currency": "USD"},
    {"name": "NVIDIA",                "ticker": "NVDA",          "exchange": "NASDAQ", "currency": "USD"},
    {"name": "Reliance Industries",   "ticker": "RELIANCE.NS",   "exchange": "NSE",    "currency": "INR"},
    {"name": "Tata Consultancy",      "ticker": "TCS.NS",        "exchange": "NSE",    "currency": "INR"},
    {"name": "Infosys",               "ticker": "INFY.NS",       "exchange": "NSE",    "currency": "INR"},
]


def _capture_stdout(func, *args, **kwargs):
    """Run func while capturing everything printed to stdout."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = old_stdout
    return result, buf.getvalue()


def _encode_plot(filename):
    """Read a PNG from outputs/ and return a base64 data-URI."""
    path = os.path.join(config.OUTPUT_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _run_analysis(ticker):
    """Run the full 7-step pipeline for one ticker and return structured results."""
    config.TICKER = ticker
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    logs = {}

    # Step 1
    df, log = _capture_stdout(lambda: data_loader.preprocess(data_loader.load_data()))
    logs["step1"] = log
    data_loader.plot_closing_price(df)

    # Step 2
    _, log = _capture_stdout(descriptive.central_tendency, df)
    logs["step2"] = log

    # Step 3
    _, log1 = _capture_stdout(descriptive.explain_components)
    descriptive.freehand_trend(df)
    _, log2 = _capture_stdout(descriptive.moving_average_trend, df)
    logs["step3"] = log1 + log2

    # Step 4
    _, log = _capture_stdout(descriptive.returns_distribution, df)
    logs["step4"] = log

    # Step 5–6
    results, log = _capture_stdout(model.run_model, df)
    logs["step5_6"] = log

    # Step 7
    decisions, log = _capture_stdout(hypothesis_tests.run_tests, df, results)
    logs["step7"] = log

    # Encode plots as base64
    plot_files = [
        ("Closing Price",      "01_closing_price.png"),
        ("Freehand Trend",     "02_freehand_trend.png"),
        ("Moving Averages",    "03_moving_average.png"),
        ("Returns vs Normal",  "04_returns_normal.png"),
        ("Degree Selection",   "05_degree_selection.png"),
        ("Model Fit",          "06_model_fit.png"),
        ("Forecast",           "07_forecast.png"),
    ]
    plots = []
    for label, fname in plot_files:
        data_uri = _encode_plot(fname)
        if data_uri:
            plots.append({"label": label, "data": data_uri})

    # Build summary
    summary = {
        "ticker":       ticker,
        "degree":       results["degree"],
        "test_mse":     round(results["test_mse"], 2),
        "test_r2":      round(results["test_r2"], 3),
        "pearson_r":    round(results["pearson_r"], 4),
        "spearman_rho": round(results["spearman_rho"], 4),
        "trend":        results["trend"],
        "last_forecast":round(results["last_forecast"], 2),
        "sigma":        round(results["sigma"], 2),
        "band_95":      round(1.96 * results["sigma"], 2),
    }

    return {
        "summary":   summary,
        "decisions": decisions,
        "logs":      logs,
        "plots":     plots,
    }


# ── API routes ───────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/concepts")
@app.route("/concepts-and-architecture")
@app.route("/syllabus")
@app.route("/docs")
def syllabus():
    return send_from_directory("static", "syllabus.html")


@app.route("/api/stocks")
def api_stocks():
    """Return the list of preset stock options."""
    return jsonify(PRESET_STOCKS)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Run the analysis for a given ticker."""
    data = request.get_json(force=True)
    ticker = data.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    acquired = _analysis_lock.acquire(blocking=False)
    if not acquired:
        return jsonify({"error": "Another analysis is already running. Please wait."}), 429

    try:
        result = _run_analysis(ticker)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        _analysis_lock.release()


if __name__ == "__main__":
    os.makedirs("static", exist_ok=True)
    print("Starting Stock Analysis Web App...")
    print("Open http://localhost:5000 in your browser")
    app.run(host="0.0.0.0", port=5000, debug=False)
