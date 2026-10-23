"""
gui.py - Enhanced desktop GUI for the stock analysis project.

WHAT: Choose a stock from a dropdown (or type any ticker), press "Run Analysis",
      read the results on the right, and click any of the 7 plot buttons to view
      an inline image in the same window.
WHY: makes the project easy to demonstrate with multiple datasets, without needing
     to know ticker symbols by heart or edit config.py.
"""
import os
import sys
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
import config
import main

# ---- Preset stocks the user can pick from the dropdown ----
PRESET_STOCKS = [
    ("Amazon (AMZN)",                   "AMZN"),
    ("Apple (AAPL)",                    "AAPL"),
    ("Tesla (TSLA)",                    "TSLA"),
    ("Google / Alphabet (GOOGL)",       "GOOGL"),
    ("Microsoft (MSFT)",                "MSFT"),
    ("Meta / Facebook (META)",          "META"),
    ("Netflix (NFLX)",                  "NFLX"),
    ("NVIDIA (NVDA)",                   "NVDA"),
    ("Reliance Industries (RELIANCE.NS)","RELIANCE.NS"),
    ("TCS (TCS.NS)",                    "TCS.NS"),
    ("Custom – type in the box below",  "__CUSTOM__"),
]

# (button label, PNG filename in outputs/)
PLOTS = [
    ("1. Closing Price",    "01_closing_price.png"),
    ("2. Freehand Trend",   "02_freehand_trend.png"),
    ("3. Moving Averages",  "03_moving_average.png"),
    ("4. Returns vs Normal","04_returns_normal.png"),
    ("5. Degree Selection", "05_degree_selection.png"),
    ("6. Model Fit",        "06_model_fit.png"),
    ("7. Forecast",         "07_forecast.png"),
]


# ---- Global state ----
_photo_ref = None     # keep a reference so tkinter doesn't garbage-collect the image
_history = []         # list of (ticker, summary_line) for the history panel


def _get_ticker():
    """Return the ticker to use: from dropdown, or custom entry if Custom is selected."""
    label = stock_var.get()
    for display, sym in PRESET_STOCKS:
        if display == label:
            if sym == "__CUSTOM__":
                return custom_entry.get().strip().upper()
            return sym
    return custom_entry.get().strip().upper() or config.TICKER


def _on_preset_change(event=None):
    """Enable/disable the custom entry box based on dropdown selection."""
    label = stock_var.get()
    is_custom = any(sym == "__CUSTOM__" and display == label for display, sym in PRESET_STOCKS)
    if is_custom:
        custom_entry.config(state="normal")
        custom_entry.focus()
    else:
        custom_entry.config(state="disabled")


def run_analysis():
    """Run all 7 steps for the selected stock, then show results and update history."""
    ticker = _get_ticker()
    if not ticker:
        messagebox.showwarning("No ticker", "Please select a stock or type a ticker symbol.")
        return

    config.TICKER = ticker
    status_label.config(text=f"⏳  Running analysis for  {ticker}  …  please wait",
                        fg="#e6a817")
    run_btn.config(state="disabled")
    window.update()

    try:
        main.main()
    except SystemExit:
        sys.stdout = sys.__stdout__
        status_label.config(text="❌  Failed – no data downloaded.", fg="#e74c3c")
        run_btn.config(state="normal")
        messagebox.showerror(
            "No data",
            f"Could not download data for '{ticker}'.\n\n"
            "Check the ticker symbol and your internet connection.\n"
            "You can also place a CSV file at  data/stock.csv  as a fallback."
        )
        return

    # Show results text
    results_path = os.path.join(config.OUTPUT_DIR, "results.txt")
    with open(results_path, encoding="utf-8") as f:
        content = f.read()
    results_box.config(state="normal")
    results_box.delete("1.0", tk.END)
    results_box.insert(tk.END, content)
    results_box.config(state="disabled")

    # Update history
    # Extract one-line summary from results
    summary = _extract_summary(content, ticker)
    _history.append(summary)
    _refresh_history()

    status_label.config(
        text=f"✅  Done – {ticker}   |   Click a plot button on the left to view it.",
        fg="#2ecc71"
    )
    run_btn.config(state="normal")

    # Auto-open the closing price plot
    show_plot("01_closing_price.png")


def _extract_summary(content, ticker):
    """Pull out the forecast trend line for the history panel."""
    for line in content.splitlines():
        if line.startswith("Forecast trend"):
            return f"{ticker}: {line.split(':', 1)[-1].strip()}"
    return f"{ticker}: analysis complete"


def _refresh_history():
    history_box.config(state="normal")
    history_box.delete("1.0", tk.END)
    for i, entry in enumerate(_history, 1):
        history_box.insert(tk.END, f"{i}. {entry}\n")
    history_box.config(state="disabled")


def show_plot(filename):
    """Display a saved PNG in the inline image panel (right side)."""
    global _photo_ref
    path = os.path.join(config.OUTPUT_DIR, filename)
    if not os.path.exists(path):
        messagebox.showinfo("Plot not ready",
                            f"The file  {filename}  does not exist yet.\n"
                            "Please run an analysis first.")
        return

    # Use PIL for better image handling (resize to fit panel)
    try:
        from PIL import Image, ImageTk
        img = Image.open(path)
        panel_w = centre.winfo_width() if 'centre' in globals() else 650
        panel_h = centre.winfo_height() if 'centre' in globals() else 450
        if panel_w < 100: panel_w = 650
        if panel_h < 100: panel_h = 450
        # Leave a little margin
        img.thumbnail((max(200, panel_w - 20), max(200, panel_h - 40)), Image.Resampling.LANCZOS)
        _photo_ref = ImageTk.PhotoImage(img)
    except Exception as err:
        try:
            _photo_ref = tk.PhotoImage(file=path)
        except Exception:
            _photo_ref = None

    if _photo_ref:
        image_label.config(image=_photo_ref, text="")
    else:
        image_label.config(image="", text=f"Could not render image: {filename}")
    plot_title_label.config(text=filename.replace("_", " ").replace(".png", "").title())


# ====================================================================
#  Build the window
# ====================================================================
window = tk.Tk()
window.title("Stock Price Trend Prediction – Polynomial Regression")
window.geometry("1180x720")
window.configure(bg="#1e1e2e")

DARK   = "#1e1e2e"
CARD   = "#2a2a3e"
ACCENT = "#7c5cbf"
GREEN  = "#2ecc71"
TEXT   = "#e0e0f0"
MUTED  = "#8888aa"
FONT   = ("Segoe UI", 10)
FONT_B = ("Segoe UI", 10, "bold")
MONO   = ("Courier New", 9)

style = ttk.Style()
style.theme_use("clam")
style.configure("TCombobox",
                fieldbackground=CARD, background=CARD,
                foreground=TEXT, selectbackground=ACCENT)

# ── Top bar ─────────────────────────────────────────────────────────
top = tk.Frame(window, bg=DARK, pady=10, padx=10)
top.pack(fill=tk.X)

tk.Label(top, text="📈 Stock Analysis", font=("Segoe UI", 13, "bold"),
         bg=DARK, fg=TEXT).pack(side=tk.LEFT, padx=(0, 20))

tk.Label(top, text="Choose stock:", font=FONT, bg=DARK, fg=MUTED).pack(side=tk.LEFT)
stock_var = tk.StringVar(value=PRESET_STOCKS[0][0])
stock_menu = ttk.Combobox(top, textvariable=stock_var, width=34, font=FONT,
                          values=[d for d, _ in PRESET_STOCKS], state="readonly")
stock_menu.pack(side=tk.LEFT, padx=6)
stock_menu.bind("<<ComboboxSelected>>", _on_preset_change)

tk.Label(top, text="or type ticker:", font=FONT, bg=DARK, fg=MUTED).pack(side=tk.LEFT, padx=(10, 2))
custom_entry = tk.Entry(top, width=10, font=FONT,
                        bg=CARD, fg=TEXT, insertbackground=TEXT,
                        disabledbackground="#333", disabledforeground=MUTED,
                        relief="flat", bd=4)
custom_entry.insert(0, "e.g. TSLA")
custom_entry.config(state="disabled")
custom_entry.pack(side=tk.LEFT)

run_btn = tk.Button(top, text="  ▶  Run Analysis  ", font=FONT_B,
                    bg=ACCENT, fg="white", activebackground="#9b7de0",
                    relief="flat", cursor="hand2", padx=8, pady=4,
                    command=run_analysis)
run_btn.pack(side=tk.LEFT, padx=14)

status_label = tk.Label(top, text="Select a stock and press  ▶  Run Analysis  to start.",
                        font=FONT, bg=DARK, fg=MUTED)
status_label.pack(side=tk.LEFT)

# ── Main body (3 columns: plot buttons | image | results) ────────────
body = tk.Frame(window, bg=DARK)
body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

# Left panel: plot buttons + history
left = tk.Frame(body, bg=CARD, width=180, padx=8, pady=8)
left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
left.pack_propagate(False)

tk.Label(left, text="📊  View Plots", font=FONT_B, bg=CARD, fg=TEXT).pack(pady=(4, 6))
for label_text, fname in PLOTS:
    btn = tk.Button(left, text=label_text, font=FONT, width=20,
                    bg="#35354f", fg=TEXT, activebackground=ACCENT,
                    activeforeground="white", relief="flat",
                    cursor="hand2", pady=3,
                    command=lambda f=fname: show_plot(f))
    btn.pack(pady=2, fill=tk.X)

tk.Label(left, text="🕓  Run History", font=FONT_B, bg=CARD, fg=TEXT).pack(pady=(16, 4))
history_box = scrolledtext.ScrolledText(left, font=MONO, height=8,
                                        bg="#1a1a2a", fg="#aaaacc",
                                        relief="flat", state="disabled",
                                        wrap=tk.WORD)
history_box.pack(fill=tk.BOTH, expand=True)

# Centre panel: inline image display
centre = tk.Frame(body, bg=CARD, padx=4, pady=4)
centre.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

plot_title_label = tk.Label(centre, text="", font=("Segoe UI", 9, "italic"),
                             bg=CARD, fg=MUTED)
plot_title_label.pack()

image_label = tk.Label(centre,
                       text="Run an analysis and click a plot button to view it here.",
                       font=FONT, bg=CARD, fg=MUTED, wraplength=500)
image_label.pack(fill=tk.BOTH, expand=True)

# Right panel: results text
right = tk.Frame(body, bg=CARD, padx=4, pady=4)
right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

tk.Label(right, text="📋  Results", font=FONT_B, bg=CARD, fg=TEXT).pack(anchor="w", pady=(2, 4))
results_box = scrolledtext.ScrolledText(right, font=MONO,
                                        bg="#111120", fg="#c8ffc8",
                                        insertbackground=TEXT, relief="flat",
                                        state="disabled", wrap=tk.WORD)
results_box.pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    window.mainloop()
