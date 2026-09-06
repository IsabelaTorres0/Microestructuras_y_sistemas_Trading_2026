"""
Quick sanity-check script: tests a list of candidate tickers to see which
ones (a) still have live 1-minute data on Yahoo Finance, and (b) look
genuinely low-volume compared to SPY.

Run this BEFORE main.py to pick a safe ticker for TICKER_ILLIQUID:

    python check_ticker.py
"""

import yfinance as yf

CANDIDATES = [
    "SPY",     # reference liquid ticker
    "GEVO",
    "UAMY",
    "SIEB",
    "BTCS",
    "IMTE",
    "TNXP",
]

for ticker in CANDIDATES:
    try:
        df = yf.download(ticker, period="1d", interval="1m",
                          auto_adjust=False, progress=False)
        if df is None or df.empty:
            print(f"{ticker:8s} -> NO DATA (likely delisted or no recent session)")
            continue

        avg_vol_1m = df["Volume"].mean()
        n_bars = len(df)
        print(f"{ticker:8s} -> OK  | {n_bars:4d} bars | avg 1-min volume ~ {avg_vol_1m:,.0f}")

    except Exception as e:
        print(f"{ticker:8s} -> ERROR: {e}")
