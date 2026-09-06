"""
Downloads 1-minute OHLCV bars via yfinance and normalizes the resulting
DataFrame (yfinance sometimes returns MultiIndex columns for single tickers
depending on version/args).

Requires network access to Yahoo Finance — will not run in a sandboxed
environment with restricted egress.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse yfinance's occasional MultiIndex columns to a flat index."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def download_1min_bars(
    ticker: str,
    period: str = "1d",
    interval: str = "1m",
) -> pd.DataFrame:
    """
    Download 1-minute bars for `ticker` over `period`.

    Returns a DataFrame indexed by timestamp with columns
    Open, High, Low, Close, Volume. Raises ValueError if the download
    is empty (bad ticker, market closed with no recent session, etc.).
    """
    raw = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    raw = _flatten_columns(raw)

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for ticker={ticker!r}, period={period!r}, "
            f"interval={interval!r}. Check the symbol and try again "
            f"(1m data is only available for the last 7 days)."
        )

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Downloaded data for {ticker!r} is missing columns: {missing}")

    df = raw[REQUIRED_COLUMNS].dropna(how="any").sort_index()
    df.index.name = "timestamp"
    return df


def save_bars(df: pd.DataFrame, ticker: str, raw_data_dir: str) -> str:
    """Persist bars to CSV under raw_data_dir, return the file path."""
    os.makedirs(raw_data_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = os.path.join(raw_data_dir, f"{ticker}_{stamp}_1m.csv")
    df.to_csv(path)
    return path


def load_bars_csv(path: str) -> pd.DataFrame:
    """Reload previously saved bars from CSV."""
    df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    return df[REQUIRED_COLUMNS]
