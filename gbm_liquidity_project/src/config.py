"""
Central configuration: tickers, download window, and GBM simulation params.
Edit TICKER_LIQUID / TICKER_ILLIQUID after checking average daily volume
(see README "Choosing tickers").
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # --- Data download ---
    ticker_liquid: str = "SPY"
    ticker_illiquid: str = "GEVO"
    period: str = "1d"       # yfinance 1m data only covers the last 7 days
    interval: str = "1m"

    # --- Paths ---
    raw_data_dir: str = "data/raw"
    outputs_dir: str = "outputs"

    # --- GBM simulation ---
    n_paths: int = 10_000
    trading_minutes_per_day: int = 390  # 6.5h NYSE session
    trading_days_per_year: int = 252

    # which ticker's realized returns to calibrate the GBM from
    calibration_ticker: str = "liquid"  # "liquid" or "illiquid"


SETTINGS = Settings()
