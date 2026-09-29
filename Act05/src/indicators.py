"""Carga de datos e indicadores técnicos (todos causales: solo usan datos hasta t)."""
import numpy as np
import pandas as pd


def load_bars(path: str, freq: str = "4h") -> pd.DataFrame:
    """Lee el CSV de 5 minutos, limpia NaN y agrega a barras de `freq` (OHLC)."""
    raw = pd.read_csv(path)
    df = raw.loc[:, ["Datetime", "Open", "High", "Low", "Close"]].rename(
        columns={"Datetime": "datetime", "Open": "open", "High": "high",
                 "Low": "low", "Close": "close"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime").dropna().sort_index()
    h = df.resample(freq, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"})
    return h.dropna()


def load_hourly(path: str) -> pd.DataFrame:
    return load_bars(path, "1h")


# ---------- Familia: tendencia ----------
def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, window: int) -> pd.Series:
    return close.ewm(span=window, adjust=False, min_periods=window).mean()


# ---------- Familia: momentum ----------
def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(avg_loss != 0, 100.0).where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_diff": line - sig})


# ---------- Familia: volatilidad ----------
def bollinger(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=0)
    high, low = mid + n_std * std, mid - n_std * std
    pct = (close - low) / (high - low)
    return pd.DataFrame({"bb_high": high, "bb_mid": mid, "bb_low": low, "bb_pct": pct})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sma_10"] = sma(out["close"], 10)
    out["sma_100"] = sma(out["close"], 100)
    out["rsi_14"] = rsi(out["close"], 14)
    out = out.join(macd(out["close"], 12, 26, 9))
    out = out.join(bollinger(out["close"], 20, 2.0))
    out["atr_14"] = atr(out, 14)
    return out
