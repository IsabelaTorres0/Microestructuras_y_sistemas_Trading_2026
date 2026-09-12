import pandas as pd
import numpy as np
import ta

# definimos los parametros para nuestro activo
#Basandonos en estos vamos a poder calcular los indicadores técnicos

def sma(df: pd.DataFrame, window: int = 20):
    return ta.trend.SMAIndicator(close=df['close'], window=window).sma_indicator()

def ema(df: pd.DataFrame, window: int = 20):
    return ta.trend.EMAIndicator(close=df['close'], window=window).ema_indicator()

def bollinger(df: pd.DataFrame, window: int = 20, n_std: float = 2.5): # las dos lineas de pisos y techos
    bb = ta.volatility.BollingerBands(df["close"], window=window, window_dev=n_std)
    return pd.DataFrame({
        "bb_high" : bb.bollinger_hband(), 
        "bb_mid" : bb.bollinger_mavg(),
        "bb_low" : bb.bollinger_lband(),
        "bb_pct" : bb.bollinger_pband(),
    }, index=df.index)

def atr(df: pd.DataFrame, window: int = 14):
    return ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=window).average_true_range()

def rsi(df: pd.DataFrame, window: int = 14):
    return ta.momentum.RSIIndicator(close=df['close'], window=window).rsi()

def macd(df: pd.DataFrame, slow: int = 26, fast: int = 12, signal: int = 9):
    n = ta.trend.MACD(
        df["close"], window_slow=slow, window_fast=fast, window_sign=signal
    )
    return pd.DataFrame({
        "macd" : n.macd(),
        "macd_signal" : n.macd_signal(),
        "macd_diff" : n.macd_diff(),
    }, index=df.index)

def stochastic(df: pd.DataFrame, window: int = 14, smooth: int = 3):
    st = ta.momentum.StochasticOscillator(
        high=df['high'], low=df['low'], close=df['close'], window=window, smooth_window=smooth
    )
    return pd.DataFrame({
        "stoch_k" : st.stoch(),
        "stoch_d" : st.stoch_signal(),
    }, index=df.index)


def add_indicators(df: pd.DataFrame):
    out = df.copy()
    out["sma_10"] = sma(df, 10)
    out["sma_40"] = sma(df, 40)
    out["sma_100"] = sma(df, 100)
    out["ema_20"] = ema(df,20)
    out = out.join(bollinger(df, 20, 2.5))
    out["atr_14"] = atr(df, 14)
    out["rsi_14"] = rsi(df, 14)
    out = out.join(macd(df, 26, 12, 9))
    out = out.join(stochastic(df, 14, 3))
    return out

