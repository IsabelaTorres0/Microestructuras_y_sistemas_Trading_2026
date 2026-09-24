"""Indicadores técnicos (SPEC §2).

Todos son causales: el valor en t solo usa barras <= t.
Se usan medias simples con min_periods = ventana, así que si cualquier
barra de la ventana no tiene precio el resultado es NaN (regla de NaN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    """Media móvil simple. Familia: tendencia."""
    return close.rolling(window, min_periods=window).mean()


def rsi_cutler(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI de Cutler (medias simples de ganancias y pérdidas). Familia: momentum."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()

    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    # Casos límite: sin pérdidas -> 100; sin movimiento -> 50.
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi.where(avg_gain.notna() & avg_loss.notna())


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range simple. Familia: volatilidad (solo para SL/TP)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1, skipna=False)
    return tr.rolling(window, min_periods=window).mean()
