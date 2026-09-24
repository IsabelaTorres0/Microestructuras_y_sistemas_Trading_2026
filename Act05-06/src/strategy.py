"""Lógica de la estrategia (SPEC §2–§5, §7).

Contiene:
- StrategyParams: todos los parámetros del SPEC en un solo lugar.
- combine_signals(): la Entry rule (función de combinación).
- compute_signals(): pipeline indicadores -> señales -> score.
- Position: estructura inmutable de una posición abierta.
- size_position(): sizing de riesgo fijo fraccional.
- check_intrabar_exit(): SL/TP con regla de empate y de gaps.
- PositionBook: máquina de estados que prohíbe dos posiciones abiertas.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .indicators import atr, rsi_cutler, sma


# ---------------------------------------------------------------------------
# Parámetros (SPEC)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StrategyParams:
    # Features
    fast_window: int = 10
    slow_window: int = 40
    rsi_window: int = 14
    atr_window: int = 14
    # Entry rule
    w_trend: float = 0.5
    w_mom: float = 0.5
    rsi_scale: float = 20.0
    theta: float = 0.75
    # Exit rule
    sl_atr_mult: float = 4.0
    tp_atr_mult: float = 8.0
    max_holding_bars: int = 48
    # Sizing
    initial_capital: float = 100_000.0
    risk_fraction: float = 0.001  # 0.10 % del equity
    max_leverage: float = 1.0
    # Costs (por lado)
    fee_rate: float = 0.0010  # 10 bps
    slippage_rate: float = 0.0001  # 1 bp

    @property
    def round_trip_bps(self) -> float:
        return 2 * (self.fee_rate + self.slippage_rate) * 1e4


# ---------------------------------------------------------------------------
# Entry rule
# ---------------------------------------------------------------------------
def combine_signals(
    s_trend: pd.Series, s_mom: pd.Series, w_trend: float = 0.5, w_mom: float = 0.5
) -> pd.Series:
    """score_t = w_trend * s_trend_t + w_mom * s_mom_t  (SPEC §3).

    Si cualquiera de las dos señales es NaN, el score es NaN (no hay señal).
    """
    return w_trend * s_trend + w_mom * s_mom


def compute_signals(bars: pd.DataFrame, p: StrategyParams = StrategyParams()) -> pd.DataFrame:
    """Pipeline completo: indicadores -> señales normalizadas -> score.

    Devuelve una copia de `bars` con columnas añadidas. No modifica la entrada.
    """
    out = bars.copy()
    close = out["Close"]

    out["sma_fast"] = sma(close, p.fast_window)
    out["sma_slow"] = sma(close, p.slow_window)
    out["rsi"] = rsi_cutler(close, p.rsi_window)
    out["atr"] = atr(out["High"], out["Low"], close, p.atr_window)

    diff = out["sma_fast"] - out["sma_slow"]
    out["s_trend"] = np.sign(diff)  # NaN se conserva
    out["s_mom"] = ((out["rsi"] - 50.0) / p.rsi_scale).clip(-1.0, 1.0)
    out["score"] = combine_signals(out["s_trend"], out["s_mom"], p.w_trend, p.w_mom)

    out["entry_signal"] = (out["score"] >= p.theta) & out["atr"].notna()
    out["exit_signal"] = out["score"] <= -p.theta
    return out


# ---------------------------------------------------------------------------
# Posición
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Position:
    side: str  # solo "long" en esta estrategia
    shares: float
    entry_price: float  # precio de ejecución (ya con slippage)
    stop_loss: float
    take_profit: float
    entry_time: Optional[pd.Timestamp] = None
    entry_fee: float = 0.0
    capped: bool = False  # True si el tope de apalancamiento redujo el tamaño

    def __post_init__(self) -> None:
        if self.side != "long":
            raise ValueError("La estrategia es solo largo")
        if self.shares <= 0:
            raise ValueError("shares debe ser > 0")
        if not (self.stop_loss < self.entry_price < self.take_profit):
            raise ValueError("Se requiere stop_loss < entry_price < take_profit")

    @property
    def risk_dollars(self) -> float:
        return self.shares * (self.entry_price - self.stop_loss)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def size_position(
    equity: float,
    entry_price: float,
    stop_loss: float,
    risk_fraction: float,
    max_leverage: float = 1.0,
    fee_rate: float = 0.0,
) -> tuple[float, bool]:
    """Riesgo fijo fraccional (SPEC §5).

    shares = (equity * risk_fraction) / (entry - stop), con tope de nocional
    tal que nocional * (1 + fee) <= equity * max_leverage.
    Devuelve (shares, capped).
    """
    stop_distance = entry_price - stop_loss
    if stop_distance <= 0:
        raise ValueError("El stop debe estar por debajo de la entrada en un largo")

    shares = equity * risk_fraction / stop_distance
    max_shares = equity * max_leverage / (entry_price * (1 + fee_rate))
    if shares > max_shares:
        return max_shares, True
    return shares, False


# ---------------------------------------------------------------------------
# Salidas intrabar
# ---------------------------------------------------------------------------
def check_intrabar_exit(
    pos: Position, bar_open: float, bar_high: float, bar_low: float, is_entry_bar: bool = False
) -> Optional[tuple[float, str]]:
    """Revisa SL/TP dentro de una barra (SPEC §7).

    - Gap: si la barra abre en o bajo el SL, sale al Open (stop_loss).
    - Empate: si Low <= SL y High >= TP en la misma barra, gana el stop.
    - Si abre sobre el TP, sale al TP (no se regala la mejora).
    Devuelve (precio_de_salida_sin_slippage, motivo) o None.
    En la barra de entrada no se aplica la regla de gap: la entrada ocurrió al Open.
    """
    if not is_entry_bar and bar_open <= pos.stop_loss:
        return bar_open, "stop_loss"
    if bar_low <= pos.stop_loss:  # incluye el caso de empate: el stop va primero
        return pos.stop_loss, "stop_loss"
    if bar_high >= pos.take_profit:
        return pos.take_profit, "take_profit"
    return None


# ---------------------------------------------------------------------------
# Máquina de estados
# ---------------------------------------------------------------------------
class PositionAlreadyOpenError(RuntimeError):
    pass


@dataclass(frozen=True)
class PositionBook:
    """Estado FLAT (position=None) o LONG (position=Position). Inmutable."""

    position: Optional[Position] = None
    bars_held: int = 0

    @property
    def is_open(self) -> bool:
        return self.position is not None

    def open(self, pos: Position) -> "PositionBook":
        if self.is_open:
            raise PositionAlreadyOpenError("Ya hay una posición abierta")
        return PositionBook(position=pos, bars_held=0)

    def close(self) -> "PositionBook":
        if not self.is_open:
            raise RuntimeError("No hay posición que cerrar")
        return PositionBook(position=None, bars_held=0)

    def tick(self) -> "PositionBook":
        """Suma una barra al contador de holding."""
        if not self.is_open:
            return self
        return PositionBook(position=self.position, bars_held=self.bars_held + 1)
