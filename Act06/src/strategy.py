"""Regla de entrada, estructura de posición, sizing con rho y reglas de salida."""
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Params:
    rho: float = 0.02            # fracción del equity arriesgada por operación
    sl_atr: float = 4.0          # stop-loss = entrada - 4.0 * ATR
    tp_atr: float = 3.0          # take-profit = entrada + 3 * ATR
    be_atr: float = 1.0          # al ganar 1 ATR el stop sube al precio de entrada
    rsi_low: float = 50.0
    rsi_high: float = 70.0
    commission_bps: float = 10.0  # por lado (0.10 %)
    slippage_bps: float = 5.0     # por lado (0.05 %)
    initial_cash: float = 100_000.0

    @property
    def reward_risk(self) -> float:
        return self.tp_atr / self.sl_atr

    @property
    def breakeven_winrate(self) -> float:
        """p* = 1 / (1 + R), sin costos."""
        return 1.0 / (1.0 + self.reward_risk)


@dataclass
class Position:
    side: str            # "long" (la estrategia solo opera largos)
    units: float         # número de unidades (BTC)
    entry_price: float
    stop_loss: float
    take_profit: float
    entry_time: pd.Timestamp = None
    atr_at_entry: float = 0.0
    be_active: bool = False
    entry_fee: float = 0.0


# ---------------------------------------------------------------- Entry rule
def entry_conditions(ind: pd.DataFrame, p: Params = Params()) -> pd.DataFrame:
    """Combinación de indicadores (se evalúa al CIERRE de la barra t).

    C1 (tendencia):  SMA_10 > SMA_100  AND  close > BB_mid
    C2 (momentum):   MACD > señal  AND  rsi_low < RSI_14 < rsi_high  AND  close < BB_high
    trigger: solo cuando el MACD cruza hacia arriba su señal, manteniendo las demás condiciones.
    """
    c1 = (ind["sma_10"] > ind["sma_100"]) & (ind["close"] > ind["bb_mid"])
    c2 = ((ind["macd"] > ind["macd_signal"])
          & (ind["rsi_14"] > p.rsi_low) & (ind["rsi_14"] < p.rsi_high)
          & (ind["close"] < ind["bb_high"]))
    macd_up_cross = (ind["macd"] > ind["macd_signal"]) & (ind["macd"].shift(1) <= ind["macd_signal"].shift(1))
    signal = (c1 & c2).fillna(False).astype(bool)
    trigger = (signal & macd_up_cross).fillna(False).astype(bool)
    return pd.DataFrame({"c1_tendencia": c1.fillna(False), "c2_momentum": c2.fillna(False),
                         "entry_signal": signal, "entry_trigger": trigger}, index=ind.index)


# ---------------------------------------------------------------- Sizing
def size_position(equity: float, entry_price: float, stop_price: float, rho: float) -> float:
    """Sizing por riesgo fijo: unidades = rho * equity / (entrada - stop).

    Si el stop se toca se pierde exactamente rho * equity (antes de costos)."""
    risk_per_unit = entry_price - stop_price
    if risk_per_unit <= 0:
        raise ValueError("El stop debe estar por debajo de la entrada en un largo")
    return rho * equity / risk_per_unit


def open_long(time, price: float, atr_value: float, equity: float, cash: float,
              p: Params) -> Position:
    stop = price - p.sl_atr * atr_value
    tp = price + p.tp_atr * atr_value
    units = size_position(equity, price, stop, p.rho)
    fee_rate = p.commission_bps / 1e4
    max_units = cash / (price * (1 + fee_rate))       # sin apalancamiento
    units = min(units, max_units)
    return Position("long", units, price, stop, tp, time, atr_value,
                    False, units * price * fee_rate)


# ---------------------------------------------------------------- Exit rules
def check_exit(pos: Position, bar_open: float, bar_high: float, bar_low: float,
               is_entry_bar: bool = False) -> Optional[Tuple[str, float]]:
    """Devuelve (motivo, precio) si la barra toca SL/TP; None si sigue abierta.

    - Gap: si la barra abre por debajo del stop, se sale al open.
    - Empate intrabar (stop y target en la misma barra): gana el stop.
    - No hay periodo máximo: si no se toca nada, la posición sigue abierta."""
    stop_reason = "break_even" if pos.be_active else "stop_loss"
    if not is_entry_bar and bar_open <= pos.stop_loss:
        return stop_reason, bar_open
    if not is_entry_bar and bar_open >= pos.take_profit:
        return "take_profit", bar_open
    hit_stop = bar_low <= pos.stop_loss
    hit_tp = bar_high >= pos.take_profit
    if hit_stop:                       # incluye el empate -> stop (conservador)
        return stop_reason, pos.stop_loss
    if hit_tp:
        return "take_profit", pos.take_profit
    return None


def update_break_even(pos: Position, bar_high: float, p: Params) -> None:
    """3er parámetro de salida: si el máximo alcanza entrada + be_atr*ATR,
    el stop se mueve al precio de entrada (aplica desde la siguiente barra)."""
    if not pos.be_active and bar_high >= pos.entry_price + p.be_atr * pos.atr_at_entry:
        pos.stop_loss = max(pos.stop_loss, pos.entry_price)
        pos.be_active = True
