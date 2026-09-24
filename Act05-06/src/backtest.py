"""Motor de backtest orientado a eventos (barra por barra), como función pura.

`backtest()` no modifica sus entradas, no usa estado global y siempre devuelve
el mismo resultado para las mismas entradas. El estado (cash, posición, equity)
es explícito y se actualiza en un ciclo sobre las barras.

Orden de eventos en cada barra válida t (SPEC §7):
  1. Ejecutar al Open las órdenes pendientes generadas al cierre de t-1
     (primero salidas, después entradas).
  2. Revisar SL/TP con High/Low (empate -> stop; gap bajo SL -> sale al Open).
  3. Sumar una barra al contador de holding.
  4. Al Close, leer señales y dejar órdenes pendientes para t+1
     (señal opuesta antes que holding máximo).
  5. Marcar a mercado: equity = cash + shares * Close.

Barras sin precio: no se ejecuta nada; una entrada pendiente se cancela,
una salida pendiente se conserva para la siguiente barra válida; el equity
se marca con el último Close conocido.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .strategy import (
    Position,
    PositionBook,
    StrategyParams,
    check_intrabar_exit,
    size_position,
)


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.DataFrame  # columnas: cash, shares, equity
    trades: pd.DataFrame  # una fila por operación cerrada
    open_position: Optional[Position]  # posición abierta al final (marcada a mercado)
    params: StrategyParams = field(default_factory=StrategyParams)


TRADE_COLUMNS = [
    "entry_time", "exit_time", "entry_price", "exit_price", "shares",
    "stop_loss", "take_profit", "exit_reason", "bars_held", "entry_fee",
    "exit_fee", "gross_pnl", "pnl", "risk_dollars", "r_multiple",
    "sl_distance_bps", "capped",
]


def backtest(signals: pd.DataFrame, params: StrategyParams = StrategyParams()) -> BacktestResult:
    """Corre la estrategia sobre `signals`.

    `signals` debe tener: Open, High, Low, Close, atr, entry_signal, exit_signal.
    """
    p = params
    idx = signals.index
    o = signals["Open"].to_numpy(dtype=float)
    h = signals["High"].to_numpy(dtype=float)
    lo = signals["Low"].to_numpy(dtype=float)
    c = signals["Close"].to_numpy(dtype=float)
    atr_v = signals["atr"].to_numpy(dtype=float)
    entry_sig = signals["entry_signal"].fillna(False).to_numpy(dtype=bool)
    exit_sig = signals["exit_signal"].fillna(False).to_numpy(dtype=bool)

    n = len(signals)
    cash_out = np.empty(n)
    shares_out = np.empty(n)
    equity_out = np.empty(n)

    # ---- Estado explícito ------------------------------------------------
    cash = p.initial_capital
    book = PositionBook()
    last_close = np.nan
    pending_entry_atr: Optional[float] = None
    pending_exit_reason: Optional[str] = None
    trades: list[dict] = []

    def close_trade(book: PositionBook, cash: float, raw_price: float, reason: str, t: int):
        pos = book.position
        fill = raw_price * (1 - p.slippage_rate)
        proceeds = pos.shares * fill
        fee = proceeds * p.fee_rate
        cash = cash + proceeds - fee
        gross = pos.shares * (fill - pos.entry_price)
        pnl = gross - pos.entry_fee - fee
        trades.append(
            dict(
                entry_time=pos.entry_time, exit_time=idx[t], entry_price=pos.entry_price,
                exit_price=fill, shares=pos.shares, stop_loss=pos.stop_loss,
                take_profit=pos.take_profit, exit_reason=reason, bars_held=book.bars_held,
                entry_fee=pos.entry_fee, exit_fee=fee, gross_pnl=gross, pnl=pnl,
                risk_dollars=pos.risk_dollars, r_multiple=pnl / pos.risk_dollars,
                sl_distance_bps=(pos.entry_price - pos.stop_loss) / pos.entry_price * 1e4,
                capped=pos.capped,
            )
        )
        return book.close(), cash

    for t in range(n):
        valid = not (np.isnan(o[t]) or np.isnan(h[t]) or np.isnan(lo[t]) or np.isnan(c[t]))

        if not valid:
            pending_entry_atr = None  # la señal caduca
            book = book.tick() if book.is_open else book
            shares = book.position.shares if book.is_open else 0.0
            cash_out[t], shares_out[t] = cash, shares
            equity_out[t] = cash + shares * (last_close if shares else 0.0)
            continue

        entered_this_bar = False

        # 1a. Salida pendiente al Open
        if pending_exit_reason is not None and book.is_open:
            book, cash = close_trade(book, cash, o[t], pending_exit_reason, t)
        pending_exit_reason = None

        # 1b. Entrada pendiente al Open
        if pending_entry_atr is not None and not book.is_open:
            entry = o[t] * (1 + p.slippage_rate)
            sl = entry - p.sl_atr_mult * pending_entry_atr
            tp = entry + p.tp_atr_mult * pending_entry_atr
            if sl > 0:
                shares, capped = size_position(
                    cash, entry, sl, p.risk_fraction, p.max_leverage, p.fee_rate
                )
                fee = shares * entry * p.fee_rate
                cash = cash - shares * entry - fee
                pos = Position("long", shares, entry, sl, tp, idx[t], fee, capped)
                book = book.open(pos)
                entered_this_bar = True
        pending_entry_atr = None

        # 2. SL / TP intrabar
        if book.is_open:
            hit = check_intrabar_exit(book.position, o[t], h[t], lo[t], entered_this_bar)
            if hit is not None:
                book = book.tick()
                book, cash = close_trade(book, cash, hit[0], hit[1], t)

        # 3. Contador de holding
        if book.is_open:
            book = book.tick()

        # 4. Señales al cierre -> órdenes para t+1
        if book.is_open:
            if exit_sig[t]:
                pending_exit_reason = "signal"
            elif book.bars_held >= p.max_holding_bars:
                pending_exit_reason = "max_holding"
        elif entry_sig[t] and not np.isnan(atr_v[t]):
            pending_entry_atr = atr_v[t]

        # 5. Marcar a mercado
        last_close = c[t]
        shares = book.position.shares if book.is_open else 0.0
        cash_out[t], shares_out[t] = cash, shares
        equity_out[t] = cash + shares * c[t]

    equity = pd.DataFrame({"cash": cash_out, "shares": shares_out, "equity": equity_out}, index=idx)
    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    return BacktestResult(equity=equity, trades=trades_df, open_position=book.position, params=p)


def run_pipeline(bars: pd.DataFrame, params: StrategyParams = StrategyParams()) -> BacktestResult:
    """indicadores -> señales -> combinación -> backtest."""
    from .strategy import compute_signals

    return backtest(compute_signals(bars, params), params)
