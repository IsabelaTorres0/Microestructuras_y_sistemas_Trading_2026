"""Motor de backtest orientado a eventos (barra por barra), función pura.

Estado explícito: cash, posición y equity. No modifica sus entradas.
Convenciones:
  * La señal se evalúa al cierre de la barra t y se ejecuta al OPEN de t+1.
  * SL/TP se calculan con el ATR de la barra t (la de la señal).
  * Stop y target en la misma barra -> se asume stop.
  * Una sola posición a la vez; sin periodo máximo de holding.
"""
from dataclasses import replace

import pandas as pd

from .strategy import Params, check_exit, open_long, update_break_even


def backtest(data: pd.DataFrame, params: Params = Params()) -> dict:
    """data debe tener columnas open, high, low, close, atr_14 y entry_trigger."""
    p = params
    fee = p.commission_bps / 1e4
    slip = p.slippage_bps / 1e4

    o = data["open"].to_numpy(); h = data["high"].to_numpy()
    l = data["low"].to_numpy(); c = data["close"].to_numpy()
    atr_v = data["atr_14"].to_numpy(); trig = data["entry_trigger"].to_numpy()
    idx = data.index

    cash = p.initial_cash
    pos = None
    pending_entry = False
    equity_prev = p.initial_cash
    rows, trades = [], []

    for i in range(len(data)):
        t = idx[i]
        entered_now = False

        # 1) Ejecutar entrada pendiente al open
        if pending_entry and pos is None:
            fill = o[i] * (1 + slip)
            pos = open_long(t, fill, atr_v[i - 1], equity_prev, cash, p)
            cash -= pos.units * fill + pos.entry_fee
            entered_now = True
        pending_entry = False

        # 2) Revisar salidas intrabar
        if pos is not None:
            ex = check_exit(pos, o[i], h[i], l[i], is_entry_bar=entered_now)
            if ex is not None:
                reason, level = ex
                fill = level * (1 - slip)
                exit_fee = pos.units * fill * fee
                cash += pos.units * fill - exit_fee
                pnl = pos.units * (fill - pos.entry_price) - pos.entry_fee - exit_fee
                trades.append({
                    "entrada": pos.entry_time, "salida": t,
                    "precio_entrada": pos.entry_price, "precio_salida": fill,
                    "stop_inicial": pos.entry_price - p.sl_atr * pos.atr_at_entry,
                    "take_profit": pos.take_profit, "unidades": pos.units,
                    "nocional": pos.units * pos.entry_price,
                    "costos": pos.entry_fee + exit_fee, "pnl": pnl,
                    "retorno_%": 100 * pnl / (pos.units * pos.entry_price),
                    "R": pnl / (pos.units * p.sl_atr * pos.atr_at_entry),
                    "motivo": reason, "barras": idx.get_loc(t) - idx.get_loc(pos.entry_time) + 1,
                })
                pos = None
            else:
                update_break_even(pos, h[i], p)

        # 3) Marcar a mercado
        units = pos.units if pos is not None else 0.0
        equity = cash + units * c[i]
        rows.append((t, cash, units, equity, pos is not None))
        equity_prev = equity

        # 4) Señal al cierre -> orden para el open siguiente
        if pos is None and bool(trig[i]):
            pending_entry = True

    eq = pd.DataFrame(rows, columns=["datetime", "cash", "units", "equity", "en_posicion"]).set_index("datetime")
    tr = pd.DataFrame(trades)
    open_pos = None
    if pos is not None:
        open_pos = {"entrada": pos.entry_time, "precio_entrada": pos.entry_price,
                    "stop": pos.stop_loss, "take_profit": pos.take_profit,
                    "unidades": pos.units, "pnl_no_realizado": pos.units * (c[-1] - pos.entry_price) - pos.entry_fee}
    return {"equity": eq, "trades": tr, "open_position": open_pos, "params": p}


def run_pipeline(ohlc: pd.DataFrame, params: Params = Params()) -> dict:
    """indicadores -> señales -> combinación -> backtest."""
    from .indicators import add_indicators
    from .strategy import entry_conditions
    ind = add_indicators(ohlc)
    sig = entry_conditions(ind, params)
    data = ind.join(sig)
    res = backtest(data, params)
    res["data"] = data
    return res


def with_costs(params: Params, round_trip_bps: float) -> Params:
    """Params con un costo total de ida y vuelta (repartido mitad por lado)."""
    return replace(params, commission_bps=round_trip_bps / 2, slippage_bps=0.0)
