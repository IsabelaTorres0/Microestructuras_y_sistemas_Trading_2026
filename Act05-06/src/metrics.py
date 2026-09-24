"""Métricas de desempeño.

Cripto opera 24/7: 288 barras de 5 min por día * 365 días.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .strategy import StrategyParams

PERIODS_PER_YEAR = 288 * 365


def returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().fillna(0.0)


def sharpe_ratio(equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = returns(equity)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0


def annual_return(equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    years = (len(equity) - 1) / periods_per_year
    total = equity.iloc[-1] / equity.iloc[0]
    return float(total ** (1 / years) - 1) if years > 0 and total > 0 else float("nan")


def annual_volatility(equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    return float(returns(equity).std(ddof=1) * np.sqrt(periods_per_year))


def drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown en cada barra, como fracción negativa desde el máximo previo."""
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def calmar_ratio(equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    mdd = max_drawdown(equity)
    return annual_return(equity, periods_per_year) / abs(mdd) if mdd < 0 else float("nan")


def turnover(trades: pd.DataFrame, equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Nocional operado (entradas + salidas) / equity promedio, anualizado."""
    traded = (trades["shares"] * (trades["entry_price"] + trades["exit_price"])).sum()
    years = len(equity) / periods_per_year
    return float(traded / equity.mean() / years) if years > 0 else float("nan")


def win_rate(trades: pd.DataFrame) -> float:
    return float((trades["pnl"] > 0).mean()) if len(trades) else float("nan")


def breakeven_win_rate(reward_risk: float, cost_in_risk_units: float) -> float:
    """p* = (1 + c) / (1 + R)  (SPEC §8)."""
    return (1 + cost_in_risk_units) / (1 + reward_risk)


def theoretical_p_star(params: StrategyParams, median_atr_bps: float) -> float:
    """p* del SPEC: usa el SL mediano de train."""
    r = params.tp_atr_mult / params.sl_atr_mult
    sl_bps = params.sl_atr_mult * median_atr_bps
    return breakeven_win_rate(r, params.round_trip_bps / sl_bps)


def per_trade_p_star(trades: pd.DataFrame, params: StrategyParams) -> float:
    """p* promedio usando el SL real de cada operación."""
    if not len(trades):
        return float("nan")
    r = params.tp_atr_mult / params.sl_atr_mult
    c = params.round_trip_bps / trades["sl_distance_bps"]
    return float(breakeven_win_rate(r, c).mean())


def summary(result, median_atr_bps: float) -> dict:
    eq = result.equity["equity"]
    tr = result.trades
    p = result.params
    wins, losses = tr.loc[tr.pnl > 0, "pnl"], tr.loc[tr.pnl <= 0, "pnl"]
    return {
        "final_equity": float(eq.iloc[-1]),
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "annual_return": annual_return(eq),
        "annual_volatility": annual_volatility(eq),
        "sharpe": sharpe_ratio(eq),
        "max_drawdown": max_drawdown(eq),
        "calmar": calmar_ratio(eq),
        "n_trades": int(len(tr)),
        "trades_per_day": len(tr) / (len(eq) / 288),
        "turnover_annual": turnover(tr, eq) if len(tr) else 0.0,
        "exposure": float((result.equity["shares"] > 0).mean()),
        "win_rate_empirical": win_rate(tr),
        "p_star_theoretical": theoretical_p_star(p, median_atr_bps),
        "p_star_per_trade": per_trade_p_star(tr, p),
        "avg_win": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss": float(losses.mean()) if len(losses) else float("nan"),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("nan"),
        "avg_r_multiple": float(tr["r_multiple"].mean()) if len(tr) else float("nan"),
        "total_fees": float((tr["entry_fee"] + tr["exit_fee"]).sum()),
        "exit_reasons": tr["exit_reason"].value_counts().to_dict(),
    }
