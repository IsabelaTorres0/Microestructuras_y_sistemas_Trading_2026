"""Métricas de desempeño: rendimiento, Sharpe, drawdown, Calmar, turnover, win rate."""
import numpy as np
import pandas as pd

BARS_PER_YEAR_1H = 24 * 365  # cripto opera 24/7


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Barras por año según el espaciado mediano del índice (4 h -> 2,190)."""
    step = pd.Series(index).diff().median()
    if step >= pd.Timedelta(days=1):
        return 252.0                      
    return pd.Timedelta(days=365) / step

def drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    return float(drawdown(equity).min())


def cagr(equity: pd.Series) -> float:
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 24 * 3600)
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def sharpe(equity: pd.Series, periods: float = None) -> float:
    periods = periods or bars_per_year(equity.index)
    r = equity.pct_change().dropna()
    if r.std() == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods))


def calmar(equity: pd.Series) -> float:
    mdd = max_drawdown(equity)
    return float(cagr(equity) / abs(mdd)) if mdd < 0 else np.nan


def turnover(trades: pd.DataFrame, equity: pd.Series) -> float:
    """Nocional operado (entrada + salida) / equity promedio, anualizado."""
    if trades.empty:
        return 0.0
    traded = (trades["unidades"] * (trades["precio_entrada"] + trades["precio_salida"])).sum()
    years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 24 * 3600)
    return float(traded / equity.mean() / years)


def win_rate(trades: pd.DataFrame) -> float:
    return float((trades["pnl"] > 0).mean()) if len(trades) else np.nan


def expectancy_R(trades: pd.DataFrame) -> float:
    """Resultado promedio por operación en múltiplos de R (riesgo inicial)."""
    return float(trades["R"].mean()) if len(trades) else np.nan


def breakeven_winrate_theoretical(params, cost_R: float = 0.0) -> float:
    """p* del SPEC: supone que TODA pérdida es -1 R y toda ganancia +TP/SL R.
    Con costos c (en R): p* = (1 + c) / (1 + R)."""
    return (1.0 + cost_R) / (1.0 + params.reward_risk)


def breakeven_winrate_payoff(trades: pd.DataFrame) -> float:
    """p* con el payoff REAL: p* = |pérdida media| / (ganancia media + |pérdida media|).
    Toma en cuenta que el break-even convierte muchas pérdidas de -1 R en ~ -0.05 R."""
    if not len(trades):
        return np.nan
    w = trades.loc[trades["pnl"] > 0, "pnl"]
    lo = trades.loc[trades["pnl"] <= 0, "pnl"]
    if not len(w) or not len(lo):
        return np.nan
    return float(abs(lo.mean()) / (w.mean() + abs(lo.mean())))


def winrate_table(trades: pd.DataFrame, params, cost_R: float) -> pd.DataFrame:
    """Win rate teórico (p* del SPEC) contra empírico (de las operaciones)."""
    rows = {
        "p* teórico sin costos  = 1/(1+R)": breakeven_winrate_theoretical(params, 0.0),
        "p* teórico con costos  = (1+c)/(1+R)": breakeven_winrate_theoretical(params, cost_R),
        "p* con payoff empírico = |L|/(W+|L|)": breakeven_winrate_payoff(trades),
        "Win rate empírico (PnL neto > 0)": win_rate(trades),
        "% de operaciones que tocan el TP": float((trades["motivo"] == "take_profit").mean()) if len(trades) else np.nan,
    }
    return pd.DataFrame({"Valor (%)": {k: 100 * v for k, v in rows.items()}})


def viability(calmar_ratio: float) -> str:
    if np.isnan(calmar_ratio) or calmar_ratio < 0.5:
        return "NO VIABLE"
    if calmar_ratio < 1.0:
        return "MARGINAL"
    return "VIABLE"


def summary(res: dict) -> pd.Series:
    eq = res["equity"]["equity"]
    tr = res["trades"]
    p = res["params"]
    wins = tr[tr["pnl"] > 0]["pnl"] if len(tr) else pd.Series(dtype=float)
    losses = tr[tr["pnl"] <= 0]["pnl"] if len(tr) else pd.Series(dtype=float)
    cal = calmar(eq)
    return pd.Series({
        "Capital inicial ($)": eq.iloc[0] if len(eq) else p.initial_cash,
        "Capital final ($)": eq.iloc[-1],
        "Ganancia/Pérdida ($)": eq.iloc[-1] - p.initial_cash,
        "Rendimiento total (%)": 100 * (eq.iloc[-1] / p.initial_cash - 1),
        "Rendimiento anualizado CAGR (%)": 100 * cagr(eq),
        "Volatilidad anualizada (%)": 100 * eq.pct_change().std() * np.sqrt(bars_per_year(eq.index)),
        "Sharpe (anualizado)": sharpe(eq),
        "Drawdown máximo (%)": 100 * max_drawdown(eq),
        "Calmar": cal,
        "Operaciones cerradas": len(tr),
        "Ganadoras": int((tr["pnl"] > 0).sum()) if len(tr) else 0,
        "Perdedoras": int((tr["pnl"] <= 0).sum()) if len(tr) else 0,
        "Win rate empírico (%)": 100 * win_rate(tr),
        "Win rate break-even p* (%)": 100 * p.breakeven_winrate,
        "Ganancia promedio ($)": wins.mean() if len(wins) else 0.0,
        "Pérdida promedio ($)": losses.mean() if len(losses) else 0.0,
        "Profit factor": wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan,
        "p* con payoff empírico (%)": 100 * breakeven_winrate_payoff(tr),
        "Expectancy (R por operación)": expectancy_R(tr),
        "Costos totales ($)": tr["costos"].sum() if len(tr) else 0.0,
        "Turnover anual (x equity)": turnover(tr, eq),
        "Tiempo en mercado (%)": 100 * res["equity"]["en_posicion"].mean(),
        "Holding promedio (barras)": tr["barras"].mean() if len(tr) else 0.0,
        "Holding promedio (horas)": (tr["salida"] - tr["entrada"]).dt.total_seconds().mean() / 3600 if len(tr) else 0.0,
        "Veredicto (Calmar)": viability(cal),
    })
