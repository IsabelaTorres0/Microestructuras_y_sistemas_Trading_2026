"""Métricas de desempeño: rendimiento, Sharpe, drawdown, Calmar, turnover, win rate."""
import numpy as np
import pandas as pd

BARS_PER_YEAR_1H = 24 * 365  # cripto opera 24/7


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Barras por año según el espaciado mediano del índice (4 h -> 2,190)."""
    step = pd.Series(index).diff().median()
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
        "Costos totales ($)": tr["costos"].sum() if len(tr) else 0.0,
        "Turnover anual (x equity)": turnover(tr, eq),
        "Tiempo en mercado (%)": 100 * res["equity"]["en_posicion"].mean(),
        "Holding promedio (barras)": tr["barras"].mean() if len(tr) else 0.0,
        "Holding promedio (horas)": (tr["salida"] - tr["entrada"]).dt.total_seconds().mean() / 3600 if len(tr) else 0.0,
        "Veredicto (Calmar)": viability(cal),
    })
