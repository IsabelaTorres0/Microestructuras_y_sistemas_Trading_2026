import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backtest import backtest, run_pipeline  # noqa: E402
from src.indicators import load_bars  # noqa: E402
from src.strategy import Params, Position, check_exit, entry_conditions, size_position  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "btc_project_train.csv")


def make_bars(rows, atr=10.0):
    """rows: lista de (open, high, low, close, trigger)."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "entry_trigger"], index=idx)
    df["atr_14"] = atr
    return df


# (a) stop y target en la misma barra -> stop_loss
def test_same_bar_stop_and_target_closes_as_stop():
    pos = Position("long", 10, entry_price=100, stop_loss=90, take_profit=120)
    reason, price = check_exit(pos, bar_open=100, bar_high=125, bar_low=85)
    assert reason == "stop_loss"
    assert price == 90


# (b) el sizing regresa exactamente el riesgo en dólares presupuestado
@pytest.mark.parametrize("equity,entry,stop,rho", [
    (100_000, 30_000, 29_500, 0.02),
    (100_000, 100, 85, 0.02),
    (57_321.4, 42_310.0, 42_014.87, 0.01),
])
def test_sizing_returns_exact_risk_budget(equity, entry, stop, rho):
    units = size_position(equity, entry, stop, rho)
    assert units * (entry - stop) == pytest.approx(rho * equity, rel=1e-12)


# (c) nunca dos posiciones abiertas al mismo tiempo
def test_never_two_open_positions():
    rng = np.random.default_rng(0)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    rows = [(c, c + abs(rng.normal(0, 1.5)), c - abs(rng.normal(0, 1.5)), c, True) for c in close]
    res = backtest(make_bars(rows, atr=1.0), Params(commission_bps=0, slippage_bps=0))
    tr = res["trades"]
    assert len(tr) > 1
    # cada entrada ocurre estrictamente después de la salida anterior
    assert (tr["entrada"].iloc[1:].to_numpy() > tr["salida"].iloc[:-1].to_numpy()).all()
    # el número de unidades vivas nunca excede el de una sola posición
    assert set(res["equity"]["en_posicion"].unique()) <= {True, False}


# Golden file: equity final calculado a mano
def test_golden_take_profit():
    """
    Barra 0: señal al cierre (ATR = 10).
    Barra 1: entra al open 100 -> SL = 60, TP = 130.
             unidades = 0.02 * 100,000 / 40 = 50
             comisión entrada = 50 * 100 * 0.001 = 5
    Barra 2: high 112 >= 110 -> stop sube a break-even (100).
    Barra 3: high 131 >= 130 -> sale en TP a 130.
             comisión salida = 50 * 130 * 0.001 = 6.5
    Equity final = 100,000 + 50 * 30 - 5 - 6.5 = 101,488.5
    """
    rows = [(100, 101, 99, 100, True), (100, 105, 95, 102, False),
            (102, 112, 101, 110, False), (110, 131, 109, 125, False),
            (125, 126, 124, 125, False)]
    res = backtest(make_bars(rows), Params(commission_bps=10, slippage_bps=0))
    assert res["equity"]["equity"].iloc[-1] == pytest.approx(101_488.5, abs=1e-3)
    assert res["trades"]["motivo"].tolist() == ["take_profit"]


def test_golden_break_even():
    """
    Igual que el anterior pero en la barra 3 el precio regresa a 99 (< 100):
    entra en 100 con SL = 60; en la barra 2 se activa break-even; luego baja a 99 y sale en 100.
    PnL = 0 - 5 - 5 = -10
    Equity final = 99,990.0
    """
    rows = [(100, 101, 99, 100, True), (100, 105, 95, 102, False),
            (102, 112, 101, 110, False), (110, 111, 99, 100, False),
            (100, 101, 99, 100, False)]
    res = backtest(make_bars(rows), Params(commission_bps=10, slippage_bps=0))
    assert res["equity"]["equity"].iloc[-1] == pytest.approx(99_990.0, abs=1e-3)
    assert res["trades"]["motivo"].tolist() == ["break_even"]


# Truncamiento del pipeline completo
@pytest.mark.parametrize("t", [800, 1800, 3000])
def test_pipeline_truncation(t):
    df = load_bars(DATA, "4h")
    full = run_pipeline(df)
    part = run_pipeline(df.iloc[:t + 1])
    cols = ["sma_10", "sma_100", "rsi_14", "macd", "macd_signal", "bb_mid",
            "bb_high", "atr_14", "entry_signal", "entry_trigger"]
    pd.testing.assert_series_equal(full["data"][cols].iloc[t], part["data"][cols].iloc[-1],
                                   check_names=False)
    pd.testing.assert_series_equal(full["equity"].iloc[t], part["equity"].iloc[-1],
                                   check_names=False)


def test_default_atr_stop_multiplier_is_four():
    assert Params().sl_atr == 4.0


def test_entry_trigger_requires_macd_upward_crossover():
    ind = pd.DataFrame({
        "sma_10": [101, 102, 103, 104, 105],
        "sma_100": [90, 90, 90, 90, 90],
        "close": [100, 101, 102, 103, 104],
        "bb_mid": [95, 95, 95, 95, 95],
        "macd": [0.5, 1.2, 1.5, 1.6, 1.9],
        "macd_signal": [0.7, 1.0, 1.2, 1.3, 1.7],
        "rsi_14": [60, 62, 61, 65, 66],
        "bb_high": [110, 110, 110, 110, 110],
    })
    out = entry_conditions(ind)
    assert out["entry_signal"].tolist() == [False, True, True, True, True]
    assert out["entry_trigger"].tolist() == [False, True, False, False, False]
