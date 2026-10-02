import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backtest import backtest, run_pipeline, scale_costs  # noqa: E402
from src.indicators import load_stock  # noqa: E402
from src import metrics as M  # noqa: E402
from src.strategy import Params, Position, check_exit, entry_conditions, size_position  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "AAPL.csv")


def make_bars(rows, atr=10.0):
    """rows: lista de (open, high, low, close, trigger)."""
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="1h")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "entry_trigger"], index=idx)
    df["atr_14"] = atr
    return df


# ======================================================================
# Tests unitarios de la Act 05
# ======================================================================

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
    assert (tr["entrada"].iloc[1:].to_numpy() > tr["salida"].iloc[:-1].to_numpy()).all()
    assert set(res["equity"]["en_posicion"].unique()) <= {True, False}


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


# ======================================================================
# Golden-file tests (Act 06): equity calculado a mano, NO con el código
# Todos usan ATR = 10, SL = 4 ATR, TP = 3 ATR, BE = 1 ATR, rho = 2 %
# ======================================================================

def test_golden_take_profit():
    """
    Comisión 10 bps, slippage 0.
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


GOLDEN_SL_ROWS = [(100, 101, 99, 100, True),   # b0: señal
                  (100, 105, 95, 100, False),  # b1: entra
                  (100, 102, 55, 58, False),   # b2: toca el stop
                  (58, 59, 57, 58, False)]     # b3: plano


def test_golden_stop_loss_with_slippage():
    """
    Costos del SPEC: comisión 10 bps + slippage 5 bps por lado.
    b1: fill = 100 * 1.0005 = 100.05
        SL = 100.05 - 40 = 60.05 ; TP = 100.05 + 30 = 130.05
        unidades = 0.02 * 100,000 / 40 = 50
        comisión entrada = 50 * 100.05 * 0.001 = 5.0025
        cash = 100,000 - 5,002.50 - 5.0025 = 94,992.4975
        equity(b1) = 94,992.4975 + 50 * 100 = 99,992.4975
    b2: low 55 <= 60.05 -> stop. fill = 60.05 * 0.9995 = 60.019975
        comisión salida = 50 * 60.019975 * 0.001 = 3.00099875
        cash = 94,992.4975 + 3,000.99875 - 3.00099875 = 97,990.49525125
    PnL = 50 * (60.019975 - 100.05) - 5.0025 - 3.00099875 = -2,009.50474875
    R   = -2,009.50474875 / (50 * 40) = -1.004752374375
    """
    res = backtest(make_bars(GOLDEN_SL_ROWS), Params())
    eq = res["equity"]["equity"]
    tr = res["trades"]
    assert eq.iloc[1] == pytest.approx(99_992.4975, abs=1e-6)
    assert eq.iloc[-1] == pytest.approx(97_990.49525125, abs=1e-6)
    assert tr["motivo"].tolist() == ["stop_loss"]
    assert tr["pnl"].iloc[0] == pytest.approx(-2_009.50474875, abs=1e-6)
    assert tr["R"].iloc[0] == pytest.approx(-1.004752374375, abs=1e-9)
    # La pérdida antes de costos es exactamente rho * equity = 2,000
    assert 50 * (60.05 - 100.05) == pytest.approx(-2_000.0)


GOLDEN_TWO_ROWS = [(100, 101, 99, 100, True),   # b0: señal 1
                   (100, 104, 96, 100, True),   # b1: entra; nueva señal IGNORADA (ya hay posición)
                   (50, 52, 48, 50, True),      # b2: gap abajo del stop -> sale al open; señal 2
                   (50, 55, 49, 54, False),     # b3: entra la 2a operación
                   (54, 61, 53, 60, False),     # b4: high 61 >= 60 -> break-even
                   (85, 90, 84, 88, False),     # b5: gap arriba del TP -> sale al open 85
                   (88, 89, 87, 88, False)]     # b6: plano
GOLDEN_TWO_EQUITY = [100_000.0, 99_995.0, 97_492.5, 97_685.0476875,
                     97_977.5251875, 99_192.03800625, 99_192.03800625]


def test_golden_two_trades_gaps_and_compounding():
    """
    Comisión 10 bps, slippage 0. Prueba: gap bajo el stop, señal ignorada con posición
    abierta, sizing con el equity actualizado, break-even y gap sobre el TP.

    Operación 1
      b1: entra en 100. SL 60, TP 130, unidades 50, comisión 5
          cash = 94,995 ; equity = 94,995 + 50*100 = 99,995
      b2: abre en 50 <= 60 -> sale al OPEN (50), no al stop.
          comisión = 50*50*0.001 = 2.5 ; cash = 94,995 + 2,500 - 2.5 = 97,492.5
          PnL1 = 50*(50-100) - 5 - 2.5 = -2,507.5
    Operación 2 (equity previo = 97,492.5)
      b3: entra en 50. SL 10, TP 80, BE a 60
          unidades = 0.02*97,492.5/40 = 48.74625
          comisión = 48.74625*50*0.001 = 2.4373125
          cash = 97,492.5 - 2,437.3125 - 2.4373125 = 95,052.7501875
          equity = 95,052.7501875 + 48.74625*54 = 97,685.0476875
      b4: high 61 >= 60 -> stop a 50 desde b5
          equity = 95,052.7501875 + 48.74625*60 = 97,977.5251875
      b5: abre en 85 >= 80 -> sale al OPEN (85)
          comisión = 48.74625*85*0.001 = 4.14343125
          cash = 95,052.7501875 + 4,143.43125 - 4.14343125 = 99,192.03800625
          PnL2 = 48.74625*35 - 2.4373125 - 4.14343125 = 1,699.53800625
    Equity final = 99,192.03800625
    """
    res = backtest(make_bars(GOLDEN_TWO_ROWS), Params(commission_bps=10, slippage_bps=0))
    eq = res["equity"]["equity"].to_numpy()
    tr = res["trades"]
    np.testing.assert_allclose(eq, GOLDEN_TWO_EQUITY, atol=1e-6)
    assert tr["motivo"].tolist() == ["stop_loss", "take_profit"]
    assert tr["precio_salida"].tolist() == [50, 85]
    assert tr["unidades"].iloc[1] == pytest.approx(48.74625, abs=1e-9)
    np.testing.assert_allclose(tr["pnl"], [-2_507.5, 1_699.53800625], atol=1e-6)
    assert res["open_position"] is None


# ======================================================================
# backtest() es una función pura
# ======================================================================
def test_backtest_is_pure():
    data = make_bars(GOLDEN_TWO_ROWS)
    before = data.copy(deep=True)
    r1 = backtest(data, Params())
    r2 = backtest(data, Params())
    pd.testing.assert_frame_equal(data, before)                       # no modifica la entrada
    pd.testing.assert_frame_equal(r1["equity"], r2["equity"])         # determinista
    pd.testing.assert_frame_equal(r1["trades"], r2["trades"])


# ======================================================================
# Costos y métricas
# ======================================================================
def test_scale_costs_30bps_equals_spec():
    p = scale_costs(Params(), 30)
    assert p.commission_bps == pytest.approx(10.0)
    assert p.slippage_bps == pytest.approx(5.0)


def test_max_drawdown_by_hand():
    eq = pd.Series([100, 120, 90, 110, 130], index=pd.date_range("2024", periods=5, freq="D"))
    assert M.max_drawdown(eq) == pytest.approx(90 / 120 - 1)      # -25 %


def test_breakeven_winrate_formulas():
    assert M.breakeven_winrate_theoretical(Params()) == pytest.approx(1 / (1 + 0.75))
    tr = pd.DataFrame({"pnl": [300.0, -100.0, -100.0]})
    assert M.breakeven_winrate_payoff(tr) == pytest.approx(100 / 400)


@pytest.mark.parametrize("t", [300, 700, 1100])
def test_pipeline_truncation(t):
    df = load_stock(DATA, "4h")
    full = run_pipeline(df)
    part = run_pipeline(df.iloc[:t + 1])
    cols = ["sma_10", "sma_100", "rsi_14", "macd", "macd_signal", "bb_mid",
            "bb_high", "atr_14", "c1_tendencia", "c2_momentum", "entry_signal", "entry_trigger"]
    pd.testing.assert_series_equal(full["data"][cols].iloc[t], part["data"][cols].iloc[-1],
                                   check_names=False)
    pd.testing.assert_series_equal(full["equity"].iloc[t], part["equity"].iloc[-1],
                                   check_names=False)
    # las operaciones cerradas hasta t también deben ser idénticas
    ft = full["trades"]
    ft = ft[ft["salida"] <= df.index[t]].reset_index(drop=True) if len(ft) else ft
 
    pt = part["trades"].reset_index(drop=True)
    if len(pt) == 0:                       # todavía no hay operaciones cerradas en t
        assert len(ft) == 0
    else:
        pd.testing.assert_frame_equal(ft, pt, check_dtype=False)