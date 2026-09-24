"""Golden-file test: 7 barras con el equity calculado a mano (no con el código).

Parámetros: capital 100,000; riesgo 1 %; SL = E - 4·ATR; TP = E + 8·ATR;
comisión 10 bps por lado; slippage 0. Señales al cierre de t, ejecución al Open de t+1.

Cálculo en papel
----------------
t0  Cierre: entry_signal, ATR = 1  -> orden de compra para t1.
t1  Compra al Open 100.  SL = 96, TP = 108.
    shares = 100,000 × 1 % / (100 − 96) = 1,000 / 4 = 250
    costo  = 250 × 100 = 25,000 ; comisión = 25
    cash   = 100,000 − 25,000 − 25 = 74,975
    High 103 < 108 y Low 99 > 96: sigue abierta.
    equity = 74,975 + 250 × 102 = 100,475
t2  High 109 ≥ TP 108 -> take_profit a 108.
    ingreso = 250 × 108 = 27,000 ; comisión = 27
    cash    = 74,975 + 27,000 − 27 = 101,948
    Cierre: entry_signal, ATR = 2 -> orden de compra para t3.
t3  Compra al Open 106.  SL = 106 − 8 = 98, TP = 106 + 16 = 122.
    shares = 101,948 × 1 % / 8 = 1,019.48 / 8 = 127.435
    costo  = 127.435 × 106 = 13,508.11 ; comisión = 13.50811
    cash   = 101,948 − 13,508.11 − 13.50811 = 88,426.38189
    Low 97 ≤ 98 -> stop_loss a 98.
    ingreso = 127.435 × 98 = 12,488.63 ; comisión = 12.48863
    cash    = 88,426.38189 + 12,488.63 − 12.48863 = 100,902.52326
t4  Cierre: entry_signal, ATR = 1 -> orden de compra para t5.
t5  Compra al Open 100.  SL = 96, TP = 108.
    shares = 100,902.52326 × 1 % / 4 = 252.25630815
    costo  = 25,225.630815 ; comisión = 25.225630815
    cash   = 100,902.52326 − 25,225.630815 − 25.225630815 = 75,651.666814185
    equity = 75,651.666814185 + 252.25630815 × 101 = 101,129.553937335
    Cierre: exit_signal -> orden de venta para t6.
t6  Venta al Open 103.
    ingreso = 252.25630815 × 103 = 25,982.39973945 ; comisión = 25.98239973945
    cash    = 75,651.666814185 + 25,982.39973945 − 25.98239973945 = 101,608.084153896
    EQUITY FINAL = 101,608.084153896
"""
import pandas as pd
import pytest

from src.backtest import backtest
from src.strategy import StrategyParams

GOLDEN_EQUITY = [
    100_000.0,
    100_475.0,
    101_948.0,
    100_902.52326,
    100_902.52326,
    101_129.553937335,
    101_608.084153896,
]
GOLDEN_REASONS = ["take_profit", "stop_loss", "signal"]


def golden_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open":  [100, 100, 102, 106, 100, 100, 103],
            "High":  [101, 103, 109, 107, 101, 102, 104],
            "Low":   [99,  99,  101, 97,  99,  99,  102],
            "Close": [100, 102, 107, 99,  100, 101, 104],
            "atr":   [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
            "entry_signal": [True, False, True, False, True, False, False],
            "exit_signal":  [False, False, False, False, False, True, False],
        },
        index=pd.date_range("2023-01-01", periods=7, freq="5min"),
    )


PARAMS = StrategyParams(risk_fraction=0.01, fee_rate=0.001, slippage_rate=0.0)


def test_golden_final_equity():
    res = backtest(golden_bars(), PARAMS)
    assert res.equity["equity"].iloc[-1] == pytest.approx(GOLDEN_EQUITY[-1], rel=1e-12)


def test_golden_equity_path():
    res = backtest(golden_bars(), PARAMS)
    assert res.equity["equity"].tolist() == pytest.approx(GOLDEN_EQUITY, rel=1e-12)


def test_golden_trades():
    res = backtest(golden_bars(), PARAMS)
    assert res.trades["exit_reason"].tolist() == GOLDEN_REASONS
    assert res.trades["shares"].tolist() == pytest.approx([250, 127.435, 252.25630815])
    assert res.open_position is None


def test_backtest_is_pure():
    bars = golden_bars()
    before = bars.copy()
    a = backtest(bars, PARAMS)
    b = backtest(bars, PARAMS)
    pd.testing.assert_frame_equal(bars, before)  # no modifica la entrada
    pd.testing.assert_frame_equal(a.equity, b.equity)  # determinista
    pd.testing.assert_frame_equal(a.trades, b.trades)
