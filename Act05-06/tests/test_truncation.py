"""Prueba de truncamiento de TODO el pipeline:
indicadores -> señales -> combinación -> backtest.

Para cada t se recalcula todo sobre df.iloc[:t+1] y se exige que los valores
en t sean idénticos a los del cálculo con la muestra completa. Si algún paso
usara información futura, el valor en t cambiaría al quitar el futuro.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest
from src.data import load_bars
from src.strategy import StrategyParams, compute_signals

DATA = Path(__file__).resolve().parents[1] / "data" / "btc_project_train.csv"
COLS_SIGNAL = ["sma_fast", "sma_slow", "rsi", "atr", "s_trend", "s_mom", "score"]
COLS_BOOL = ["entry_signal", "exit_signal"]
COLS_STATE = ["cash", "shares", "equity"]


def check_truncation(bars: pd.DataFrame, ts: list[int], params: StrategyParams):
    full_sig = compute_signals(bars, params)
    full_res = backtest(full_sig, params)
    for t in ts:
        cut = bars.iloc[: t + 1]
        sig_t = compute_signals(cut, params)
        res_t = backtest(sig_t, params)

        a, b = full_sig.iloc[t], sig_t.iloc[t]
        for col in COLS_SIGNAL:
            assert (np.isnan(a[col]) and np.isnan(b[col])) or a[col] == b[col], (t, col)
        for col in COLS_BOOL:
            assert a[col] == b[col], (t, col)
        for col in COLS_STATE:
            assert full_res.equity[col].iloc[t] == res_t.equity[col].iloc[t], (t, col)

        # Las operaciones cerradas hasta t deben ser las mismas
        closed = full_res.trades[full_res.trades["exit_time"] <= bars.index[t]]
        assert len(closed) == len(res_t.trades), t
        if len(closed):
            pd.testing.assert_frame_equal(closed.reset_index(drop=True), res_t.trades)


@pytest.fixture(scope="module")
def real_bars():
    if not DATA.exists():
        pytest.skip("No está el CSV en data/")
    bars = load_bars(str(DATA))
    # Ventana con huecos de datos para ejercitar también la regla de NaN
    na = bars["Close"].isna().to_numpy()
    first_gap = int(np.argmax(na))
    start = max(0, first_gap - 1_500)
    return bars.iloc[start : start + 4_000]


def test_truncation_full_pipeline_real_data(real_bars):
    rng = np.random.default_rng(42)
    ts = sorted(set(rng.integers(45, len(real_bars) - 1, 25).tolist()))
    # Agregar barras justo antes y después de huecos
    na = np.flatnonzero(real_bars["Close"].isna().to_numpy())
    if len(na):
        ts += [int(na[0]) - 1, int(na[0]), int(na[-1]) + 1, int(na[-1]) + 50]
    ts += [len(real_bars) - 1]
    check_truncation(real_bars, sorted(set(ts)), StrategyParams())


def test_truncation_full_pipeline_synthetic_with_trades():
    """Serie sintética con tendencia y ruido para asegurar que haya operaciones."""
    rng = np.random.default_rng(7)
    n = 3_000
    close = 30_000 * np.exp(np.cumsum(rng.normal(0.00005, 0.002, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.001, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.001, n))
    bars = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close},
        index=pd.date_range("2023-01-01", periods=n, freq="5min"),
    )
    bars.iloc[1_000:1_010] = np.nan  # un hueco
    params = StrategyParams()
    assert len(backtest(compute_signals(bars, params), params).trades) > 20
    ts = list(range(50, n, 97)) + [999, 1_000, 1_010, 1_011, n - 1]
    check_truncation(bars, sorted(set(ts)), params)
