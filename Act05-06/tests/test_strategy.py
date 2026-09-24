import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest
from src.strategy import (
    Position,
    PositionAlreadyOpenError,
    PositionBook,
    StrategyParams,
    check_intrabar_exit,
    combine_signals,
    size_position,
)


# ---------------------------------------------------------------------------
# (a) SL y TP dentro de la misma barra -> stop_loss
# ---------------------------------------------------------------------------
def test_same_bar_stop_and_target_closes_as_stop_loss():
    pos = Position("long", shares=10, entry_price=100.0, stop_loss=96.0, take_profit=108.0)
    # La barra toca ambos niveles: Low 95 <= 96 y High 110 >= 108
    result = check_intrabar_exit(pos, bar_open=100.0, bar_high=110.0, bar_low=95.0)
    assert result == (96.0, "stop_loss")


def test_same_bar_tie_inside_backtest_is_stop_loss():
    p = StrategyParams(fee_rate=0, slippage_rate=0, risk_fraction=0.01)
    idx = pd.date_range("2023-01-01", periods=2, freq="5min")
    df = pd.DataFrame(
        {
            "Open": [100, 100], "High": [101, 110], "Low": [99, 95], "Close": [100, 100],
            "atr": [1.0, 1.0], "entry_signal": [True, False], "exit_signal": [False, False],
        },
        index=idx,
    )
    trades = backtest(df, p).trades
    assert len(trades) == 1
    assert trades.loc[0, "exit_reason"] == "stop_loss"
    assert trades.loc[0, "exit_price"] == 96.0


def test_gap_below_stop_exits_at_open_not_at_stop():
    pos = Position("long", shares=10, entry_price=100.0, stop_loss=96.0, take_profit=108.0)
    assert check_intrabar_exit(pos, bar_open=94.0, bar_high=95.0, bar_low=93.0) == (94.0, "stop_loss")


def test_only_target_hit_is_take_profit():
    pos = Position("long", shares=10, entry_price=100.0, stop_loss=96.0, take_profit=108.0)
    assert check_intrabar_exit(pos, 101.0, 109.0, 100.0) == (108.0, "take_profit")


# ---------------------------------------------------------------------------
# (b) El sizing regresa exactamente el riesgo presupuestado
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "equity, entry, stop, risk_fraction",
    [
        (100_000, 30_000.0, 29_910.0, 0.001),
        (100_000, 100.0, 96.0, 0.01),
        (57_321.4, 25_431.7, 25_358.2, 0.001),
    ],
)
def test_sizing_returns_exact_risk_budget(equity, entry, stop, risk_fraction):
    shares, capped = size_position(equity, entry, stop, risk_fraction, max_leverage=1.0)
    assert not capped
    assert shares * (entry - stop) == pytest.approx(equity * risk_fraction, rel=1e-12)


def test_sizing_respects_leverage_cap():
    # Stop a 1 bp: el riesgo de 0.1 % pediría 10x de nocional; el tope lo limita a 1x
    shares, capped = size_position(100_000, 30_000.0, 29_997.0, 0.001, max_leverage=1.0)
    assert capped
    assert shares * 30_000.0 == pytest.approx(100_000)


def test_sizing_rejects_stop_above_entry():
    with pytest.raises(ValueError):
        size_position(100_000, 100.0, 101.0, 0.01)


# ---------------------------------------------------------------------------
# (c) La máquina de estados nunca mantiene dos posiciones abiertas
# ---------------------------------------------------------------------------
def test_state_machine_rejects_second_open():
    pos = Position("long", 1, 100.0, 96.0, 108.0)
    book = PositionBook().open(pos)
    with pytest.raises(PositionAlreadyOpenError):
        book.open(pos)


def test_backtest_never_holds_two_positions():
    """Señal de entrada en TODAS las barras: aun así, nunca se traslapan operaciones."""
    rng = np.random.default_rng(0)
    n = 2_000
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    df = pd.DataFrame(
        {
            "Open": open_, "High": high, "Low": low, "Close": close,
            "atr": np.full(n, 0.2), "entry_signal": np.ones(n, bool),
            "exit_signal": np.zeros(n, bool),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="5min"),
    )
    res = backtest(df, StrategyParams(max_holding_bars=10))
    tr = res.trades
    assert len(tr) > 10
    # Cada entrada ocurre después (o en la misma barra que) la salida anterior
    assert (tr["entry_time"].iloc[1:].values >= tr["exit_time"].iloc[:-1].values).all()
    # Las acciones en cartera nunca exceden las de una sola posición
    allowed = set(np.round(tr["shares"], 12)) | {0.0}
    if res.open_position is not None:
        allowed.add(round(res.open_position.shares, 12))
    assert set(np.round(res.equity["shares"].unique(), 12)) <= allowed


# ---------------------------------------------------------------------------
# Entry rule
# ---------------------------------------------------------------------------
def test_combine_signals_formula():
    s_trend = pd.Series([1.0, 1.0, -1.0, np.nan])
    s_mom = pd.Series([0.5, 1.0, -1.0, 1.0])
    out = combine_signals(s_trend, s_mom)
    assert out.iloc[:3].tolist() == [0.75, 1.0, -1.0]
    assert np.isnan(out.iloc[3])
