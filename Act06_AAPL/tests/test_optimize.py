import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backtest import backtest  # noqa: E402
from src import metrics as M  # noqa: E402
from src.optimize import (PENALTY, calmar_objetivo, forward_chaining_folds, objetivo,  # noqa: E402
                          random_walk, theta_base, theta_to_params)
from src.strategy import Params  # noqa: E402


def make_bars(rows, atr=10.0):
    idx = pd.date_range("2024-01-01", periods=len(rows), freq="D")
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close", "entry_trigger"], index=idx)
    df["atr_14"] = atr
    return df


def test_theta_to_params_only_changes_exits():
    p = theta_to_params({"sl_atr": 2.5, "tp_atr": 5.0, "be_atr": 1.5})
    assert (p.sl_atr, p.tp_atr, p.be_atr) == (2.5, 5.0, 1.5)
    assert p.rho == Params().rho and p.commission_bps == Params().commission_bps
    assert theta_to_params(theta_base()) == Params()


def test_objective_is_calmar_of_backtest():
    """J(θ) debe ser exactamente CAGR / |MDD| del backtest cuando el MDD supera el piso."""
    rng = np.random.default_rng(3)
    close = 100 + np.cumsum(rng.normal(0, 1, 400))
    rows = [(c, c + abs(rng.normal(0, 1.5)), c - abs(rng.normal(0, 1.5)), c, i % 7 == 0)
            for i, c in enumerate(close)]
    data = make_bars(rows, atr=1.0)
    theta = {"sl_atr": 2.0, "tp_atr": 3.0, "be_atr": 1.0}
    res = backtest(data, theta_to_params(theta))
    eq = res["equity"]["equity"]
    assert abs(M.max_drawdown(eq)) > 0.01
    assert objetivo(data, theta, min_trades=1) == pytest.approx(M.calmar(eq))


def test_objective_penalizes_too_few_trades():
    rows = [(100, 101, 99, 100, False)] * 30
    assert objetivo(make_bars(rows), theta_base()) == PENALTY


def test_objective_uses_drawdown_floor():
    """Con un drawdown menor al 1 % el denominador es 1 %, no el drawdown real."""
    idx = pd.date_range("2024-01-01", periods=366, freq="D")
    eq = pd.Series(np.linspace(100_000, 110_000, 366), index=idx)
    eq.iloc[100] = eq.iloc[99] * 0.999                    # drawdown de -0.1 %
    res = {"trades": pd.DataFrame({"pnl": [1, 2, 3]}), "equity": pd.DataFrame({"equity": eq})}
    assert calmar_objetivo(res, min_trades=3) == pytest.approx(M.cagr(eq) / 0.01)


def test_forward_chaining_folds_are_ordered_and_disjoint():
    folds = forward_chaining_folds(n=1000, start=100, train_min=400, test_len=125)
    assert len(folds) == 4                                  # 500-625, 625-750, 750-875, 875-1000
    for f in folds:
        assert f["train"][0] == 100                         # ventana creciente (anclada)
        assert f["train"][1] == f["test"][0]                # el test empieza justo después del train
    tests = [f["test"] for f in folds]
    assert all(a[1] == b[0] for a, b in zip(tests, tests[1:]))  # tests contiguos, sin traslape
    assert tests[-1][1] <= 1000


def test_random_walk_keeps_bar_shape_and_breaks_order():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=500)
    c = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, 500)))
    o = c * np.exp(rng.normal(0, 0.005, 500))
    df = pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.01, "low": np.minimum(o, c) * 0.99,
                       "close": c}, index=idx)
    rw = random_walk(df, seed=1)
    assert rw.index.equals(df.index)
    assert (rw["high"] >= rw[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (rw["low"] <= rw[["open", "close"]].min(axis=1) + 1e-9).all()
    r_real = np.log(df["close"]).diff().dropna()
    r_rw = np.log(rw["close"]).diff().dropna()
    assert r_rw.std() == pytest.approx(r_real.std(), rel=0.25)       # misma volatilidad aprox.
    assert not np.allclose(rw["close"].to_numpy(), df["close"].to_numpy())
    pd.testing.assert_frame_equal(random_walk(df, seed=1), rw)       # reproducible con la semilla
