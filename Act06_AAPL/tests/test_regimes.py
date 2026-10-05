import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.backtest import backtest  # noqa: E402
from src.optimize import preparar  # noqa: E402
from src.regimes import (FEATURES, REGIMENES, backtest_regimes, elbow_point, fit_hmm, fit_kmeans,  # noqa: E402
                         fit_rules, fit_scaler, gate_by_regime, hmm_filter, name_states,
                         predict_hmm, predict_kmeans, predict_rules, regime_features, regime_stats)
from src.strategy import Params  # noqa: E402


def synthetic_ohlc(n=700, seed=5):
    """Precios con tres tramos distintos: tendencia tranquila, lateral y crisis."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    third = n // 3
    r = np.r_[rng.normal(0.0015, 0.008, third),                       # tendencia
              rng.normal(0.0, 0.008, third) * np.tile([1, -1], third)[:third],   # va y viene
              rng.normal(-0.001, 0.035, n - 2 * third)]              # crisis
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.normal(0, 0.003, n))
    h = np.maximum(o, c) * (1 + abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


@pytest.fixture(scope="module")
def setup():
    df = synthetic_ohlc()
    feats = regime_features(df)
    tr = feats.iloc[:500]
    sc = fit_scaler(tr)
    return df, feats, sc, sc.transform(feats)


# ---- 1. Prueba de truncamiento en CADA columna de la matriz de features
@pytest.mark.parametrize("col", FEATURES)
@pytest.mark.parametrize("t", [150, 333, 650])
def test_regime_features_truncation(setup, col, t):
    df, feats, _, _ = setup
    part = regime_features(df.iloc[:t + 1])
    assert part[col].iloc[-1] == pytest.approx(feats[col].iloc[t], rel=1e-12, abs=1e-12)


def test_features_have_expected_warmup(setup):
    _, feats, _, _ = setup
    assert feats["vol"].first_valid_index() == feats.index[20]      # 20 rendimientos
    assert feats["trend"].first_valid_index() == feats.index[99]    # SMA de 100
    assert feats["ac1"].first_valid_index() == feats.index[61]      # 60 pares (r_t, r_{t-1})


def test_scaler_uses_train_only(setup):
    _, feats, sc, z = setup
    ztr = z.iloc[:500].dropna()
    assert np.allclose(ztr.mean(), 0, atol=1e-9) and np.allclose(ztr.std(ddof=0), 1, atol=1e-9)
    assert sc.mean["vol"] == pytest.approx(feats.iloc[:500].dropna()["vol"].mean())   # solo filas completas de train


# ---- 2a. Reglas
def test_rules_are_exhaustive_and_prioritize_crisis(setup):
    _, feats, _, _ = setup
    th = fit_rules(feats.iloc[:500])
    lab = predict_rules(feats, th)
    ok = feats[FEATURES].notna().all(axis=1)
    assert lab[ok].isin(REGIMENES).all() and lab[~ok].isna().all()
    assert (lab[ok & (feats["vol"] > th["vol_alta"])] == "Crisis").all()
    f = pd.DataFrame({"vol": [th["vol_alta"] * 2, th["vol_alta"] / 2, th["vol_alta"] / 2, th["vol_alta"] / 2],
                      "trend": [9.0, th["trend_fuerte"] * 2, th["trend_fuerte"] / 2, th["trend_fuerte"] * 2],
                      "ac1": [0.5, th["ac_baja"] + 0.1, th["ac_baja"] + 0.1, th["ac_baja"] - 0.1]})
    assert predict_rules(f, th).tolist() == ["Crisis", "Tendencia", "Mean reverting", "Mean reverting"]


def test_name_states_from_centroids():
    c = pd.DataFrame({"vol": [-0.5, 2.0, -0.4], "trend": [1.5, -0.3, 0.1], "ac1": [0.2, 0.0, -0.8]})
    assert name_states(c) == {1: "Crisis", 0: "Tendencia", 2: "Mean reverting"}


def test_elbow_point_on_known_curve():
    inercia = pd.Series([1000, 400, 120, 100, 90, 82, 76, 71], index=range(1, 9))
    assert elbow_point(inercia) == 3


# ---- 2b / 2c. K-means y HMM
def test_kmeans_labels_are_causal(setup):
    _, _, _, z = setup
    fit = fit_kmeans(z.iloc[:500])
    full = predict_kmeans(z, fit)
    part = predict_kmeans(z.iloc[:401], fit)
    assert full.iloc[400] == part.iloc[-1]
    assert set(full.dropna().unique()) <= set(REGIMENES)


def test_hmm_filter_is_causal_and_matches_hmmlearn(setup):
    _, _, _, z = setup
    fit = fit_hmm(z.iloc[:500])
    X = z.dropna().to_numpy()
    full = hmm_filter(fit["model"], X)
    assert np.allclose(full.sum(axis=1), 1)
    for t in (50, 200, 420):
        part = hmm_filter(fit["model"], X[:t + 1])
        assert np.allclose(part[-1], full[t], atol=1e-10)             # no cambia al truncar
        # en la ULTIMA barra, filtrado = suavizado: debe coincidir con hmmlearn
        assert np.allclose(part[-1], fit["model"].predict_proba(X[:t + 1])[-1], atol=1e-6)
    lab = predict_hmm(z, fit)
    assert set(lab["filtrada"].dropna().unique()) <= set(REGIMENES)


def test_regime_stats_by_hand():
    lab = pd.Series(["Crisis"] * 4 + ["Tendencia"] * 6 + ["Crisis"] * 2 + ["Mean reverting"] * 9)
    s = regime_stats(lab)
    assert s["Duración media (barras)"] == pytest.approx(21 / 4)      # 4 rachas en 21 barras
    assert s["Transiciones por mes"] == pytest.approx(3.0)            # 3 cambios en 21 barras = 1 mes
    assert s["% Crisis"] == pytest.approx(100 * 6 / 21)


# ---- 7. Operar los regímenes
def test_backtest_regimes_equals_backtest_with_single_theta(setup):
    """Si los tres regímenes usan el mismo θ, el resultado debe ser idéntico al backtest normal."""
    df, feats, _, _ = setup
    data = preparar(df)
    data["etiqueta"] = predict_rules(feats, fit_rules(feats.iloc[:500]))
    d = data.iloc[100:]
    p = Params(sl_atr=2.0, tp_atr=2.5, be_atr=1.0)
    a = backtest(d, p)
    b = backtest_regimes(d, {r: p for r in REGIMENES}, default=p)
    assert len(a["trades"]) > 3
    pd.testing.assert_frame_equal(a["equity"], b["equity"])
    np.testing.assert_allclose(a["trades"]["pnl"], b["trades"]["pnl"])


def test_backtest_regimes_skips_disabled_regime_and_is_pure(setup):
    df, feats, _, _ = setup
    data = preparar(df)
    data["etiqueta"] = predict_rules(feats, fit_rules(feats.iloc[:500]))
    d = data.iloc[100:]
    before = d.copy(deep=True)
    p = Params(sl_atr=2.0, tp_atr=2.5, be_atr=1.0)
    res = backtest_regimes(d, {"Crisis": None, "Tendencia": p, "Mean reverting": p}, default=p)
    pd.testing.assert_frame_equal(d, before)
    assert "Crisis" not in set(res["trades"]["regimen"])
    only = backtest(gate_by_regime(d, "Tendencia"), p)
    assert len(only["trades"]) == int((backtest_regimes(d, {"Tendencia": p}, default=p)["trades"]["regimen"] == "Tendencia").sum())
