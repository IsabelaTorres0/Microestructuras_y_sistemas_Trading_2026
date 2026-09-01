import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.gbm_simulation import (
    GBMParams,
    realized_log_returns,
    calibrate_gbm,
    simulate_gbm_paths,
    verify_mean_median_identity,
)

BARS_PER_YEAR = 390 * 252


def test_realized_log_returns_matches_manual_calc():
    close = pd.Series([100.0, 101.0, 99.5, 102.0])
    log_rets = realized_log_returns(close)

    expected = np.log(close / close.shift(1)).dropna()
    pd.testing.assert_series_equal(log_rets, expected.rename("log_return"))
    assert len(log_rets) == 3


def test_calibrate_gbm_recovers_known_params_exactly():
    """
    Deterministic check of the calibration formula: construct a synthetic
    return series whose SAMPLE mean/variance exactly match a target
    (mu, sigma), then verify calibrate_gbm inverts the formula exactly.

    (We deliberately avoid asserting recovery from *randomly drawn* returns
    here: the standard error of the drift estimate scales as
    sigma / sqrt(n * dt), so even 50,000 one-minute bars — about half a
    trading year — leave mu essentially unidentifiable from noise alone.
    That's a real, well-known property of GBM calibration, not a bug; the
    'illiquid vs. GBM' analysis paragraph below leans on this same idea.)
    """
    true_mu = 0.10
    true_sigma = 0.30
    dt = 1.0 / BARS_PER_YEAR

    per_bar_mean = (true_mu - 0.5 * true_sigma ** 2) * dt
    per_bar_std = true_sigma * np.sqrt(dt)

    # Two points with exactly the target mean and (population) std.
    synthetic_log_returns = pd.Series(
        [per_bar_mean - per_bar_std, per_bar_mean + per_bar_std]
    )
    # pandas .var() defaults to ddof=1 (sample variance); rescale the two-point
    # spread so the *sample* variance matches per_bar_std**2 exactly.
    synthetic_log_returns = pd.Series(
        [per_bar_mean - per_bar_std / np.sqrt(2), per_bar_mean + per_bar_std / np.sqrt(2)]
    )

    params = calibrate_gbm(synthetic_log_returns, bars_per_year=BARS_PER_YEAR)

    assert params.sigma == pytest.approx(true_sigma, rel=1e-9)
    assert params.mu == pytest.approx(true_mu, rel=1e-9)
    assert params.dt == pytest.approx(dt)


def test_calibrate_gbm_drift_is_noisy_on_random_samples():
    """Sanity check of the known limitation: sigma is well estimated from a
    large random sample, but mu's error is large relative to its magnitude
    over a single session's worth of 1-min bars."""
    true_mu = 0.10
    true_sigma = 0.30
    dt = 1.0 / BARS_PER_YEAR
    n = 50_000

    rng = np.random.default_rng(123)
    per_bar_drift = (true_mu - 0.5 * true_sigma ** 2) * dt
    per_bar_vol = true_sigma * np.sqrt(dt)
    synthetic_log_returns = pd.Series(
        rng.normal(loc=per_bar_drift, scale=per_bar_vol, size=n)
    )

    params = calibrate_gbm(synthetic_log_returns, bars_per_year=BARS_PER_YEAR)

    # sigma converges fast: tight tolerance is reasonable
    assert params.sigma == pytest.approx(true_sigma, rel=0.05)
    # mu does not converge at this sample size — assert it stays finite and
    # of plausible order of magnitude, not that it matches true_mu closely
    assert np.isfinite(params.mu)
    assert abs(params.mu) < 5.0


def test_calibrate_gbm_requires_min_observations():
    with pytest.raises(ValueError):
        calibrate_gbm(pd.Series([0.001]), bars_per_year=BARS_PER_YEAR)


def test_simulate_gbm_paths_shape_and_positivity():
    params = GBMParams(mu=0.08, sigma=0.25, dt=1 / BARS_PER_YEAR)
    paths = simulate_gbm_paths(s0=100.0, params=params, n_steps=50, n_paths=200, seed=1)

    assert paths.shape == (200, 51)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)  # GBM paths never cross zero


def test_verify_mean_median_identity_within_tolerance():
    """With 10,000 paths, simulated mean/median should track the closed-form
    GBM identity within a small relative error."""
    params = GBMParams(mu=0.08, sigma=0.25, dt=1 / BARS_PER_YEAR)
    s0 = 100.0
    n_steps = 390  # one trading day of 1-min steps

    paths = simulate_gbm_paths(s0=s0, params=params, n_steps=n_steps, n_paths=10_000, seed=7)
    result = verify_mean_median_identity(paths, s0=s0, params=params)

    assert result["mean_rel_error"] < 0.01
    assert result["median_rel_error"] < 0.01
    assert result["theoretical_median"] < result["theoretical_mean"]  # Jensen's gap
