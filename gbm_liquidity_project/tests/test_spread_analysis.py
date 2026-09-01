import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.spread_analysis import (
    compute_high_low,
    compute_high_low_pct,
    summary_stats,
    plot_distribution,
)


@pytest.fixture
def synthetic_ohlc():
    idx = pd.date_range("2026-08-28 09:30", periods=5, freq="1min")
    return pd.DataFrame(
        {
            "Open":  [100.0, 101.0, 100.5, 102.0, 101.5],
            "High":  [101.0, 101.8, 101.0, 103.0, 102.0],
            "Low":   [99.5, 100.7, 100.0, 101.5, 101.0],
            "Close": [100.8, 100.9, 100.7, 102.5, 101.8],
            "Volume": [1000, 1200, 900, 1500, 1100],
        },
        index=idx,
    )


def test_compute_high_low_values(synthetic_ohlc):
    hl = compute_high_low(synthetic_ohlc)
    expected = synthetic_ohlc["High"] - synthetic_ohlc["Low"]
    pd.testing.assert_series_equal(hl, expected.rename("high_low"))
    assert (hl >= 0).all()


def test_compute_high_low_pct_values(synthetic_ohlc):
    hl_pct = compute_high_low_pct(synthetic_ohlc)
    expected = (synthetic_ohlc["High"] - synthetic_ohlc["Low"]) / synthetic_ohlc["Close"]
    pd.testing.assert_series_equal(hl_pct, expected.rename("high_low_pct"))


def test_summary_stats_keys_and_values():
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="x")
    stats = summary_stats(series)

    assert set(stats.keys()) == {"mean", "median", "std", "p10", "p90", "max", "n"}
    assert stats["mean"] == pytest.approx(3.0)
    assert stats["median"] == pytest.approx(3.0)
    assert stats["max"] == pytest.approx(5.0)
    assert stats["n"] == 5


def test_plot_distribution_writes_file(tmp_path, synthetic_ohlc):
    hl_a = compute_high_low(synthetic_ohlc)
    # second series with more spread to exercise a distinct distribution
    hl_b = hl_a * 3 + 0.1

    outpath = tmp_path / "hl_dist.png"
    result_path = plot_distribution(
        hl_a, hl_b,
        label_liquid="LIQ", label_illiquid="ILLIQ",
        title="test", outpath=str(outpath),
    )

    assert os.path.exists(result_path)
    assert os.path.getsize(result_path) > 0
