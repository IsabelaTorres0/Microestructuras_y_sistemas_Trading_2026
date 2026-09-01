import os
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import download_1min_bars, save_bars, load_bars_csv


def _fake_yf_dataframe():
    idx = pd.date_range("2026-08-28 09:30", periods=3, freq="1min")
    return pd.DataFrame(
        {
            "Open": [10.0, 10.1, 10.05],
            "High": [10.2, 10.3, 10.15],
            "Low": [9.9, 10.0, 9.95],
            "Close": [10.1, 10.05, 10.1],
            "Adj Close": [10.1, 10.05, 10.1],
            "Volume": [500, 600, 550],
        },
        index=idx,
    )


@patch("src.data_loader.yf.download")
def test_download_1min_bars_success(mock_download):
    mock_download.return_value = _fake_yf_dataframe()

    df = download_1min_bars("FAKE", period="1d", interval="1m")

    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 3
    mock_download.assert_called_once()


@patch("src.data_loader.yf.download")
def test_download_1min_bars_empty_raises(mock_download):
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError, match="No data returned"):
        download_1min_bars("BADTICKER")


@patch("src.data_loader.yf.download")
def test_save_and_reload_bars_roundtrip(mock_download, tmp_path):
    mock_download.return_value = _fake_yf_dataframe()
    df = download_1min_bars("FAKE")

    path = save_bars(df, "FAKE", raw_data_dir=str(tmp_path))
    assert os.path.exists(path)

    reloaded = load_bars_csv(path)
    pd.testing.assert_frame_equal(
        reloaded.reset_index(drop=True), df.reset_index(drop=True)
    )
