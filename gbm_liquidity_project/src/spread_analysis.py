"""
High-minus-low per bar as a proxy for intrabar "tightness" (a stand-in for
bid-ask spread when only OHLCV bars are available, not quotes/L1 data).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_high_low(df: pd.DataFrame) -> pd.Series:
    """Absolute high-minus-low per bar."""
    return (df["High"] - df["Low"]).rename("high_low")


def compute_high_low_pct(df: pd.DataFrame) -> pd.Series:
    """High-minus-low as a fraction of the bar's close (comparable across
    tickers with very different price levels)."""
    hl = df["High"] - df["Low"]
    return (hl / df["Close"]).rename("high_low_pct")


def summary_stats(series: pd.Series) -> Dict[str, float]:
    """Basic distribution summary: mean, median, std, and key quantiles."""
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "p10": float(series.quantile(0.10)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
        "n": int(series.shape[0]),
    }


def plot_distribution(
    series_liquid: pd.Series,
    series_illiquid: pd.Series,
    label_liquid: str,
    label_illiquid: str,
    title: str,
    outpath: str,
) -> str:
    """Overlaid histograms comparing the two distributions. Returns outpath."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bins = np.linspace(
        0,
        max(series_liquid.quantile(0.99), series_illiquid.quantile(0.99)),
        60,
    )

    ax.hist(
        series_liquid.clip(upper=bins[-1]),
        bins=bins,
        alpha=0.6,
        label=f"{label_liquid} (liquid)",
        density=True,
    )
    ax.hist(
        series_illiquid.clip(upper=bins[-1]),
        bins=bins,
        alpha=0.6,
        label=f"{label_illiquid} (illiquid)",
        density=True,
    )

    ax.set_title(title)
    ax.set_xlabel(series_liquid.name)
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath
