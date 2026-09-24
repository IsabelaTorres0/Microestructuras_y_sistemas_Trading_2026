"""Carga y limpieza del dataset BTC 5 min (SPEC §1)."""
from __future__ import annotations

import pandas as pd

BAR = "5min"

SPLITS = {
    "train": ("2022-06-01 00:00", "2023-03-31 23:55"),
    "validation": ("2023-04-01 00:00", "2023-08-31 23:55"),
    "test": ("2023-09-01 00:00", "2023-12-31 00:00"),
}


def load_bars(path: str) -> pd.DataFrame:
    """Lee el CSV y lo re-muestrea a una malla fija de 5 min en UTC.

    - Timestamps fuera de la malla se asignan a su barra con floor('5min').
    - Open = primero, High = máx., Low = mín., Close = último.
    - Las barras sin precio quedan como NaN (no se rellenan).
    - Volume no se usa (columna poco confiable, ver SPEC §1).
    """
    raw = pd.read_csv(path)
    raw = raw.dropna(subset=["Open", "High", "Low", "Close"])
    raw["dt"] = pd.to_datetime(raw["Timestamp"], unit="s").dt.floor(BAR)
    raw = raw.sort_values("Timestamp")

    bars = raw.groupby("dt").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    )
    grid = pd.date_range(bars.index.min(), bars.index.max(), freq=BAR)
    bars = bars.reindex(grid)
    bars.index.name = "datetime"
    return bars


def split(bars: pd.DataFrame, name: str) -> pd.DataFrame:
    start, end = SPLITS[name]
    return bars.loc[start:end]
