import matplotlib.pyplot as plt
import pandas as pd

def _slice(df: pd.DataFrame, start=None, end=None):
    if start is None and end is None:
        return df
    return df.loc[start:end]


def plot_price_overlays(
    df: pd.DataFrame,
    overlay_cols=("sma_10", "sma_40", "ema_20"),
    bollinger=True,
    start=None,
    end=None,
    ax=None,
    title="Precio con overlays",
):
    """
    Grafica el precio de cierre junto con medias móviles / bandas de
    Bollinger encima. `df` debe ser el resultado de add_indicators (o el
    df original con al menos la columna 'close' más las columnas pedidas
    en overlay_cols).
    """
    data = _slice(df, start, end)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(data.index, data["close"], label="close", color="black", linewidth=1)

    for col in overlay_cols:
        if col in data.columns:
            ax.plot(data.index, data[col], label=col, linewidth=1)

    if bollinger and {"bb_high", "bb_low"}.issubset(data.columns):
        ax.plot(data.index, data["bb_high"], color="gray", linestyle="--", linewidth=0.8, label="bb_high")
        ax.plot(data.index, data["bb_low"], color="gray", linestyle="--", linewidth=0.8, label="bb_low")
        ax.fill_between(data.index, data["bb_low"], data["bb_high"], color="gray", alpha=0.08)

    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_rsi(df: pd.DataFrame, start=None, end=None, ax=None, title="RSI (14)"):
    """
    Grafica rsi_14 con las líneas de referencia de sobrecompra (70) y
    sobreventa (30).
    """
    data = _slice(df, start, end)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(14, 3))

    ax.plot(data.index, data["rsi_14"], color="purple", linewidth=1)
    ax.axhline(70, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(30, color="green", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(alpha=0.3)

    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_macd(df: pd.DataFrame, start=None, end=None, ax=None, title="MACD"):
    """
    Grafica la línea MACD, su señal, y el histograma (macd_diff).
    """
    data = _slice(df, start, end)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(14, 3))

    ax.plot(data.index, data["macd"], label="macd", color="blue", linewidth=1)
    ax.plot(data.index, data["macd_signal"], label="signal", color="orange", linewidth=1)

    colors = ["green" if v >= 0 else "red" for v in data["macd_diff"]]
    ax.bar(data.index, data["macd_diff"], color=colors, alpha=0.4, width=1.0, label="hist")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_stochastic(df: pd.DataFrame, start=None, end=None, ax=None, title="Estocástico"):
    """
    Grafica %K y %D del oscilador estocástico.
    """
    data = _slice(df, start, end)
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(14, 3))

    ax.plot(data.index, data["stoch_k"], label="%K", linewidth=1)
    ax.plot(data.index, data["stoch_d"], label="%D", linewidth=1)
    ax.axhline(80, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(20, color="green", linestyle="--", linewidth=0.8)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    if own_fig:
        plt.tight_layout()
        plt.show()
    return ax


def plot_full_panel(
    df: pd.DataFrame,
    overlay_cols=("sma_10", "sma_100", "ema_20"),
    start=None,
    end=None,
    figsize=(14, 12),
):

    fig, axes = plt.subplots(
        4, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1, 1]},
    )

    plot_price_overlays(df, overlay_cols=overlay_cols, start=start, end=end, ax=axes[0])
    plot_rsi(df, start=start, end=end, ax=axes[1])
    plot_macd(df, start=start, end=end, ax=axes[2])
    plot_stochastic(df, start=start, end=end, ax=axes[3])

    plt.tight_layout()
    plt.show()
    return fig, axes