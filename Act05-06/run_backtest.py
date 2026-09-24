"""Corre la estrategia de la Act05 sobre train y validation y genera los entregables.

Uso:  python run_backtest.py [--include-test]
El conjunto de test solo se corre con --include-test (una sola vez, al final).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.backtest import backtest
from src.data import load_bars, split
from src.metrics import drawdown, summary
from src.strategy import StrategyParams, compute_signals

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "btc_project_train.csv"
OUT = ROOT / "results"
COST_GRID_BPS = list(range(0, 55, 5))


def params_for_round_trip(base: StrategyParams, rt_bps: float) -> StrategyParams:
    """Escala comisión y slippage manteniendo su proporción del SPEC (10:1)."""
    per_side = rt_bps / 2 / 1e4
    share_fee = base.fee_rate / (base.fee_rate + base.slippage_rate)
    return replace(base, fee_rate=per_side * share_fee, slippage_rate=per_side * (1 - share_fee))


def plot_equity_and_drawdown(eq: pd.Series, name: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, height_ratios=[2, 1])
    axes[0].plot(eq.index, eq.values, lw=1, color="#1f5fa8")
    axes[0].axhline(eq.iloc[0], color="grey", lw=0.8, ls="--")
    axes[0].set_ylabel("Equity (USD)")
    axes[0].set_title(f"Curva de equity — {name}")
    dd = drawdown(eq) * 100
    axes[1].fill_between(dd.index, dd.values, 0, color="#c0392b", alpha=0.6)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_title(f"Curva de drawdown — {name}")
    fig.tight_layout()
    fig.savefig(OUT / f"equity_drawdown_{name}.png", dpi=130)
    plt.close(fig)


def cost_sensitivity(sig: pd.DataFrame, base: StrategyParams, median_atr_bps: float) -> pd.DataFrame:
    rows = []
    for rt in COST_GRID_BPS:
        res = backtest(sig, params_for_round_trip(base, rt))
        s = summary(res, median_atr_bps)
        rows.append({"round_trip_bps": rt, "sharpe": s["sharpe"], "final_equity": s["final_equity"],
                     "p_star": s["p_star_theoretical"], "win_rate": s["win_rate_empirical"]})
    return pd.DataFrame(rows)


def plot_cost_sensitivity(tables: dict[str, pd.DataFrame], spec_rt: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, t in tables.items():
        axes[0].plot(t.round_trip_bps, t.sharpe, marker="o", label=name)
        axes[1].plot(t.round_trip_bps, t.final_equity, marker="o", label=name)
    for ax, ylab in zip(axes, ["Sharpe anualizado", "Equity final (USD)"]):
        ax.axvline(spec_rt, color="grey", ls="--", lw=1, label=f"SPEC ({spec_rt:.0f} bps)")
        ax.set_xlabel("Costo ida y vuelta (bps)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].axhline(0, color="black", lw=0.8)
    axes[1].axhline(100_000, color="black", lw=0.8)
    fig.suptitle("Sensibilidad a costos")
    fig.tight_layout()
    fig.savefig(OUT / "cost_sensitivity.png", dpi=130)
    plt.close(fig)


def main(include_test: bool) -> None:
    OUT.mkdir(exist_ok=True)
    params = StrategyParams()
    bars = load_bars(str(DATA))
    # Indicadores sobre toda la serie (son causales: verificado por el test de truncamiento),
    # así validation no pierde barras de warm-up.
    signals = compute_signals(bars, params)

    train_sig = split(signals, "train")
    median_atr_bps = float((train_sig["atr"] / train_sig["Close"] * 1e4).median())

    sets = ["train", "validation"] + (["test"] if include_test else [])
    all_metrics, sens = {}, {}
    for name in sets:
        sig = split(signals, name)
        res = backtest(sig, params)
        all_metrics[name] = summary(res, median_atr_bps)
        res.trades.to_csv(OUT / f"trades_{name}.csv", index=False)
        res.equity.to_csv(OUT / f"equity_{name}.csv")
        plot_equity_and_drawdown(res.equity["equity"], name)
        sens[name] = cost_sensitivity(sig, params, median_atr_bps)
        sens[name].to_csv(OUT / f"cost_sensitivity_{name}.csv", index=False)

    plot_cost_sensitivity(sens, params.round_trip_bps)
    all_metrics["_meta"] = {"median_atr_bps_train": median_atr_bps,
                            "round_trip_bps": params.round_trip_bps}
    (OUT / "metrics.json").write_text(json.dumps(all_metrics, indent=2, default=str))

    table = pd.DataFrame({k: v for k, v in all_metrics.items() if not k.startswith("_")})
    print(table.to_string())
    for name, t in sens.items():
        print(f"\nSensibilidad a costos — {name}\n{t.to_string(index=False)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true")
    main(ap.parse_args().include_test)
