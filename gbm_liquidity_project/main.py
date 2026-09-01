import os

from src.config import SETTINGS
from src.data_loader import download_1min_bars, save_bars
from src.spread_analysis import (
    compute_high_low,
    compute_high_low_pct,
    summary_stats,
    plot_distribution,
)
from src.gbm_simulation import (
    realized_log_returns,
    calibrate_gbm,
    simulate_gbm_paths,
    verify_mean_median_identity,
)


def main() -> None:
    s = SETTINGS
    os.makedirs(s.outputs_dir, exist_ok=True)
    #Download
    print(f"Downloading 1-min bars: {s.ticker_liquid} (liquid)...")
    df_liquid = download_1min_bars(s.ticker_liquid, period=s.period, interval=s.interval)
    save_bars(df_liquid, s.ticker_liquid, s.raw_data_dir)

    print(f"Downloading 1-min bars: {s.ticker_illiquid} (illiquid)...")
    df_illiquid = download_1min_bars(s.ticker_illiquid, period=s.period, interval=s.interval)
    save_bars(df_illiquid, s.ticker_illiquid, s.raw_data_dir)

    print(f"  {s.ticker_liquid}: {len(df_liquid)} bars")
    print(f"  {s.ticker_illiquid}: {len(df_illiquid)} bars")

   #liquidity
    hl_liquid = compute_high_low(df_liquid)
    hl_illiquid = compute_high_low(df_illiquid)
    hl_pct_liquid = compute_high_low_pct(df_liquid)
    hl_pct_illiquid = compute_high_low_pct(df_illiquid)

    print("\n--- High-Low (absolute, price units) ---")
    print(f"{s.ticker_liquid}:   {summary_stats(hl_liquid)}")
    print(f"{s.ticker_illiquid}: {summary_stats(hl_illiquid)}")

    print("\n--- High-Low (% of close) ---")
    print(f"{s.ticker_liquid}:   {summary_stats(hl_pct_liquid)}")
    print(f"{s.ticker_illiquid}: {summary_stats(hl_pct_illiquid)}")

    plot_distribution(
        hl_liquid, hl_illiquid,
        s.ticker_liquid, s.ticker_illiquid,
        title="High-Low per bar (absolute) — tightness proxy",
        outpath=os.path.join(s.outputs_dir, "hl_distribution_abs.png"),
    )
    plot_distribution(
        hl_pct_liquid, hl_pct_illiquid,
        s.ticker_liquid, s.ticker_illiquid,
        title="High-Low per bar (% of close) — tightness proxy",
        outpath=os.path.join(s.outputs_dir, "hl_distribution_pct.png"),
    )
    print(f"\nSaved distribution plots to {s.outputs_dir}/")

    #Calibration and simulation
    calib_df = df_liquid if s.calibration_ticker == "liquid" else df_illiquid
    calib_ticker = s.ticker_liquid if s.calibration_ticker == "liquid" else s.ticker_illiquid

    log_rets = realized_log_returns(calib_df["Close"])
    bars_per_year = s.trading_minutes_per_day * s.trading_days_per_year
    params = calibrate_gbm(log_rets, bars_per_year=bars_per_year)

    print(f"\n--- GBM calibration (from {calib_ticker} realized 1-min returns) ---")
    print(f"  annualized mu:    {params.mu:.6f}")
    print(f"  annualized sigma: {params.sigma:.6f}")

    s0 = float(calib_df["Close"].iloc[-1])
    n_steps = len(log_rets)  # simulate forward over the same horizon as one session

    paths = simulate_gbm_paths(s0=s0, params=params, n_steps=n_steps, n_paths=s.n_paths, seed=42)
    identity = verify_mean_median_identity(paths, s0=s0, params=params)

    print(f"\n--- Mean/median identity check ({s.n_paths} paths, T={identity['T_years']:.5f}y) ---")
    print(f"  Mean:   theoretical={identity['theoretical_mean']:.4f}  "
          f"simulated={identity['simulated_mean']:.4f}  "
          f"rel_error={identity['mean_rel_error']:.4%}")
    print(f"  Median: theoretical={identity['theoretical_median']:.4f}  "
          f"simulated={identity['simulated_median']:.4f}  "
          f"rel_error={identity['median_rel_error']:.4%}")

    print("\nPipeline completed. ")


if __name__ == "__main__":
    main()

#La simulación viola el trading continuo sin fricciones. La ecidencia es que GEVO tiene menos barras esperadas además de que mantiene un spread más amplio, 
#lo que indica que no se puede negociar continuamente al precio de mercado sin costos. (High-Low como % del cierre) fue en promedio ~10 veces mayor que el de SPY (0.304% vs 0.029%), con un percentil 90 que supera el máximo observado en el nombre líquido.
#Se puede decir que para medirlo numericamente usamos el High-Low por barra como proxy de tightness, comparando la distribución entre el ticker líquido y el ilíquido.
#Ademas de calcular la correlación entre los retornos log. de los tickers líquido e ilíquido para evaluar la dependencia entre ellos.
