# Act 06 — Backtest orientado a eventos (BTC, barras de 4 h)

- `Act06.ipynb` — notebook ejecutado: motor orientado a eventos, golden-file, truncamiento del pipeline, curva de equity y drawdown, lista de operaciones, sensibilidad a costos (0 a 50 bps), métricas, win rate teórico contra empírico, auditoría de sesgos y conclusión
- `SPEC.md` — especificación actualizada (SL = 4 ATR, p* = 57.1 %)
- `src/indicators.py` — carga, agregación a barras e indicadores
- `src/strategy.py` — regla de entrada, `Position`, sizing, salidas
- `src/backtest.py` — `backtest()` puro orientado a eventos, `run_pipeline()`, `with_costs()` (Act 05) y `scale_costs()` (Act 06)
- `src/metrics.py` — rendimiento, Sharpe, drawdown, Calmar, turnover, win rate, **p\* con payoff empírico, expectancy en R, `winrate_table()`**
- `tests/test_strategy.py` — 18 pruebas: unitarias, 4 golden-file, pureza, costos, drawdown, p* y truncamiento
- `operaciones.csv` — lista completa de operaciones
- `data/btc_project_train.csv` — datos

Ejecutar: abrir `Act06.ipynb` desde esta carpeta. Tests: `python -m pytest tests`.
