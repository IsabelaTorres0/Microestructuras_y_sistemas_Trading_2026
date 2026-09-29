# Act 05 — Estrategia SMA + Bollinger + RSI + MACD (BTC, barras de 4 h)

- `SPEC.md` — especificación de la estrategia, p* y auditoría de sesgos
- `Act05.ipynb` — notebook con todas las tablas, gráficas, métricas y conclusión (ya ejecutado)
- `src/indicators.py` — carga, agregación a barras (4 h por defecto) e indicadores
- `src/strategy.py` — regla de entrada, `Position`, sizing con ρ, salidas (SL, TP, break-even)
- `src/backtest.py` — `backtest()` orientado a eventos (función pura)
- `src/metrics.py` — rendimiento, Sharpe, drawdown, Calmar, turnover, win rate
- `tests/test_strategy.py` — tests (a)(b)(c), golden file y truncamiento
- `operaciones.csv` — lista completa de operaciones
- `data/btc_project_train.csv` — datos

Ejecutar: abrir `Act05.ipynb` desde esta carpeta. Tests: `python -m pytest tests`.
