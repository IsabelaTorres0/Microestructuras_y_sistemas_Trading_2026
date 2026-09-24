# Resultados y auditoría de sesgos — Act06

Motor orientado a eventos (`src/backtest.py`), parámetros exactamente como en `SPEC.md`.
Se corrió **train** y **validation**. El **test no se ha corrido**; queda reservado para una sola ejecución final (`python run_backtest.py --include-test`).

## 1. Resultados (costo de 22 bps ida y vuelta)

| Métrica | Train (jun 2022 – mar 2023) | Validation (abr – ago 2023) |
|---|---|---|
| Equity final (desde 100,000) | 21,621 | 46,434 |
| Retorno total | −78.4 % | −53.6 % |
| Sharpe anualizado | −26.7 | −29.0 |
| Drawdown máximo | −78.4 % | −53.6 % |
| Calmar | −1.07 | −1.57 |
| Operaciones (por día) | 1,907 (6.3) | 854 (5.6) |
| Turnover anual | 1,696× | 1,571× |
| Exposición (fracción de barras en posición) | 42 % | 41 % |
| **Win rate empírico** | **29.7 %** | **23.8 %** |
| **p\* teórico (SPEC)** | **58.6 %** | **58.6 %** |
| p\* promedio con el SL real de cada trade | 64.0 % | 64.1 % |
| Profit factor | 0.32 | 0.23 |
| R-múltiplo promedio (neto) | −0.84 | −0.94 |
| Comisiones + slippage pagados | 77,161 | 47,728 |
| Salidas: SL / TP / señal / holding máx. | 804 / 508 / 457 / 138 | 323 / 173 / 279 / 79 |

### Sensibilidad a costos (ver `results/cost_sensitivity.png`)

| Costo ida y vuelta | 0 bps | 5 | 10 | 20 | 22 (SPEC) | 30 | 50 |
|---|---|---|---|---|---|---|---|
| Sharpe train | **+2.78** | −4.74 | −12.1 | −24.6 | −26.7 | −33.9 | −44.6 |
| Sharpe validation | −0.89 | −8.35 | −15.4 | −27.1 | −29.0 | −35.7 | −45.3 |

### Diagnóstico

1. **La estrategia no tiene ventaja bruta suficiente.** Sin costos, el win rate en train (37.7 %) supera apenas el break-even sin costos (33.3 %) y el R-múltiplo bruto promedio es ≈ 0.00. En validation, incluso a 0 bps, el Sharpe es negativo, así que la pequeña ventaja de train no se generaliza.
2. **Los costos consumen 0.84 R por operación.** El SL real promedio fue de 37.7 bps (mediana 29.8), y 22 bps de costos sobre eso equivalen a pagar casi una pérdida completa en cada trade. Por eso el p\* por trade (64 %) es incluso mayor que el del SPEC (58.6 %).
3. **Break-even de costos:** la curva de train cruza Sharpe = 0 entre 0 y 5 bps ida y vuelta. Ningún exchange minorista cobra eso con órdenes taker.
4. **Conclusión:** el problema no es de implementación sino de diseño. Operar ~6 veces al día en barras de 5 min con un stop de ~30 bps es incompatible con costos de 22 bps. Esto confirma cuantitativamente la advertencia del SPEC §8.

## 2. Tabla de auditoría de sesgos

| Sesgo | Riesgo en esta configuración | Evidencia concreta | Estado |
|---|---|---|---|
| **Look-ahead** | Bajo | (1) `tests/test_truncation.py` recalcula todo el pipeline (indicadores → señales → combinación → backtest) sobre `df.iloc[:t+1]` en 30+ puntos, incluidos los bordes de huecos, y exige igualdad exacta de indicadores, score, cash, shares, equity y lista de trades. (2) **Prueba de mutación:** al inyectar `close.shift(-1)` en la SMA, ambos tests de truncamiento fallan, así que el test sí detecta look-ahead. (3) Las señales al cierre de t se ejecutan en el Open de t+1. (4) El ATR del SL/TP es el de la barra de señal. (5) El equity se marca a mercado y no se fuerza un cierre al final, lo que alteraría el valor en t. | Controlado |
| **Survivorship** | Medio | Un solo activo, sin universo que filtrar. Pero elegir BTC es en sí una selección *ex post*: es la cripto que sobrevivió y dominó, cuando en 2022 colapsaron LUNA, FTT y otras. Los resultados no se pueden generalizar a "cripto". Además el dataset viene preprocesado por el curso: no se sabe qué exchange ni qué regla de agregación se usó (el Open difiere del Close previo en una mediana de 1.4 bps). | Reconocido, no corregible con un solo activo |
| **Overfitting / data snooping** | Bajo a medio | (1) Una sola configuración probada: SMA10/SMA40/RSI14 son ventanas elegidas por el usuario antes de ver resultados, y θ, los múltiplos de ATR, el riesgo y los costos quedaron fijos en `SPEC.md` antes del primer backtest. (2) Número de variantes evaluadas = 1, sin optimización. (3) Del train solo se usaron estadísticas descriptivas (ATR mediano de 7.27 bps y frecuencia de cruces) para calibrar p\* y justificar θ, no métricas de desempeño. (4) **Riesgo residual:** el ancho de stop se eligió al ver una tabla de p\* calculada con el ATR del periodo completo, y se imprimieron estadísticas descriptivas del precio en el periodo de test (inicio, fin, rango). No se corrió la estrategia en test, pero el analista ya sabe que fue un periodo alcista (+62.6 %). Cualquier cambio posterior debe registrarse en el changelog del SPEC. | Mayormente controlado; contaminación descriptiva declarada |
| **Ejecución optimista** | Bajo a medio | Supuestos **pesimistas** implementados y probados: empate SL/TP → stop (test a); gap bajo el SL → sale al Open, no al SL (237 de 804 stops en train, con 2.8 bps medianos de pérdida extra); gap sobre el TP → sale al TP sin mejora; comisión taker de 10 bps + 1 bp de slippage por lado; entradas canceladas si la siguiente barra no tiene precio. **Supuestos aún optimistas:** 1 bp de slippage en eventos de estrés (colapso de FTX, nov 2022) probablemente es bajo; se asume ejecución exacta en el Open sin latencia; se asume liquidez infinita al precio del stop. Con resultados ya negativos, estos sesgos solo harían el resultado real peor. | Mayormente controlado |
| **Selección / periodo de muestra** | Alto | Train (jun 2022 – mar 2023) cubre un mercado bajista y de crisis (BTC: 31,835 → mínimo 15,612 con FTX → 28,480; −10.5 %). Validation (abr – ago 2023) fue lateral bajista (−8.9 %). Una estrategia **solo largo** de seguimiento de tendencia fue evaluada en régimen desfavorable, y el test (sep – dic 2023) es alcista, así que un resultado positivo en test **no** probaría ventaja, sino exposición al rally. Faltan barras: 2,495 en train y 1,314 en validation, sin precio y sin operaciones. Diecinueve meses son un solo ciclo de mercado. | Riesgo alto, declarado |

## 3. Reproducibilidad

```bash
pip install pandas numpy matplotlib pytest
python -m pytest            # 18 tests
python run_backtest.py      # train + validation -> results/
```
