# SPEC — Estrategia "Tendencia + Momentum con Bollinger" sobre BTC, barras de 4 horas (Act 05 / Act 06)

**Autora:** Isabela Torres Septien Uribe

> **Versión Act 06.** Se actualizó para que coincida con el código (`src/strategy.py`): stop-loss de **4 ATR** (antes 1.5), por lo que R, p\* y el costo en R cambian. Se corrigió la ventana de la SMA lenta en la auditoría (100, no 40) y se reclasificó el riesgo de overfitting.

## 1. Universo y frecuencia
| Campo | Valor |
|---|---|
| Activo | BTC-USD (spot) |
| Fuente | `btc_project_train.csv` (barras de 5 min, 166,450 filas; se eliminan las 3,745 filas con NaN) |
| Frecuencia de trabajo | **Barras de 4 horas**: open = primer open, high = máximo, low = mínimo, close = último close del bloque (00, 04, 08, 12, 16, 20 h) |
| Rango | 2022-06-01 00:00 → 2023-12-31 00:00 (3,450 barras de 4 h) |
| Train (60 %) | 2022-06-01 00:00 → 2023-05-15 00:00 (2,070 barras) |
| Test (20 %) | 2023-05-15 04:00 → 2023-09-07 00:00 (690 barras) |
| Validation (20 %) | 2023-09-07 04:00 → 2023-12-31 00:00 (690 barras) |

La división es cronológica (sin barajar). Cada tramo arranca con $100,000.

**Historial de cambios (decididos después de ver resultados):**
1. Frecuencia de 1 h → 4 h (con 1 h el costo era ≈ 0.43 R por operación).
2. Stop-loss de 1.5 ATR → 4 ATR.
3. Filtro de tendencia con SMA_100 y disparador por cruce alcista del MACD.

## 2. Features
| Indicador | Ventana | Familia |
|---|---|---|
| SMA_10, SMA_100 | 10 y 100 barras | Tendencia |
| Bandas de Bollinger (BB_mid, BB_high, BB_low) | 20 barras, 2 σ | Volatilidad |
| RSI_14 (Wilder) | 14 barras | Momentum (oscilador) |
| MACD (12, 26) y señal (9) | 12 / 26 / 9 | Momentum |
| ATR_14 (Wilder) | 14 barras | Volatilidad (solo para SL/TP) |

## 3. Entry rule (solo largos), 2 condiciones
Evaluada al **cierre** de la barra *t*:

```
C1 (tendencia) = SMA_10(t) > SMA_100(t)  AND  close(t) > BB_mid(t)
C2 (momentum)  = MACD(t) > Señal(t)  AND  50 < RSI_14(t) < 70  AND  close(t) < BB_high(t)

entry_signal(t)  = C1(t) AND C2(t)
entry_trigger(t) = entry_signal(t) AND (MACD(t-1) <= Señal(t-1))      # cruce alcista del MACD
```

Se abre un largo si `entry_trigger(t) = True` y no hay posición abierta.

## 4. Exit rule, 3 parámetros
| # | Parámetro | Regla |
|---|---|---|
| 1 | Stop-loss | `SL = P_entrada − 4.0 × ATR_14(t)` |
| 2 | Take-profit | `TP = P_entrada + 3.0 × ATR_14(t)` |
| 3 | Stop a break-even | Si el máximo de una barra alcanza `P_entrada + 1.0 × ATR_14(t)`, el stop sube a `P_entrada` a partir de la **siguiente** barra |

* **Señal opuesta:** no se usa.
* **Periodo máximo de holding:** ninguno (al final de la muestra se marca a mercado).

## 5. Sizing: riesgo fijo ρ
```
unidades = ρ × Equity(t) / (P_entrada − SL)       con ρ = 2 %
```
Restricción sin apalancamiento: `unidades ≤ cash / (P_entrada × (1 + comisión))`. Capital inicial **$100,000**; riesgo de **$2,000** por operación con el equity inicial.

## 6. Costs: posición larga
| Concepto | Valor | Justificación |
|---|---|---|
| Comisión | 10 bps por lado | Tarifa spot taker estándar |
| Slippage | 5 bps por lado | Spread e impacto típicos de BTC a mercado |
| Borrow fee | No aplica | Solo largos |
| **Total ida y vuelta** | **30 bps** | |

## 7. Conventions
* Señal al **cierre** de *t*; ejecución al **open de t+1** × (1 + slippage).
* SL, TP y BE usan el ATR de la barra de la señal (*t*).
* **Gap:** si una barra abre por debajo del stop (o por encima del TP) se sale al open.
* **Empate intrabar:** SL (o BE) y TP en la misma barra → se asume el stop.
* El break-even se activa para la barra siguiente.
* Una sola posición a la vez; señales con posición abierta se ignoran.

## 8. Win rate de break-even (umbral a superar)
Con R = TP / SL = 3.0 / 4.0 = **0.75**:

```
p* = 1 / (1 + R) = 1 / 1.75 = 57.1 %          (sin costos)
```

Con costos: el riesgo típico por operación es 4 × ATR ≈ 4.28 % del precio, así que 30 bps equivalen a c ≈ 0.07 R:

```
p*_costos = (1 + c) / (1 + R) ≈ 1.07 / 1.75 ≈ 61.2 %
```

**Limitación de esta fórmula:** supone que toda pérdida es −1 R. Con el break-even la mayoría de las pérdidas son ≈ −0.05 R, así que en la Act 06 también se reporta el umbral con el payoff real, `p*_payoff = |pérdida media| / (ganancia media + |pérdida media|)`.

## 9. Criterio de viabilidad (Calmar)
`Calmar = CAGR / |Drawdown máximo|`: **> 1 viable**, 0.5 a 1 marginal, < 0.5 no viable.

## 10. Auditoría de sesgos
La tabla completa, con la evidencia numérica calculada, está en la sección 12 de `Act06.ipynb`.

| Sesgo | Estado |
|---|---|
| Look-ahead | Controlado |
| Survivorship | Bajo |
| Overfitting / data snooping | **Alto** |
| Ejecución optimista | Controlado parcialmente |
| Selección / periodo de muestra | Relevante |
