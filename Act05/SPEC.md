# SPEC — Estrategia "Tendencia + Momentum con Bollinger" sobre BTC, barras de 4 horas (Act 05)

**Autora:** Isabela Torres Septien Uribe

## 1. Universo y frecuencia
| Campo | Valor |
|---|---|
| Activo | BTC-USD (spot) |
| Fuente | `btc_project_train.csv` (barras de 5 min, 166,450 filas; se eliminan las filas con NaN) |
| Frecuencia de trabajo | **Barras de 4 horas**: open = primer open, high = máximo, low = mínimo, close = último close del bloque de 4 h (00, 04, 08, 12, 16, 20 h) |
| Rango | 2022-06-01 00:00 → 2023-12-31 00:00 (3,450 barras de 4 h) |
| Train (60 %) | 2022-06-01 00:00 → 2023-05-15 00:00 (2,070 barras) |
| Test (20 %) | 2023-05-15 04:00 → 2023-09-07 00:00 (690 barras) |
| Validation (20 %) | 2023-09-07 04:00 → 2023-12-31 00:00 (690 barras) |

La división es cronológica (sin barajar). Cada tramo arranca con $100,000.

> **Cambio respecto a la versión anterior:** la primera versión usaba barras de 1 hora. Ahí el ATR era de ≈ 0.47 % del precio y el stop quedaba tan cerca que los costos (30 bps) equivalían a 0.43 R por operación. Se cambió **solo la frecuencia** a 4 horas (ATR ≈ 1.07 %); el resto de los parámetros no se tocó.

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
entry_trigger(t) = entry_signal(t) AND (MACD(t) > Señal(t)) AND (MACD(t-1) <= Señal(t-1))
```

Se abre un largo si `entry_trigger(t) = True` y no hay posición abierta. El disparador exige un cruce alcista del MACD sobre su señal, evitando entrar solo porque la combinación está “encendida” desde hace varias barras mientras el impulso ya se está agotando. Las bandas de Bollinger cumplen dos papeles: el precio sobre la banda media confirma la tendencia y el precio por debajo de la banda superior evita comprar cuando el precio está sobre-extendido.

## 4. Exit rule, 3 parámetros
| # | Parámetro | Regla |
|---|---|---|
| 1 | Stop-loss | `SL = P_entrada − 1.5 × ATR_14(t)` |
| 2 | Take-profit | `TP = P_entrada + 3.0 × ATR_14(t)` |
| 3 | Stop a break-even | Si el máximo de una barra alcanza `P_entrada + 1.0 × ATR_14(t)`, el stop sube a `P_entrada` a partir de la **siguiente** barra |

* **Señal opuesta:** no se usa; la posición no se cierra por un cruce contrario.
* **Periodo máximo de holding:** **ninguno.** Si el precio no tiene la volatilidad suficiente para tocar el SL/BE ni el TP, la posición se mantiene abierta indefinidamente (al final de la muestra se marca a mercado).

## 5. Sizing: riesgo fijo ρ
```
unidades = ρ × Equity(t) / (P_entrada − SL)       con ρ = 2 %
```
Si se toca el stop, la pérdida es exactamente ρ × Equity (antes de costos). Restricción: sin apalancamiento, `unidades ≤ cash / (P_entrada × (1 + comisión))`. Cuando el ATR es pequeño este tope se activa y el riesgo real queda por debajo del 2 %.

Capital inicial: **$100,000**. Presupuesto de riesgo: **$2,000 por operación** con el equity inicial.

## 6. Costs: posición larga
| Concepto | Valor | Justificación |
|---|---|---|
| Comisión | 10 bps por lado (0.10 %) | Tarifa spot taker estándar de exchanges grandes (p. ej. nivel base de Binance) |
| Slippage | 5 bps por lado | Spread y impacto típicos de BTC en órdenes a mercado de este tamaño |
| Borrow fee | No aplica | Solo largos, sin apalancamiento ni préstamo |
| **Total ida y vuelta** | **30 bps** | |

## 7. Conventions
* La señal se calcula con datos hasta el **cierre** de *t* y se ejecuta al **open de t+1** (precio × (1 + slippage)).
* SL, TP y BE usan el ATR de la barra de la señal (*t*), nunca el de la barra de ejecución.
* **Gap:** si una barra abre por debajo del stop, se sale al open.
* **Empate intrabar:** si SL (o BE) y TP caen dentro de la misma barra, **se asume el stop** (supuesto conservador).
* El break-even se revisa después de las salidas y se activa para la barra siguiente, porque no se conoce el orden de high y low dentro de la barra.
* Una sola posición a la vez.

## 8. Win rate de break-even (umbral a superar)
Con R = TP / SL = 3.0 / 1.5 = 2:

```
p* = 1 / (1 + R) = 1 / 3 = 33.3 %          (sin costos)
```

Con costos: en barras de 4 h el riesgo típico por operación es 1.5 × ATR ≈ 1.5 × 1.07 % ≈ 1.60 % del precio, así que 30 bps equivalen a c ≈ 0.19 R. Entonces:

```
p*_costos = (1 + c) / (R + 1) ≈ 1.19 / 3 ≈ 39.6 %
```

**La estrategia debe superar p\* = 33.3 % (sin costos) y ≈ 40 % con costos.** Las salidas por break-even cuentan como pérdidas porque cuestan la comisión.

## 9. Criterio de viabilidad (Calmar)
`Calmar = CAGR / |Drawdown máximo|`: **> 1 viable**, 0.5 a 1 marginal, < 0.5 no viable.

## 10. Auditoría de sesgos
| Sesgo | Estado | Evidencia |
|---|---|---|
| Look-ahead | Controlado | Todos los indicadores son causales (rolling y EWM con `adjust=False`). La prueba de truncamiento del pipeline completo (`test_pipeline_truncation`, t = 800, 1800, 3000) recalcula indicadores, señales y backtest sobre `df.iloc[:t+1]` y el valor en *t* no cambia. La orden se ejecuta al open de t+1. |
| Survivorship | Bajo | Un solo activo (BTC), que existió todo el periodo. No se elige de un universo filtrado por los que sobrevivieron. Aun así, estudiar BTC por ser el cripto más exitoso es un sesgo de selección leve. |
| Overfitting / data snooping | Moderado | Parámetros estándar de la literatura (10/40, RSI 14, MACD 12-26-9, BB 20-2), fijados antes de correr y **sin optimizar**. SL/TP/ρ se eligieron a priori. El cambio de 1 h a 4 h se hizo después de ver los resultados de 1 h: es una decisión informada por la muestra, justificada por un diagnóstico de costos y no por buscar el mejor resultado; es la única modificación. Se reportan train, test y validation por separado. |
| Ejecución optimista | Controlado parcialmente | Entrada al open siguiente con slippage, empate → stop, gaps → salida al open. Limitación: la barra de 4 h se construye desde barras de 5 min y hay huecos de datos (≈3,700 filas NaN eliminadas); en huecos largos el precio real pudo saltar el stop. |
| Selección / periodo de muestra | Relevante | Solo 19 meses (jun-2022 a dic-2023): incluye el mercado bajista de 2022 (caída de FTX) y la recuperación de 2023. No hay ciclos completos; el resultado puede no generalizar. |
