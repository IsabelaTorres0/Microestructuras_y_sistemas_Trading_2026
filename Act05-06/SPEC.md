# SPEC — Estrategia BTC 5 min: tendencia (SMA10/SMA40) + momentum (RSI14), solo largo

Este documento fija todas las decisiones **antes** de correr cualquier backtest. Cualquier cambio posterior debe registrarse con fecha y motivo.

---

## 1. Universe and frequency

| Campo | Valor |
|---|---|
| Activo | BTC-USD (spot) |
| Fuente | `btc_project_train.csv` (proporcionado por el curso) |
| Frecuencia | Barras de 5 minutos, UTC |
| Rango total | 2022-06-01 00:00 → 2023-12-31 00:00 |

**Split cronológico (sin barajar):**

| Conjunto | Desde | Hasta | Barras (malla 5 min) | Uso |
|---|---|---|---|---|
| Train | 2022-06-01 00:00 | 2023-03-31 23:55 | 87,552 | Diseño y depuración |
| Validation | 2023-04-01 00:00 | 2023-08-31 23:55 | 44,064 | Elegir entre pocas variantes predefinidas |
| Test | 2023-09-01 00:00 | 2023-12-31 00:00 | 34,849 | Se corre **una sola vez**, al final |

**Limpieza de datos (decidida tras inspección, sin mirar desempeño):**

1. El archivo tiene ~1,100 timestamps fuera de la malla (intervalos de 1–4 min). Se re-muestrea a una malla fija de 5 min con `floor('5min')`: Open = primero, High = máx., Low = mín., Close = último.
2. Hay 4,323 barras de la malla sin precio (huecos de hasta ~22 h). **No se rellenan.** En una barra sin precio no se generan señales ni se ejecutan órdenes.
3. La columna `Volume` está 47% vacía y contiene valores imposibles (máx. ≈ 8.5×10¹²), por lo que **no se usa ningún indicador de volumen**.

---

## 2. Features

| Indicador | Ventana | Familia | Definición |
|---|---|---|---|
| SMA10 | 10 barras (50 min) | Tendencia (rápida) | Media simple del Close |
| SMA40 | 40 barras (3 h 20 min) | Tendencia (lenta) | Media simple del Close |
| RSI14 | 14 barras | Momentum / oscilador | RSI de Cutler: `100 − 100/(1 + mean(ganancias,14)/mean(pérdidas,14))` |
| ATR14 | 14 barras | Volatilidad (solo para SL/TP) | Media simple del True Range |

**Regla de NaN:** si cualquier barra dentro de la ventana de un indicador no tiene precio, el indicador es NaN y no hay señal. Se usan medias simples (y RSI de Cutler en lugar de Wilder) precisamente para que esta regla sea exacta y un hueco no contamine el indicador indefinidamente.

**Warm-up:** las primeras 40 barras válidas no generan señal.

---

## 3. Entry rule

Señales normalizadas en [−1, 1]:

```
s_trend_t = sign(SMA10_t − SMA40_t)
s_mom_t   = clip((RSI14_t − 50) / 20, −1, 1)

score_t   = 0.5 · s_trend_t + 0.5 · s_mom_t
```

**Entrada larga:** `score_t ≥ θ`, con **θ = 0.75**, y sin posición abierta.

Interpretación: θ = 0.75 exige tendencia alcista (`s_trend = +1`) **y** RSI ≥ 60. El filtro de momentum existe para reducir la frecuencia de operación: en train, SMA10 y SMA40 se cruzan ~8 veces al día, lo que con los costos de la sección 6 sería inviable.

---

## 4. Exit rule

Sea `E` el precio de entrada y `ATR_t` el ATR14 de la barra de señal (conocido al cierre de t).

| Parámetro | Valor |
|---|---|
| Stop-loss | `SL = E − 4 · ATR_t` |
| Take-profit | `TP = E + 8 · ATR_t` |
| Ratio R = TP/SL | **2** |
| Señal opuesta | Si `score_t ≤ −θ` con posición abierta, se cierra en la apertura de t+1 |
| Holding máximo | 48 barras (4 h); se cierra en la apertura de la barra 49 |

Los niveles SL y TP se fijan al entrar y no se mueven (sin trailing).

---

## 5. Sizing

**Esquema:** riesgo fijo fraccional.

```
riesgo_$  = equity_t × 0.10 %
acciones  = riesgo_$ / (E − SL)
nocional  = min(acciones × E, equity_t)      # tope 1× equity: spot, sin apalancamiento
```

- Capital inicial: **$100,000**. Se permiten fracciones de BTC.
- El riesgo presupuestado es la pérdida de precio si se toca el SL, **sin incluir costos**.
- Con SL mediano ≈ 29 bps, el nocional típico es ≈ 0.34× equity. El tope de 1× solo se activa cuando `4·ATR < 10 bps`, lo que ocurre en ~5% de las barras. En ese caso el riesgo real es menor al presupuestado y la operación se marca con `capped = True`.
# Calcular sizing con rho

---

## 6. Costs

| Costo | Valor por lado | Justificación |
|---|---|---|
| Comisión | 10 bps | Tarifa taker base de exchanges spot de alta liquidez (nivel sin descuentos). Se asume taker porque la ejecución es a mercado en la apertura |
| Slippage | 1 bp | BTC-USD es de los pares más líquidos; 1 bp es conservador para órdenes de este tamaño en la apertura de una barra |
| Borrow fee | No aplica | Estrategia solo largo, sin apalancamiento |
| **Total ida y vuelta** | **22 bps** | |

El slippage se aplica en contra en cada ejecución: `fill_compra = precio × (1 + 0.0001)`, `fill_venta = precio × (1 − 0.0001)`. La comisión se cobra sobre el nocional ejecutado.

Se reportará sensibilidad de 0 a 50 bps ida y vuelta.

---

## 7. Conventions

1. **Tiempo de acción:** las señales se calculan con datos hasta el **cierre de t** y se ejecutan en la **apertura de t+1**. Nunca se opera al precio de la barra que generó la señal.
2. **Revisión de SL/TP:** desde la barra de entrada (inclusive) se revisa con High y Low de cada barra.
3. **Empates intrabar:** si en la misma barra `Low ≤ SL` y `High ≥ TP`, se asume que **se tocó primero el stop** → salida `stop_loss`. Es el supuesto pesimista, porque con OHLC no se puede saber el orden real.
4. **Gaps:** si una barra abre por debajo del SL, la salida es al **Open** (no al SL). Si abre por encima del TP, la salida es al TP (no se regala la mejora).
5. **Barras sin precio:** una posición abierta se mantiene; el SL/TP se evalúa en la siguiente barra válida con la regla de gaps.
6. **Una sola posición:** nunca hay más de una posición abierta. Una señal de entrada con posición abierta se ignora.
7. **Prioridad de salidas en una barra:** (1) SL/TP intrabar, (2) señal opuesta, (3) holding máximo.

---

## 8. Umbral a superar: win rate de break-even

Con ganancia `R` y pérdida `1` (en unidades de riesgo) y un costo ida y vuelta `c` expresado en esas mismas unidades:

```
p·(R − c) − (1 − p)·(1 + c) = 0   ⇒   p* = (1 + c) / (1 + R)
```

Valores de esta estrategia (ATR14 mediano en train = 7.27 bps):

| Concepto | Valor |
|---|---|
| SL mediano = 4 × 7.27 bps | 29.1 bps |
| Costo ida y vuelta | 22 bps |
| c = 22 / 29.1 | 0.757 |
| R | 2 |
| Sin costos: 1 / (1 + R) | 33.3 % |
| **Con costos: p\*** | **58.6 %** |

**La estrategia debe lograr un win rate mayor a 58.6 % para ser rentable.** Como el SL depende del ATR de cada operación, `c` cambia por operación. En `metrics.py` también se reportará el p* promedio calculado con el SL real de cada trade.

---

## 9. Registro de cambios

| Fecha | Cambio | Motivo |
|---|---|---|
| 2026-09-23 | Versión inicial | — |
