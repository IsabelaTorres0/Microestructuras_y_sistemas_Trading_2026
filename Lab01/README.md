# Lab 01 — Cotizaciones óptimas de un formador de mercado

**Microestructuras y Sistemas de Trading — ITESO**
Modelo de Copeland y Galai (1983)

## Integrantes

<!-- PENDIENTE: confirmar nombres completos y el número de equipo antes de entregar -->
- Jesús Flores
- Isabela Torres

## Descripción

Se implementa el modelo de Copeland-Galai para un formador de mercado que cotiza un Bid y un Ask
sobre un activo cuyo valor verdadero $P$ es desconocido y se distribuye Erlang(K=60, λ=3). Frente
a él llegan traders de liquidez, que no saben nada y operan de todos modos, y traders informados,
que conocen $P$ y solo operan cuando el formador está mal cotizado. El proyecto resuelve
numéricamente las cotizaciones que maximizan la utilidad esperada por trader que llega, simula
10,000 trades bajo tres regímenes de cotización, corre un Monte Carlo de 1,000 corridas y mide cómo
se abre el spread óptimo cuando aumenta la probabilidad de que quien llega esté informado.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows; en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducir todos los resultados

```bash
python main.py
```

Ese comando imprime las cuatro secciones del reporte (parámetros, optimización, simulación y
Monte Carlo, sensibilidad) y regenera las cinco figuras en `docs/figuras/`.

Las pruebas se corren aparte:

```bash
pytest
```

**Semilla aleatoria:** fijada en `SEMILLA = 42`, declarada en `main.py` (`np.random.seed(42)`) y en
`src/simulation.py`, donde cada simulación crea su propio `np.random.default_rng(semilla)`. Con eso
las cifras de abajo se reproducen exactamente.

## Estructura

```
main.py                    ejecuta todo el proyecto
src/model.py               utilidad esperada, integrales de pérdida y optimización
src/simulation.py          simulador de trades y Monte Carlo
src/plots.py               las figuras
tests/test_model.py        pruebas con pytest
notebooks/analysis.ipynb   solo análisis y figuras, sin lógica
docs/presentacion.pdf      presentación de la exposición
```

## Resultados del caso base

| Cantidad | Valor |
|---|---|
| Bid óptimo | 16.45 |
| Ask óptimo | 23.43 |
| Spread | 6.98 |
| Utilidad esperada por trader que llega | 0.84 |

Medio spread de 3.53 del lado Ask (probabilidad de ejecución 0.218) y de 3.45 del lado Bid
(probabilidad 0.224). Como referencia, sin informados ($\pi_I = 0$) el óptimo analítico del
monopolista es $s^* = \alpha/(2\beta) = 3.125$ por lado.

Simulación de 10,000 trades:

| Régimen | Spread | P&L total | P&L de liquidez | P&L de informados | % informados sin operar |
|---|---|---|---|---|---|
| Óptimo (23.43 / 16.45) | 6.98 | +20,353.14 | +21,253.28 | −900.14 | 81.4% |
| Estrecho (20.05 / 19.75) | 0.30 | −6,634.01 | +1,017.36 | −7,651.38 | 4.6% |
| Amplio (21.40 / 18.40) | 3.00 | +5,551.81 | +9,202.41 | −3,650.60 | 43.9% |

Monte Carlo de 1,000 corridas de 1,000 trades:

| Régimen | P&L promedio | Desv. estándar | P(pérdida) |
|---|---|---|---|
| Óptimo | 2,008.60 | 87.14 | 0.000 |
| Estrecho | −672.70 | 77.58 | 1.000 |
| Amplio | 542.91 | 77.94 | 0.000 |

## Preguntas de análisis

### 1. ¿Por qué los traders informados generan la necesidad de un spread?

En el régimen estrecho el formador cobra un medio spread de 0.15 del lado del Ask y 0.05 del lado
del Bid. Contra los traders de liquidez gana **+1,017.36** en 10,000 trades, que es exactamente el
negocio que quiere hacer. Pero contra los informados pierde **−7,651.38**, siete veces y media más
de lo que ganó, y el resultado neto es **−6,634.01**: cotizar pegado a $S_0$ lo lleva a la quiebra.

La razón es que el informado no es un trader más: solo aparece cuando el formador está equivocado.
Con el spread estrecho apenas el **4.6%** de los informados se abstiene, o sea que prácticamente
cada vez que llega uno, opera, y opera contra él. El spread positivo no existe para cobrarle al
cliente, existe para que el informado tenga que estar *muy* en lo cierto antes de que le convenga
operar. En el óptimo esa misma pérdida cae a −900.14 porque el **81.4%** de los informados ya no
encuentra rentable operar.

### 2. ¿Cómo cambia el costo de selección adversa conforme se amplía el spread?

Cae, y cae por dos vías distintas que conviene separar:

| Régimen | Spread | P&L de informados | % informados sin operar |
|---|---|---|---|
| Estrecho | 0.30 | −7,651.38 | 4.6% |
| Amplio | 3.00 | −3,650.60 | 43.9% |
| Óptimo | 6.98 | −900.14 | 81.4% |

La primera vía es que cada trade informado duele menos: si el Ask es más alto, el informado que
compra se lleva $P - A$, que es menor. La segunda, y es la que más pesa, es que **el informado
simplemente deja de operar**: conforme $[B, A]$ se ensancha, el valor verdadero $P$ cae adentro del
intervalo cada vez más seguido y no hay trade. Pasar de 4.6% a 81.4% de abstención es lo que
convierte una pérdida de 7,651 en una de 900.

Lo que impide ampliar el spread indefinidamente es la otra mitad de la función objetivo: la
probabilidad de ejecución no informada $\pi_{LB}(s) = \max(0,\ 0.50 - 0.08 s)$ se anula en
$s = 6.25$. El óptimo de 3.53 y 3.45 por lado es el punto donde dejar de perder contra el informado
todavía compensa lo que se deja de ganar con el de liquidez.

### 3. ¿Cuál régimen acumula el mayor desbalance de inventario y por qué?

Medido sobre 1,000 corridas de 10,000 trades (una sola corrida es demasiado ruidosa para decidirlo):

| Régimen | Deriva teórica | Inventario medio | Inventario absoluto medio | Desv. estándar |
|---|---|---|---|---|
| Óptimo | −76.2 | −74.5 | **90.6** | 82.6 |
| Estrecho | +13.1 | +16.0 | 80.1 | **97.6** |
| Amplio | −28.0 | −24.6 | 75.8 | 91.4 |

Hay que distinguir dos cosas que suelen confundirse:

- **Ruido.** El trader de liquidez elige lado al azar, así que aporta una caminata aleatoria pura.
  Gana el **estrecho**, porque es el que más trades ejecuta (9,815 de 10,000) y una caminata más
  larga tiene más varianza: desviación estándar de 97.6.
- **Deriva sistemática.** Solo el informado desbalancea de forma sostenida, y ahí gana el
  **óptimo**. Sus cotizaciones están metidas en las colas de la Erlang, que es asimétrica a la
  derecha, así que la cola de arriba pesa más que la de abajo: el informado compra al Ask más
  seguido de lo que vende al Bid y el formador queda **corto**, unas 75 unidades en promedio. La
  fórmula cerrada $n \cdot \pi_I \cdot [F(B) - (1 - F(A))]$ reproduce la simulación con menos de
  3 unidades de diferencia en los tres regímenes.

Sumando ambos efectos, el **régimen óptimo es el que más inventario carga en valor absoluto**
(90.6 unidades). O sea que el régimen que maximiza la utilidad esperada es también el que deja al
formador con la posición más grande y más direccional.

**El riesgo real que el modelo no captura:** ese inventario tiene que financiarse y valuarse a
mercado en el tiempo, y el modelo no tiene tiempo. Aquí cada trade se liquida contra el $P$ de esa
iteración y no hay consecuencia por quedarse corto 75 unidades; en la realidad esa posición se
carga con margen, se revalúa mientras el precio se mueve y puede forzar una liquidación en el peor
momento. Un formador real cotiza asimétricamente para descargar inventario (skewing) justo porque
sí paga ese costo. Aquí no aparece porque el modelo es de un solo periodo repetido, sin memoria.

### 4. ¿Cómo se comporta el spread óptimo al variar $\pi_I$? ¿Coincide con la teoría?

| $\pi_I$ | Ask | Bid | Spread | Utilidad esperada |
|---|---|---|---|---|
| 0.1 | 23.11 | 16.71 | 6.40 | 1.3787 |
| 0.4 | 23.43 | 16.45 | 6.98 | 0.8403 |
| 0.7 | 24.00 | 16.01 | 7.99 | 0.3366 |

El spread es **creciente en $\pi_I$**, que es lo que predice la teoría: entre más probable es que
quien llega esté informado, más caro sale cotizar cerca de $S_0$ y más se abre el formador. La
utilidad esperada, en cambio, cae (1.38 → 0.84 → 0.34), porque ampliar el spread es una defensa,
no un negocio: reduce la pérdida contra el informado a costa de espantar al de liquidez.

El contraste cuantitativo con la teoría es el caso límite $\pi_I \to 0$. Sin informados el problema
se reduce a maximizar $(\alpha - \beta s)\,s$ por lado, cuyo máximo analítico está en
$s^* = \alpha/(2\beta) = 3.125$, o sea un spread de **6.25**. La optimización numérica con
$\pi_I = 0$ devuelve exactamente eso (es una de las pruebas de `tests/test_model.py`), y con
$\pi_I = 0.1$ da 6.40, apenas por encima. La curva sale del valor teórico correcto.

Vale la pena señalar que $\alpha/\beta = 6.25$ por lado es el punto donde la probabilidad de
ejecución se anula, **no** el óptimo. Que el spread total del caso $\pi_I \to 0$ coincida
numéricamente con ese 6.25 es consecuencia de que $2 \cdot \alpha/(2\beta) = \alpha/\beta$.

### 5. Tres limitaciones del modelo para un formador de mercado real

1. **No hay tiempo ni costo de inventario.** Como se explicó en la pregunta 3, el formador termina
   corto unas 75 unidades y el modelo no le cobra nada por eso: ni financiamiento, ni margen, ni
   revaluación. Un formador real cotiza asimétricamente para descargar posición, y eso aquí ni
   siquiera es representable.
2. **El formador no aprende.** $P$ se sortea de nuevo en cada trade y las cotizaciones se quedan
   fijas. En la realidad el flujo *es* información: una racha de compras al Ask es evidencia de que
   el valor está por encima de $S_0$, y cualquier formador movería su precio de referencia. Este
   modelo es de un solo periodo repetido, sin actualización bayesiana.
3. **La probabilidad de ejecución es un supuesto lineal impuesto, no un mercado.**
   $\pi_{LB}(s) = \max(0,\ 0.50 - 0.08 s)$ trata al formador como monopolista: nadie más cotiza y la
   única razón por la que el cliente no opera es que el spread le pareció caro. En un mercado real
   hay competencia por prioridad en el libro, tamaños distintos por trade (aquí todos los trades son
   de una unidad) y órdenes que se parten entre varios formadores.

**Limitación de interpretación, exigida por el enunciado:** la simulación fuerza la ejecución del
trader de liquidez, es decir ignora $\pi_{LB}(s)$ al momento de simular. Por eso las cifras de las
tablas son **rentabilidad por trade ejecutado y no por unidad de tiempo**, y un régimen muy amplio
—que en la realidad casi nunca se ejecutaría— sale favorecido. Se ve claro en el régimen óptimo: su
utilidad teórica es 0.84 por trader que *llega*, que sí descuenta la probabilidad de ejecución de
~0.22, mientras que el P&L por trade de la simulación es 2.04, que no la descuenta.

## Uso de asistencia de IA

<!-- PENDIENTE: ajustar según lo que haya usado cada quien -->
Se usó Claude Code como asistente en: la estructura del proyecto, la implementación de las
integrales de pérdida con `scipy.integrate.quad` y su validación contra la forma cerrada de la
Erlang, la vectorización del simulador de Monte Carlo y la redacción de este README. Las decisiones
de modelado, los supuestos de la simulación y la interpretación de los resultados son
responsabilidad de los integrantes, y ambos podemos explicar cualquier línea del código entregado.
