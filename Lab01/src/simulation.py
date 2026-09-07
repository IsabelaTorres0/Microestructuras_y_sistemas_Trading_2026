"""
Simulador de trades (seccion 3.3).

Un trade por iteracion: llega un trader, se sortea el valor verdadero P y se registra
el resultado del formador de mercado. Toda la logica vive aqui; los notebooks solo llaman.

SUPUESTOS (hay que poder defenderlos en la exposicion):

1. P ~ Erlang(K, lambda) se sortea de nuevo en CADA trade. El formador no aprende de
   los trades anteriores: el modelo de Copeland-Galai es de un solo periodo repetido,
   no un modelo de aprendizaje bayesiano.
2. El trader de liquidez (prob. pi_L) no mira P y elige lado con probabilidad 1/2.
   Su ejecucion se FUERZA: la simulacion no descuenta pi_LB(s). Es la advertencia de
   interpretacion obligatoria del enunciado: los resultados son rentabilidad por trade
   EJECUTADO, no por unidad de tiempo, y por eso un regimen amplio que en la realidad
   casi nunca se ejecutaria sale favorecido.
3. El trader informado (prob. pi_I) conoce P y solo opera cuando le conviene: compra al
   Ask si P > A, vende al Bid si P < B. Si B <= P <= A no opera y no hay trade.
4. P&L valuado contra el P de esa iteracion: si el trader compra al Ask el formador
   vende y gana A - P; si el trader vende al Bid el formador compra y gana P - B.
5. El inventario del formador se mueve al reves del trader: si el trader compra, el
   formador queda corto (-1).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import Params

SEMILLA = 42

# Regimenes obligatorios del enunciado (seccion 3.3), como (ask, bid).
REGIMENES_FIJOS = {
    "Estrecho": (20.05, 19.75),
    "Amplio": (21.40, 18.40),
}


@dataclass(frozen=True)
class Cotizacion:
    """Par (ask, bid) con nombre, para no andar pasando tuplas sueltas."""

    nombre: str
    ask: float
    bid: float

    @property
    def spread(self) -> float:
        return self.ask - self.bid


def regimenes(resultado: dict) -> dict[str, Cotizacion]:
    """Los tres regimenes del enunciado. `resultado` viene de optimizar_cotizaciones."""
    cotizaciones = {"Óptimo": (resultado["ask"], resultado["bid"])}
    cotizaciones.update(REGIMENES_FIJOS)
    return {n: Cotizacion(n, a, b) for n, (a, b) in cotizaciones.items()}


def _sortear(cot: Cotizacion, p: Params, forma, rng):
    """Sortea trades y devuelve (P, informado, lado, pnl, cambio_inventario).

    `forma` es un entero (una corrida) o una tupla (n_corridas, n_trades), asi que la
    misma funcion sirve para el detalle trade a trade y para el Monte Carlo.
    lado: 1 el trader compra al Ask, -1 vende al Bid, 0 el informado se abstiene.
    """
    # Erlang(K, lambda) es una Gamma de forma K y escala 1/lambda.
    P = rng.gamma(p.K, 1.0 / p.lam, size=forma)

    informado = rng.random(forma) < p.pi_i
    compra_liquidez = rng.random(forma) < 0.5

    lado_informado = np.where(P > cot.ask, 1, np.where(P < cot.bid, -1, 0))
    lado_liquidez = np.where(compra_liquidez, 1, -1)
    lado = np.where(informado, lado_informado, lado_liquidez)

    pnl = np.where(lado == 1, cot.ask - P, np.where(lado == -1, P - cot.bid, 0.0))
    cambio_inventario = -lado.astype(float)

    return P, informado, lado, pnl, cambio_inventario


def simular_trades(cot: Cotizacion, p: Params, n_trades: int, semilla: int = SEMILLA) -> pd.DataFrame:
    """Detalle trade por trade de una corrida. Registra lo que pide la seccion 3.3."""
    rng = np.random.default_rng(semilla)
    P, informado, lado, pnl, cambio_inventario = _sortear(cot, p, n_trades, rng)

    return pd.DataFrame({
        "P": P,
        "informado": informado,
        "lado": lado,
        "pnl": pnl,
        "cambio_inventario": cambio_inventario,
        "pnl_acumulado": pnl.cumsum(),
        "inventario_acumulado": cambio_inventario.cumsum(),
    })


def simular_corridas(cot: Cotizacion, p: Params, n_corridas: int, n_trades: int,
                     semilla: int = SEMILLA) -> dict[str, np.ndarray]:
    """Monte Carlo: n_corridas independientes de n_trades. Devuelve P&L e inventario finales.

    Se sortea la matriz completa (n_corridas, n_trades) de un jalon en vez de iterar en
    Python: son 10^6 numeros, cabe de sobra en memoria y es dos ordenes de magnitud mas rapido.
    """
    rng = np.random.default_rng(semilla)
    _, _, lado, pnl, cambio_inventario = _sortear(cot, p, (n_corridas, n_trades), rng)

    return {
        "pnl_final": pnl.sum(axis=1),
        "inventario_final": cambio_inventario.sum(axis=1),
        "trades_ejecutados": (lado != 0).sum(axis=1),
    }


def resumen_simulacion(sims: dict[str, pd.DataFrame], regs: dict[str, Cotizacion]) -> pd.DataFrame:
    """Tabla comparativa de los tres regimenes sobre la corrida de detalle."""
    filas = []
    for nombre, d in sims.items():
        inf = d[d["informado"]]
        liq = d[~d["informado"]]
        filas.append({
            "Régimen": nombre,
            "Spread": regs[nombre].spread,
            "P&L total": d["pnl"].sum(),
            "P&L por trade": d["pnl"].mean(),
            "P&L de liquidez": liq["pnl"].sum(),
            "P&L de informados": inf["pnl"].sum(),
            "% informados sin operar": (inf["lado"] == 0).mean() * 100.0,
            "Inventario final": d["cambio_inventario"].sum(),
        })
    return pd.DataFrame(filas).set_index("Régimen")


def resumen_montecarlo(mc: dict[str, dict], n_trades: int) -> pd.DataFrame:
    """Tabla del Monte Carlo: P&L promedio, desviacion estandar y probabilidad de perdida."""
    filas = []
    for nombre, r in mc.items():
        pnl = r["pnl_final"]
        filas.append({
            "Régimen": nombre,
            "P&L promedio": pnl.mean(),
            "Desv. estándar": pnl.std(ddof=1),
            "P(pérdida)": float((pnl < 0).mean()),
            "P&L por trade": pnl.mean() / n_trades,
        })
    return pd.DataFrame(filas).set_index("Régimen")


def deriva_inventario_teorica(cot: Cotizacion, p: Params, n_trades: int) -> float:
    """Deriva sistematica esperada del inventario: n * pi_I * [F(B) - (1 - F(A))].

    Solo el informado desbalancea de forma sostenida; el de liquidez es caminata aleatoria.
    Sirve para contrastar la simulacion contra un numero cerrado.
    """
    dist = p.distribucion()
    return n_trades * p.pi_i * float(dist.cdf(cot.bid) - dist.sf(cot.ask))
