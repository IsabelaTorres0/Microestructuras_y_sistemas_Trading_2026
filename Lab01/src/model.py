from dataclasses import dataclass, replace

import numpy as np
from scipy import integrate, optimize, stats

# Cota inferior estricta para el Bid: el dominio es (0, S0], abierto en cero.
EPS_BID = 1e-6


@dataclass(frozen=True)
class Params:
    """Parametros del caso base (seccion 3.1). Fijos e iguales para todos los equipos."""

    S0: float = 19.90      # precio de referencia
    K: int = 60            # forma de la Erlang del valor verdadero
    lam: float = 3.0       # tasa de la Erlang
    pi_i: float = 0.40     # probabilidad de que el trader este informado
    pi_l: float = 0.60     # probabilidad de que el trader sea de liquidez
    alpha: float = 0.50    # intercepto de la demanda no informada
    beta: float = 0.08     # pendiente de la demanda no informada

    def __post_init__(self):
        if not np.isclose(self.pi_i + self.pi_l, 1.0):
            raise ValueError(f"pi_i + pi_l debe sumar 1, se recibio {self.pi_i + self.pi_l}")
        if self.beta <= 0 or self.alpha <= 0:
            raise ValueError("alpha y beta deben ser positivos")

    def distribucion(self):
        """Distribucion del valor verdadero P ~ Erlang(K, lambda).

        scipy parametriza la Erlang con `a` = K y `scale` = 1/lambda.
        """
        return stats.erlang(a=self.K, scale=1.0 / self.lam)

    def con_pi_i(self, pi_i: float) -> "Params":
        """Copia con otra probabilidad de informado, manteniendo pi_i + pi_l = 1.

        Se usa en el analisis de sensibilidad (seccion 3.5) y en la prueba con pi_i = 0.
        """
        return replace(self, pi_i=pi_i, pi_l=1.0 - pi_i)


def prob_ejecucion(s, p: Params):
    """Probabilidad de ejecucion de la demanda no informada: max(0, alpha - beta*s).

    Es la misma funcion para el lado de compra y el de venta (pi_LB = pi_LS).
    El truncamiento en cero es obligatorio: sin el, la probabilidad se volveria
    negativa para s > alpha/beta y la utilidad creceria sin cota al ampliar el spread.
    """
    return np.maximum(0.0, p.alpha - p.beta * np.asarray(s, dtype=float))


def spread_choke(p: Params) -> float:
    """Medio spread al que la probabilidad de ejecucion llega a cero: alpha/beta."""
    return p.alpha / p.beta


def ganancia_liquidez(A: float, B: float, p: Params) -> float:
    """Ganancia esperada frente a traders de liquidez (primer corchete de Pi)."""
    s_ask = A - p.S0
    s_bid = p.S0 - B
    return p.pi_l * float(
        prob_ejecucion(s_ask, p) * s_ask + prob_ejecucion(s_bid, p) * s_bid
    )


def perdida_informados(A: float, B: float, p: Params) -> tuple[float, float]:
    """Perdida esperada frente a informados, por lado, via integracion numerica.

    Lado Ask:  E[(P - A)+]  = int_A^inf (P - A) f(P) dP   -> el informado compra si P > A
    Lado Bid:  E[(B - P)+]  = int_0^B  (B - P) f(P) dP   -> el informado vende si P < B

    Se usa scipy.integrate.quad (cuadratura adaptativa), no sumas discretas.
    Devuelve las dos integrales por separado para poder analizarlas de forma independiente.
    """
    f = p.distribucion().pdf

    perd_ask, _ = integrate.quad(lambda P: (P - A) * f(P), A, np.inf, limit=200)
    perd_bid, _ = integrate.quad(lambda P: (B - P) * f(P), 0.0, B, limit=200)
    return float(perd_ask), float(perd_bid)


def perdida_informados_analitica(A: float, B: float, p: Params) -> tuple[float, float]:
    """Mismas integrales en forma cerrada, para validar la cuadratura.

    Para P ~ Gamma(K, escala 1/lambda) se cumple E[P * 1{P > A}] = (K/lambda) * (1 - F_{K+1}(A)),
    donde F_{K+1} es la CDF de una Gamma con forma K+1 e igual escala. De ahi:

        int_A^inf (P - A) f(P) dP = (K/lambda)*(1 - F_{K+1}(A)) - A*(1 - F_K(A))
        int_0^B  (B - P) f(P) dP = B*F_K(B) - (K/lambda)*F_{K+1}(B)

    No forma parte del modelo pedido; existe solo como contraste en tests/test_model.py.
    """
    escala = 1.0 / p.lam
    media = p.K * escala
    dist_k = stats.erlang(a=p.K, scale=escala)
    dist_k1 = stats.erlang(a=p.K + 1, scale=escala)

    perd_ask = media * dist_k1.sf(A) - A * dist_k.sf(A)
    perd_bid = B * dist_k.cdf(B) - media * dist_k1.cdf(B)
    return float(perd_ask), float(perd_bid)


def utilidad_esperada(A: float, B: float, p: Params) -> float:
    """Utilidad esperada por trader que llega:

        Pi(A,B) = pi_L * [pi_LB(A-S0)*(A-S0) + pi_LS(S0-B)*(S0-B)]
                - pi_I * [int_A^inf (P-A) f(P) dP + int_0^B (B-P) f(P) dP]
    """
    perd_ask, perd_bid = perdida_informados(A, B, p)
    return ganancia_liquidez(A, B, p) - p.pi_i * (perd_ask + perd_bid)


def objetivo(x, p: Params) -> float:
    """Negativo de Pi(A,B), para poder minimizar. x = (A, B)."""
    A, B = x
    return -utilidad_esperada(A, B, p)


def medio_spread_monopolista(p: Params) -> float:
    """Medio spread optimo analitico cuando pi_i = 0: s* = alpha / (2*beta).

    Con pi_i = 0 la utilidad por lado es pi_L * (alpha - beta*s) * s, una parabola
    concava cuyo maximo esta en la raiz de alpha - 2*beta*s = 0. Notese que
    alpha/beta (= 6.25 en el caso base) NO es el optimo sino el punto donde la
    probabilidad de ejecucion se anula y la utilidad vale exactamente cero.
    """
    return p.alpha / (2.0 * p.beta)


def optimizar_cotizaciones(p: Params, x0=None) -> dict:
    """Maximiza Pi(A,B) sujeto a A en [S0, inf) y B en (0, S0].

    Se minimiza `objetivo` con scipy.optimize.minimize (L-BFGS-B, que admite cotas
    por variable). Si no se da `x0` se corren varios arranques y se devuelve el mejor.

    El multistart no es adorno: para s > alpha/beta en ambos lados la probabilidad de
    ejecucion es cero, Pi es identicamente cero y el gradiente tambien, asi que un
    arranque en esa meseta hace que L-BFGS-B reporte convergencia en un punto que no
    es optimo. Los arranques por defecto viven dentro de la region con demanda positiva.
    """
    if x0 is not None:
        arranques = [list(x0)]
    else:
        s_mono = medio_spread_monopolista(p)
        arranques = [
            [p.S0 + s_mono, p.S0 - s_mono],
            [p.S0 + 0.5 * s_mono, p.S0 - 0.5 * s_mono],
            [p.S0 + 1.5 * s_mono, p.S0 - 1.5 * s_mono],
        ]

    cotas = [(p.S0, None), (EPS_BID, p.S0)]
    mejor = None
    for arranque in arranques:
        res = optimize.minimize(objetivo, arranque, args=(p,), method="L-BFGS-B", bounds=cotas)
        if res.success and (mejor is None or res.fun < mejor.fun):
            mejor = res

    if mejor is None:
        raise RuntimeError("Ningun arranque de la optimizacion convergio")

    A, B = float(mejor.x[0]), float(mejor.x[1])
    perd_ask, perd_bid = perdida_informados(A, B, p)

    return {
        "bid": B,
        "ask": A,
        "spread": A - B,
        "medio_spread_ask": A - p.S0,
        "medio_spread_bid": p.S0 - B,
        "utilidad": -float(mejor.fun),
        "ganancia_liquidez": ganancia_liquidez(A, B, p),
        "perdida_ask": perd_ask,
        "perdida_bid": perd_bid,
        "prob_ejec_ask": float(prob_ejecucion(A - p.S0, p)),
        "prob_ejec_bid": float(prob_ejecucion(p.S0 - B, p)),
    }


def tabla_sensibilidad(valores_pi_i=(0.1, 0.4, 0.7), p: Params = None):
    """Seccion 3.5: reoptimiza para varios pi_I y devuelve la tabla comparativa.

    Se devuelve un DataFrame indexado por pi_I para poder graficarlo directo.
    """
    import pandas as pd

    p = p or Params()
    filas = []
    for valor in valores_pi_i:
        r = optimizar_cotizaciones(p.con_pi_i(valor))
        filas.append({
            "pi_I": valor,
            "Ask": r["ask"],
            "Bid": r["bid"],
            "Spread": r["spread"],
            "Utilidad esperada": r["utilidad"],
        })
    return pd.DataFrame(filas).set_index("pi_I")
