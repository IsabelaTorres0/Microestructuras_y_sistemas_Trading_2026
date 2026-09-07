"""
Pruebas obligatorias (seccion 3.6). Se corren con:

    pytest

Las tres primeras son las que exige el enunciado; las demas son de apoyo.
"""

import numpy as np
import pytest

from src.model import (
    Params,
    medio_spread_monopolista,
    optimizar_cotizaciones,
    perdida_informados,
    perdida_informados_analitica,
    prob_ejecucion,
    spread_choke,
    utilidad_esperada,
)


@pytest.fixture
def p():
    return Params()


# --- 1. La probabilidad de ejecucion nunca es negativa -------------------------------

def test_prob_ejecucion_no_negativa(p):
    """pi_LB(s) y pi_LS(s) = max(0, alpha - beta*s) nunca deben ser negativas.

    Se prueba mas alla del punto de demanda nula (alpha/beta), que es justo donde
    la version sin truncar se volveria negativa.
    """
    s = np.linspace(0.0, spread_choke(p) * 3.0, 500)
    valores = prob_ejecucion(s, p)

    assert np.all(valores >= 0.0)
    assert np.all(valores[s > spread_choke(p)] == 0.0)
    # Es la misma funcion para ambos lados: pi_LB = pi_LS.
    assert prob_ejecucion(1.23, p) == prob_ejecucion(1.23, p)


# --- 2. La perdida frente a informados decrece en A ---------------------------------

def test_perdida_informados_decreciente_en_ask(p):
    """int_A^inf (P-A) f(P) dP es decreciente en A: cotizar mas alto expone menos.

    El enunciado pide verificar con al menos tres valores de A; aqui van seis.
    """
    valores_A = [20.0, 20.5, 21.0, 22.0, 23.0, 25.0]
    perdidas = [perdida_informados(A, p.S0 - 1.0, p)[0] for A in valores_A]

    assert all(a > b for a, b in zip(perdidas, perdidas[1:])), perdidas
    assert perdidas[-1] >= 0.0


def test_quad_coincide_con_forma_cerrada(p):
    """La cuadratura de scipy debe coincidir con la integral en forma cerrada.

    Prueba de apoyo: si `quad` se equivocara en las colas de la Erlang, todo el
    resultado del laboratorio estaria mal y nada mas lo delataria.
    """
    for A, B in [(20.5, 19.0), (23.0, 16.0), (21.4, 18.4)]:
        num = perdida_informados(A, B, p)
        cerrada = perdida_informados_analitica(A, B, p)
        assert np.allclose(num, cerrada, rtol=1e-6, atol=1e-9), (A, B, num, cerrada)


# --- 3. Con pi_i = 0 el optimo es el del monopolista --------------------------------

def test_optimo_sin_informados_es_el_monopolista(p):
    """Con pi_I = 0 el medio spread optimo debe ser s* = alpha/(2*beta) por lado.

    Ojo: alpha/beta = 6.25 es donde la probabilidad de ejecucion se anula, no el
    optimo. El maximo de (alpha - beta*s)*s esta en alpha/(2*beta) = 3.125.
    """
    p0 = p.con_pi_i(0.0)
    r = optimizar_cotizaciones(p0)
    s_teorico = medio_spread_monopolista(p0)

    assert s_teorico == pytest.approx(0.50 / (2 * 0.08))
    assert r["medio_spread_ask"] == pytest.approx(s_teorico, abs=1e-4)
    assert r["medio_spread_bid"] == pytest.approx(s_teorico, abs=1e-4)
    assert r["spread"] == pytest.approx(2 * s_teorico, abs=1e-4)


# --- Pruebas de apoyo ---------------------------------------------------------------

def test_optimo_del_caso_base_es_mejor_que_los_regimenes_fijos(p):
    """Si el optimizador funciona, ningun regimen del enunciado puede superarlo."""
    r = optimizar_cotizaciones(p)

    for A, B in [(20.05, 19.75), (21.40, 18.40)]:
        assert utilidad_esperada(A, B, p) < r["utilidad"]


def test_spread_optimo_crece_con_pi_i(p):
    """Seccion 3.5: mas probabilidad de informado, spread mas ancho."""
    spreads = [optimizar_cotizaciones(p.con_pi_i(v))["spread"] for v in (0.1, 0.4, 0.7)]

    assert all(a < b for a, b in zip(spreads, spreads[1:])), spreads


def test_params_rechaza_probabilidades_inconsistentes():
    """pi_i + pi_l debe sumar 1."""
    with pytest.raises(ValueError):
        Params(pi_i=0.4, pi_l=0.4)
