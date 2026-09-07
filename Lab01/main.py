"""
Punto de entrada del laboratorio. Reproduce todos los resultados con:

    python main.py

Semilla global fijada en SEMILLA para que la simulacion sea reproducible.
"""

import numpy as np

from src.model import (
    Params,
    medio_spread_monopolista,
    optimizar_cotizaciones,
    spread_choke,
    tabla_sensibilidad,
)
from src.plots import (
    DIR_FIGURAS,
    figura_inventario,
    figura_montecarlo,
    figura_pnl_acumulado,
    figura_prob_ejecucion,
    figura_sensibilidad,
)
from src.simulation import (
    deriva_inventario_teorica,
    regimenes,
    resumen_montecarlo,
    resumen_simulacion,
    simular_corridas,
    simular_trades,
)

SEMILLA = 42


def reporte_optimizacion(p: Params) -> dict:
    """Seccion 3.2: resuelve y reporta las cotizaciones optimas del caso base."""
    r = optimizar_cotizaciones(p)

    print("=" * 68)
    print("3.1  PARAMETROS DEL CASO BASE")
    print("=" * 68)
    dist = p.distribucion()
    print(f"  Precio de referencia S0      : {p.S0:.2f}")
    print(f"  Valor verdadero P            : Erlang(K={p.K}, lambda={p.lam:g})"
          f"  ->  E[P]={dist.mean():.2f}, sd={dist.std():.2f}")
    print(f"  Prob. informado   pi_I       : {p.pi_i:.2f}")
    print(f"  Prob. liquidez    pi_L       : {p.pi_l:.2f}")
    print(f"  Demanda no informada         : max(0, {p.alpha:.2f} - {p.beta:.2f}*s)"
          f"  ->  se anula en s = {spread_choke(p):.2f}")

    print()
    print("=" * 68)
    print("3.2  COTIZACIONES OPTIMAS")
    print("=" * 68)
    print(f"  Bid optimo                   : {r['bid']:.2f}")
    print(f"  Ask optimo                   : {r['ask']:.2f}")
    print(f"  Spread                       : {r['spread']:.2f}")
    print(f"  Utilidad esperada por trade  : {r['utilidad']:.2f}")
    print()
    print("  Desglose (por trader que llega):")
    print(f"    Ganancia frente a liquidez : {r['ganancia_liquidez']:.4f}")
    print(f"    Perdida esperada lado Ask  : {p.pi_i * r['perdida_ask']:.4f}"
          f"   (integral = {r['perdida_ask']:.4f})")
    print(f"    Perdida esperada lado Bid  : {p.pi_i * r['perdida_bid']:.4f}"
          f"   (integral = {r['perdida_bid']:.4f})")
    print()
    print("  Medios spreads y probabilidad de ejecucion:")
    print(f"    Lado Ask: s = {r['medio_spread_ask']:.4f}   pi_LB = {r['prob_ejec_ask']:.4f}")
    print(f"    Lado Bid: s = {r['medio_spread_bid']:.4f}   pi_LS = {r['prob_ejec_bid']:.4f}")
    print()
    print(f"  Referencia sin informados (pi_I = 0): s* = alpha/(2*beta) = "
          f"{medio_spread_monopolista(p):.4f} por lado")

    return r


def reporte_simulacion(p: Params, r: dict, n_trades: int = 10_000):
    """Seccion 3.3: detalle de una corrida de n_trades para los tres regimenes."""
    regs = regimenes(r)
    sims = {nombre: simular_trades(cot, p, n_trades) for nombre, cot in regs.items()}

    print()
    print("=" * 68)
    print(f"3.3  SIMULACION DE {n_trades:,} TRADES POR REGIMEN")
    print("=" * 68)
    print(resumen_simulacion(sims, regs).round(2).to_string())
    print()
    print("  Inventario: deriva teorica n*pi_I*[F(B) - (1-F(A))] contra la corrida")
    for nombre, cot in regs.items():
        teorica = deriva_inventario_teorica(cot, p, n_trades)
        print(f"    {nombre:9s}: teorica = {teorica:8.1f}   simulada = "
              f"{sims[nombre]['cambio_inventario'].sum():8.1f}")
    print()
    print("  ADVERTENCIA (exigida por el enunciado): la simulacion fuerza la ejecucion")
    print("  del trader de liquidez, asi que estas cifras son P&L por trade EJECUTADO,")
    print("  no por unidad de tiempo. El regimen amplio sale favorecido por eso.")

    return regs, sims


def reporte_montecarlo(p: Params, regs: dict, n_corridas: int = 1_000, n_trades: int = 1_000):
    """Seccion 3.3: Monte Carlo de n_corridas independientes."""
    mc = {nombre: simular_corridas(cot, p, n_corridas, n_trades)
          for nombre, cot in regs.items()}

    print()
    print("=" * 68)
    print(f"3.3  MONTE CARLO: {n_corridas:,} CORRIDAS DE {n_trades:,} TRADES")
    print("=" * 68)
    print(resumen_montecarlo(mc, n_trades).round(4).to_string())

    return mc


def reporte_sensibilidad(p: Params):
    """Seccion 3.5: spread optimo contra pi_I."""
    tabla = tabla_sensibilidad(p=p)

    print()
    print("=" * 68)
    print("3.5  SENSIBILIDAD A pi_I")
    print("=" * 68)
    print(tabla.round(4).to_string())
    print()
    print(f"  Referencia pi_I -> 0: spread = 2*alpha/(2*beta) = "
          f"{2 * medio_spread_monopolista(p):.4f}")

    return tabla


def generar_figuras(p: Params, r: dict, regs: dict, sims: dict, mc: dict, tabla, n_trades: int):
    """Seccion 3.4: las cuatro figuras obligatorias, mas la de sensibilidad."""
    figura_prob_ejecucion(p, r)
    figura_pnl_acumulado(sims, regs)
    figura_inventario(sims, regs)
    figura_montecarlo(mc, n_trades)
    figura_sensibilidad(tabla, p)

    print()
    print(f"  Figuras guardadas en {DIR_FIGURAS}")


def main():
    np.random.seed(SEMILLA)
    p = Params()

    r = reporte_optimizacion(p)
    regs, sims = reporte_simulacion(p, r)
    mc = reporte_montecarlo(p, regs)
    tabla = reporte_sensibilidad(p)
    generar_figuras(p, r, regs, sims, mc, tabla, n_trades=1_000)


if __name__ == "__main__":
    main()
