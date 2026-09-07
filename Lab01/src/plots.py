"""
Generacion de las figuras obligatorias (seccion 3.4).

Cada funcion devuelve la figura de matplotlib y, si se le pasa `ruta`, la guarda.
Los notebooks solo llaman a estas funciones; no definen logica de graficado propia.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .model import Params, medio_spread_monopolista, prob_ejecucion, spread_choke

DIR_FIGURAS = Path(__file__).resolve().parents[1] / "docs" / "figuras"


def _guardar(fig, ruta):
    """Guarda la figura creando el directorio si hace falta."""
    if ruta is None:
        return
    ruta = Path(ruta)
    if not ruta.is_absolute():
        ruta = DIR_FIGURAS / ruta
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")


def figura_prob_ejecucion(p: Params, resultado: dict = None, ruta="01_prob_ejecucion.png"):
    """Figura 1: probabilidad de ejecucion no informada contra el medio spread.

    Marca el punto donde la probabilidad llega a cero (s = alpha/beta). Si se pasa
    `resultado` de optimizar_cotizaciones, marca tambien los medios spreads optimos.
    """
    choke = spread_choke(p)
    s = np.linspace(0.0, choke * 1.3, 600)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s, prob_ejecucion(s, p), lw=2, color="#1f77b4",
            label=r"$\pi_{LB}(s)=\pi_{LS}(s)=\max(0,\ \alpha-\beta s)$")

    ax.plot([choke], [0.0], "o", ms=9, color="#d62728", zorder=5,
            label=f"Probabilidad nula: $s$ = {choke:.2f}")
    ax.annotate(f"s = {choke:.2f}" + "\n" + r"$\pi_L$ = 0", xy=(choke, 0.0), xytext=(choke + 0.35, 0.085),
                arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728", fontsize=10)

    if resultado is not None:
        for lado, clave_s, clave_p, color in (
            ("Ask", "medio_spread_ask", "prob_ejec_ask", "#2ca02c"),
            ("Bid", "medio_spread_bid", "prob_ejec_bid", "#9467bd"),
        ):
            ax.plot([resultado[clave_s]], [resultado[clave_p]], "s", ms=8, color=color, zorder=5,
                    label=f"Óptimo {lado}: $s$ = {resultado[clave_s]:.3f}, "
                          rf"$\pi_L$ = {resultado[clave_p]:.3f}")

    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xlim(0, choke * 1.3)
    ax.set_ylim(-0.03, p.alpha * 1.12)
    ax.set_xlabel("Medio spread $s$ desde $S_0$ (unidades de precio)")
    ax.set_ylabel("Probabilidad de ejecución de un trader de liquidez")
    ax.set_title("Probabilidad de ejecución no informada contra el spread cotizado")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.25)

    _guardar(fig, ruta)
    return fig


def figura_pnl_acumulado(sims: dict, regs: dict, ruta="02_pnl_acumulado.png"):
    """Figura 2: P&L acumulado a lo largo de los trades, los tres regimenes en los mismos ejes."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for nombre, d in sims.items():
        cot = regs[nombre]
        ax.plot(d.index, d["pnl_acumulado"], lw=1.5,
                label=f"{nombre} (A={cot.ask:.2f}, B={cot.bid:.2f}, spread={cot.spread:.2f})")

    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Número de trade")
    ax.set_ylabel("P&L acumulado del formador")
    ax.set_title(f"P&L acumulado en {len(next(iter(sims.values()))):,} trades por régimen")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    _guardar(fig, ruta)
    return fig


def figura_inventario(sims: dict, regs: dict, ruta="03_inventario.png"):
    """Figura 3: inventario acumulado del formador, los tres regimenes."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for nombre, d in sims.items():
        ax.plot(d.index, d["inventario_acumulado"], lw=1.2,
                label=f"{nombre} (final = {d['cambio_inventario'].sum():+.0f})")

    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Número de trade")
    ax.set_ylabel("Inventario acumulado (unidades; negativo = corto)")
    ax.set_title("Inventario acumulado del formador de mercado por régimen")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)

    _guardar(fig, ruta)
    return fig


def figura_montecarlo(mc: dict, n_trades: int, ruta="04_montecarlo.png"):
    """Figura 4: histograma del P&L final de las corridas de Monte Carlo."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for nombre, r in mc.items():
        pnl = r["pnl_final"]
        ax.hist(pnl, bins=50, alpha=0.55,
                label=f"{nombre}: media {pnl.mean():,.0f}, P(pérdida) = {(pnl < 0).mean():.3f}")

    ax.axvline(0.0, color="black", lw=1.2, ls="--", label="P&L = 0")
    ax.set_xlabel(f"P&L final de la corrida ({n_trades:,} trades)")
    ax.set_ylabel("Número de corridas")
    ax.set_title(f"Distribución del P&L final — Monte Carlo de {len(next(iter(mc.values()))['pnl_final']):,} corridas")
    ax.legend()
    ax.grid(alpha=0.25)

    _guardar(fig, ruta)
    return fig


def figura_sensibilidad(sensibilidad, p: Params, ruta="05_sensibilidad_pi_i.png"):
    """Seccion 3.5: spread optimo contra pi_I, contra la referencia teorica de pi_I -> 0."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(sensibilidad.index, sensibilidad["Spread"], "o-", lw=2, color="#1f77b4",
            label="Spread óptimo (numérico)")
    ax.axhline(2 * medio_spread_monopolista(p), color="#d62728", ls="--", lw=1.5,
               label=rf"Límite teórico $\pi_I \to 0$: $2\alpha/(2\beta)$ = "
                     f"{2 * medio_spread_monopolista(p):.2f}")

    for x, y in zip(sensibilidad.index, sensibilidad["Spread"]):
        ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=9)

    ax.set_xlabel(r"Probabilidad de trader informado $\pi_I$")
    ax.set_ylabel("Spread óptimo $A - B$")
    ax.set_title("Spread óptimo contra la probabilidad de trader informado")
    ax.legend()
    ax.grid(alpha=0.25)

    _guardar(fig, ruta)
    return fig
