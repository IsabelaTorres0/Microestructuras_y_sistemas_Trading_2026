"""Validación y optimización de hiperparámetros de la estrategia.

    θ* = arg max_θ  J( backtest(data_train, θ) ),     J = Calmar = CAGR / |Drawdown máximo|

θ son los tres parámetros de salida: stop-loss, take-profit y break-even (en múltiplos de ATR).
Las reglas de entrada y las ventanas de los indicadores NO se optimizan.

Contenido:
  * objetivo(data, theta)            -> J(θ): el Calmar del backtest con θ
  * optimizar_optuna(data, ...)      -> θ* con Optuna (TPE)
  * walk_forward(data, ...)          -> validación walk-forward con forward chaining (ventana creciente)
  * random_walk(df, seed)            -> precios aleatorios con la misma forma de barra (prueba de nulidad)

Nota sobre los indicadores: se calculan UNA vez sobre toda la serie y luego se cortan los tramos.
Es válido porque todos los indicadores son causales (lo demuestra la prueba de truncamiento del
pipeline): el valor en t solo depende de datos hasta t. Así cada tramo no pierde 100 barras de
calentamiento de la SMA_100.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from .backtest import backtest
from .indicators import add_indicators
from .metrics import cagr, max_drawdown
from .strategy import Params, entry_conditions

# Espacio de búsqueda de θ (múltiplos de ATR) y paso de la rejilla
THETA_SPACE = {"sl_atr": (1.0, 6.0), "tp_atr": (1.0, 6.0), "be_atr": (0.5, 3.0)}
STEP = 0.25
PENALTY = -10.0          # J cuando el backtest tiene muy pocas operaciones para medir nada


# ---------------------------------------------------------------- θ <-> Params
def theta_to_params(theta: dict, base: Params = Params()) -> Params:
    """Convierte θ = {"sl_atr", "tp_atr", "be_atr"} en un Params completo (el resto queda igual)."""
    return replace(base, **{k: float(v) for k, v in theta.items()})


def theta_base(base: Params = Params()) -> dict:
    """θ de la Act 05 / Act 06 (SL 4, TP 3, BE 1)."""
    return {"sl_atr": base.sl_atr, "tp_atr": base.tp_atr, "be_atr": base.be_atr}


# ---------------------------------------------------------------- Función objetivo
def calmar_objetivo(res: dict, min_trades: int = 3, mdd_floor: float = 0.01) -> float:
    """Calmar para optimizar = CAGR / max(|MDD|, mdd_floor).

    * Si hay menos de `min_trades` operaciones cerradas regresa PENALTY: con 0-2 operaciones
      el Calmar no mide nada y el optimizador premiaría "no operar".
    * `mdd_floor` (1 %) evita dividir entre un drawdown casi cero, que haría explotar el Calmar
      con una o dos operaciones ganadoras.
    """
    tr = res["trades"]
    if len(tr) < min_trades:
        return PENALTY
    eq = res["equity"]["equity"]
    return float(cagr(eq) / max(abs(max_drawdown(eq)), mdd_floor))


def objetivo(data: pd.DataFrame, theta: dict, base: Params = Params(),
             min_trades: int = 3, mdd_floor: float = 0.01) -> float:
    """J(θ) = Calmar( backtest(data, θ) ).   Dos argumentos principales: los datos y θ."""
    res = backtest(data, theta_to_params(theta, base))
    return calmar_objetivo(res, min_trades, mdd_floor)


# ---------------------------------------------------------------- Optuna
def optimizar_optuna(data: pd.DataFrame, n_trials: int = 200, seed: int = 42,
                     base: Params = Params(), min_trades: int = 3, space: dict = None):
    """θ* = arg max_θ J(backtest(data, θ)) con Optuna (sampler TPE, semilla fija).

    Regresa (theta_star, J_star, study)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = space or THETA_SPACE

    def _obj(trial):
        theta = {k: trial.suggest_float(k, lo, hi, step=STEP) for k, (lo, hi) in space.items()}
        return objetivo(data, theta, base, min_trades)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.enqueue_trial(theta_base(base))        # el θ de la Act 06 siempre se evalúa
    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)
    return dict(study.best_params), float(study.best_value), study


# ---------------------------------------------------------------- Walk-forward
def forward_chaining_folds(n: int, start: int, train_min: int, test_len: int) -> list:
    """Cortes de walk-forward con forward chaining (ventana de entrenamiento creciente).

    fold k:  train = [start, start + train_min + k·test_len)
             test  = [fin del train, fin del train + test_len)
    Cada test está siempre DESPUÉS de su train y los tests no se traslapan."""
    folds, k = [], 0
    while True:
        tr_end = start + train_min + k * test_len
        te_end = tr_end + test_len
        if te_end > n:
            break
        folds.append({"fold": k + 1, "train": (start, tr_end), "test": (tr_end, te_end)})
        k += 1
    return folds


def walk_forward(data: pd.DataFrame, start: int, train_min: int, test_len: int,
                 n_trials: int = 100, seed: int = 42, base: Params = Params(),
                 min_trades: int = 3) -> dict:
    """En cada fold: θ*_k = arg max J(backtest(train_k, θ)) con Optuna y se evalúa θ*_k en test_k.
    También se evalúa el θ base en el mismo test_k para comparar.

    Regresa la tabla por fold y las curvas de equity fuera de muestra encadenadas."""
    rows, oos_opt, oos_base = [], [], []
    for f in forward_chaining_folds(len(data), start, train_min, test_len):
        tr = data.iloc[f["train"][0]:f["train"][1]]
        te = data.iloc[f["test"][0]:f["test"][1]]
        th, j_train, _ = optimizar_optuna(tr, n_trials, seed + f["fold"], base, min_trades)
        r_opt = backtest(te, theta_to_params(th, base))
        r_base = backtest(te, theta_to_params(theta_base(base), base))
        e_opt, e_base = r_opt["equity"]["equity"], r_base["equity"]["equity"]
        oos_opt.append(e_opt.pct_change().fillna(e_opt.iloc[0] / base.initial_cash - 1))
        oos_base.append(e_base.pct_change().fillna(e_base.iloc[0] / base.initial_cash - 1))
        rows.append({
            "Fold": f["fold"],
            "Train": f"{tr.index[0]:%Y-%m-%d} → {tr.index[-1]:%Y-%m-%d}",
            "Test": f"{te.index[0]:%Y-%m-%d} → {te.index[-1]:%Y-%m-%d}",
            "Barras train": len(tr), "Barras test": len(te),
            "SL*": th["sl_atr"], "TP*": th["tp_atr"], "BE*": th["be_atr"],
            "J train (θ*)": j_train,
            "Rend. test θ* (%)": 100 * (e_opt.iloc[-1] / base.initial_cash - 1),
            "Rend. test θ base (%)": 100 * (e_base.iloc[-1] / base.initial_cash - 1),
            "Ops test θ*": len(r_opt["trades"]), "Ops test θ base": len(r_base["trades"]),
            "DD máx test θ* (%)": 100 * max_drawdown(e_opt),
        })
    tabla = pd.DataFrame(rows).set_index("Fold")
    eq_opt = base.initial_cash * (1 + pd.concat(oos_opt)).cumprod()
    eq_base = base.initial_cash * (1 + pd.concat(oos_base)).cumprod()
    return {"tabla": tabla, "equity_oos_opt": eq_opt, "equity_oos_base": eq_base}


# ---------------------------------------------------------------- Random walk (prueba de nulidad)
def random_walk(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Serie de precios aleatoria con la misma "forma" de barra que los datos reales.

    Para cada día se guardan cuatro razones: gap = open/close_anterior, high/open, low/open y
    close/open. Luego se sortean días CON reemplazo (bootstrap iid) y se reconstruye la serie
    desde el mismo precio inicial. Resultado: mismos rendimientos diarios, misma volatilidad,
    mismos gaps y mismos rangos intradía, pero SIN tendencias ni momentum reales (el orden de
    los días es aleatorio). Si la estrategia gana igual en estas series, su ventaja es suerte."""
    rng = np.random.default_rng(seed)
    prev_c = df["close"].shift(1)
    rel = pd.DataFrame({"gap": df["open"] / prev_c, "h": df["high"] / df["open"],
                        "l": df["low"] / df["open"], "c": df["close"] / df["open"]}).dropna().to_numpy()
    pick = rel[rng.integers(0, len(rel), size=len(df))]
    o = np.empty(len(df)); h = np.empty(len(df)); l = np.empty(len(df)); c = np.empty(len(df))
    last = df["close"].iloc[0]
    for i, (g, hr, lr, cr) in enumerate(pick):
        o[i] = last * g; h[i] = o[i] * hr; l[i] = o[i] * lr; c[i] = o[i] * cr
        last = c[i]
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=df.index)


def preparar(df: pd.DataFrame, base: Params = Params()) -> pd.DataFrame:
    """indicadores -> señales -> combinación (la entrada que necesita backtest)."""
    ind = add_indicators(df)
    return ind.join(entry_conditions(ind, base))