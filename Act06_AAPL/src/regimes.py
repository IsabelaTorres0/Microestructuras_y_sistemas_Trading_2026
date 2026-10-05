"""Act 07: regímenes de mercado (Crisis, Tendencia, Mean reverting).

Matriz de features (todas causales: el valor en t solo usa datos hasta t):
  * vol   = desviación estándar de los rendimientos log de `vol_win` días       -> nivel de volatilidad
  * trend = (SMA_fast - SMA_slow) / (close * vol)                                -> fuerza de tendencia
            (diferencia de medias medida en "días de volatilidad", con signo)
  * ac1   = autocorrelación lag 1 de los rendimientos en `ac_win` días          -> fuerza de mean reverting
            (negativa = el precio tiende a regresarse)

Tres clasificadores, los tres AJUSTADOS SOLO CON TRAIN:
  * Reglas   (umbrales = cuantiles de train)
  * K-means  (k = 3, features estandarizadas con media y desviación de train)
  * HMM      (cadena de Markov oculta gaussiana de 3 estados)

Etiqueta FILTRADA = la que se puede conocer en t con datos hasta t (es la que se opera).
Etiqueta VITERBI  = la secuencia más probable usando TODA la serie (mira al futuro: solo sirve
                    para comparar, no para operar).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import backtest
from .optimize import PENALTY, objetivo, optimizar_optuna, theta_to_params
from .strategy import Params, check_exit, open_long, update_break_even

FEATURES = ["vol", "trend", "ac1"]
REGIMENES = ["Crisis", "Tendencia", "Mean reverting"]
BARS_PER_MONTH = 21


# ====================================================================== 1. Features
def regime_features(df: pd.DataFrame, vol_win: int = 20, fast: int = 10, slow: int = 100,
                    ac_win: int = 60) -> pd.DataFrame:
    """Matriz de features de régimen. Solo usa rolling hacia atrás -> causal."""
    close = df["close"]
    ret = np.log(close).diff()
    vol = ret.rolling(vol_win, min_periods=vol_win).std()
    sma_f = close.rolling(fast, min_periods=fast).mean()
    sma_s = close.rolling(slow, min_periods=slow).mean()
    trend = (sma_f - sma_s) / (close * vol)
    ac1 = ret.rolling(ac_win, min_periods=ac_win).corr(ret.shift(1))
    return pd.DataFrame({"vol": vol, "trend": trend, "ac1": ac1}, index=df.index)


@dataclass
class Scaler:
    """Estandarización con media y desviación de TRAIN (no se recalcula con datos nuevos)."""
    mean: pd.Series
    std: pd.Series

    def transform(self, feats: pd.DataFrame) -> pd.DataFrame:
        return (feats[FEATURES] - self.mean) / self.std


def fit_scaler(feats_train: pd.DataFrame) -> Scaler:
    f = feats_train[FEATURES].dropna()
    return Scaler(f.mean(), f.std(ddof=0))


# ====================================================================== 2a. Reglas
def fit_rules(feats_train: pd.DataFrame, q_vol: float = 0.80, q_trend: float = 0.50,
              q_ac: float = 0.25) -> dict:
    """Umbrales de las reglas = cuantiles de train."""
    f = feats_train[FEATURES].dropna()
    return {"vol_alta": float(f["vol"].quantile(q_vol)),
            "trend_fuerte": float(f["trend"].abs().quantile(q_trend)),
            "ac_baja": float(f["ac1"].quantile(q_ac)),
            "q_vol": q_vol, "q_trend": q_trend, "q_ac": q_ac}


def predict_rules(feats: pd.DataFrame, th: dict) -> pd.Series:
    """Reglas en orden de prioridad (cada barra recibe exactamente una etiqueta):

    1. Crisis          si vol > vol_alta                                 (la volatilidad manda)
    2. Tendencia       si no es crisis, |trend| >= trend_fuerte y ac1 > ac_baja
    3. Mean reverting  en cualquier otro caso (tendencia débil o autocorrelación muy negativa)
    """
    crisis = feats["vol"] > th["vol_alta"]
    tendencia = (~crisis) & (feats["trend"].abs() >= th["trend_fuerte"]) & (feats["ac1"] > th["ac_baja"])
    lab = pd.Series("Mean reverting", index=feats.index, dtype=object)
    lab[tendencia] = "Tendencia"
    lab[crisis] = "Crisis"
    lab[feats[FEATURES].isna().any(axis=1)] = np.nan
    return lab


# ====================================================================== nombres de estados
def name_states(centers: pd.DataFrame) -> dict:
    """Nombra los 3 estados a partir de sus centroides (features estandarizadas):

      * Crisis          = el estado con el centroide de volatilidad más alto
      * Tendencia       = de los dos restantes, el de mayor |trend|
      * Mean reverting  = el que queda
    """
    c = centers.copy()
    crisis = c["vol"].idxmax()
    rest = c.drop(index=crisis)
    tendencia = rest["trend"].abs().idxmax()
    mr = [i for i in rest.index if i != tendencia][0]
    return {crisis: "Crisis", tendencia: "Tendencia", mr: "Mean reverting"}


# ====================================================================== 2b. K-means
def elbow(z_train: pd.DataFrame, k_max: int = 8, seed: int = 42) -> pd.DataFrame:
    """Regla del codo: inercia (suma de distancias al cuadrado) y silhouette para k = 1..k_max."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    X = z_train.dropna().to_numpy()
    rows = []
    for k in range(1, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append({"k": k, "Inercia": km.inertia_,
                     "Silhouette": silhouette_score(X, km.labels_) if k > 1 else np.nan})
    out = pd.DataFrame(rows).set_index("k")
    out["Caída de inercia (%)"] = -100 * out["Inercia"].pct_change()
    return out


def elbow_point(inercia: pd.Series) -> int:
    """Codo geométrico: el k más alejado de la recta que une el primer y el último punto."""
    k = inercia.index.to_numpy(dtype=float); y = inercia.to_numpy(dtype=float)
    kn = (k - k[0]) / (k[-1] - k[0]); yn = (y - y[-1]) / (y[0] - y[-1])
    dist = np.abs(kn + yn - 1) / np.sqrt(2)
    return int(inercia.index[int(np.argmax(dist))])


def fit_kmeans(z_train: pd.DataFrame, k: int = 3, seed: int = 42):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z_train.dropna().to_numpy())
    centers = pd.DataFrame(km.cluster_centers_, columns=FEATURES)
    return {"model": km, "centers": centers, "names": name_states(centers)}


def predict_kmeans(z: pd.DataFrame, fit: dict) -> pd.Series:
    """Asigna cada barra al centroide más cercano. Es causal: los centroides son de train y la
    barra t solo usa sus propias features."""
    ok = z[FEATURES].notna().all(axis=1)
    lab = pd.Series(np.nan, index=z.index, dtype=object)
    lab[ok] = pd.Series(fit["model"].predict(z.loc[ok, FEATURES].to_numpy()), index=z.index[ok]).map(fit["names"])
    return lab


# ====================================================================== 2c. HMM
def fit_hmm(z_train: pd.DataFrame, k: int = 3, seed: int = 42, n_init: int = 5):
    """HMM gaussiano de k estados. Se prueban varias semillas y se queda el de mayor verosimilitud."""
    from hmmlearn.hmm import GaussianHMM
    X = z_train.dropna().to_numpy()
    best, best_ll = None, -np.inf
    for s in range(n_init):
        m = GaussianHMM(n_components=k, covariance_type="full", n_iter=300, tol=1e-4,
                        random_state=seed + s, min_covar=1e-3)
        try:
            m.fit(X); ll = m.score(X)
        except Exception:
            continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = m, ll
    if best is None:
        raise RuntimeError("El HMM no convergió con ninguna semilla")
    centers = pd.DataFrame(best.means_, columns=FEATURES)
    return {"model": best, "centers": centers, "names": name_states(centers), "loglik": best_ll}


def _log_emission(model, X: np.ndarray) -> np.ndarray:
    """log p(x_t | estado) para cada barra y estado (gaussiana multivariada)."""
    from scipy.stats import multivariate_normal
    return np.column_stack([multivariate_normal.logpdf(X, mean=model.means_[j], cov=model.covars_[j],
                                                       allow_singular=True)
                            for j in range(model.n_components)])


def hmm_filter(model, X: np.ndarray) -> np.ndarray:
    """Probabilidad FILTRADA P(s_t | x_1..x_t) con el algoritmo forward.

    Es la versión causal: la probabilidad en t no cambia al agregar datos posteriores."""
    logb = _log_emission(model, X)
    logA = np.log(model.transmat_ + 1e-300)
    T, K = logb.shape
    out = np.empty((T, K))
    a = np.log(model.startprob_ + 1e-300) + logb[0]
    a -= np.logaddexp.reduce(a)
    out[0] = a
    for t in range(1, T):
        a = logb[t] + np.logaddexp.reduce(a[:, None] + logA, axis=0)
        a -= np.logaddexp.reduce(a)
        out[t] = a
    return np.exp(out)


def predict_hmm(z: pd.DataFrame, fit: dict) -> pd.DataFrame:
    """Regresa la etiqueta filtrada (causal) y la de Viterbi (usa toda la serie)."""
    ok = z[FEATURES].notna().all(axis=1)
    X = z.loc[ok, FEATURES].to_numpy()
    filt = pd.Series(hmm_filter(fit["model"], X).argmax(axis=1), index=z.index[ok]).map(fit["names"])
    vit = pd.Series(fit["model"].predict(X), index=z.index[ok]).map(fit["names"])
    out = pd.DataFrame(index=z.index, columns=["filtrada", "viterbi"], dtype=object)
    out.loc[ok, "filtrada"] = filt; out.loc[ok, "viterbi"] = vit
    return out


# ====================================================================== 3. Comparación
def regime_stats(labels: pd.Series, z: pd.DataFrame = None) -> dict:
    """Duración media, transiciones por mes, participación de cada régimen y silhouette."""
    lab = labels.dropna()
    cambio = lab != lab.shift(1)
    n_runs = int(cambio.sum())
    trans = n_runs - 1
    out = {"Duración media (barras)": len(lab) / n_runs if n_runs else np.nan,
           "Transiciones por mes": trans / (len(lab) / BARS_PER_MONTH) if len(lab) else np.nan}
    for r in REGIMENES:
        out[f"% {r}"] = 100 * float((lab == r).mean())
    if z is not None:
        from sklearn.metrics import silhouette_score
        zz = z.loc[lab.index, FEATURES].dropna()
        ll = lab.loc[zz.index]
        out["Silhouette"] = silhouette_score(zz.to_numpy(), ll.to_numpy()) if ll.nunique() > 1 else np.nan
    return out


# ====================================================================== 7. Operar los regímenes
def backtest_regimes(data: pd.DataFrame, params_by_regime: dict, label_col: str = "etiqueta",
                     default: Params = Params()) -> dict:
    """Mismo motor orientado a eventos que backtest(), pero los parámetros de salida de cada
    operación se eligen con el régimen de la barra de la SEÑAL (etiqueta filtrada en t):

        θ usado = params_by_regime[ etiqueta(t) ]

    Si el valor del diccionario es None, en ese régimen no se abren operaciones.
    Costos, rho y capital se toman de `default` (son los mismos en todos los regímenes).
    Función pura: no modifica `data`."""
    fee = default.commission_bps / 1e4
    slip = default.slippage_bps / 1e4
    o = data["open"].to_numpy(); h = data["high"].to_numpy()
    l = data["low"].to_numpy(); c = data["close"].to_numpy()
    atr_v = data["atr_14"].to_numpy(); trig = data["entry_trigger"].to_numpy()
    lab = data[label_col].to_numpy()
    idx = data.index

    cash = default.initial_cash
    pos, p_pos, lab_pos = None, None, None
    pending = None                       # Params de la orden pendiente (o None)
    pending_lab = None
    equity_prev = default.initial_cash
    rows, trades = [], []

    for i in range(len(data)):
        t = idx[i]
        entered_now = False
        if pending is not None and pos is None:
            fill = o[i] * (1 + slip)
            p_pos, lab_pos = pending, pending_lab
            pos = open_long(t, fill, atr_v[i - 1], equity_prev, cash, p_pos)
            cash -= pos.units * fill + pos.entry_fee
            entered_now = True
        pending = None

        if pos is not None:
            ex = check_exit(pos, o[i], h[i], l[i], is_entry_bar=entered_now)
            if ex is not None:
                reason, level = ex
                fill = level * (1 - slip)
                exit_fee = pos.units * fill * fee
                cash += pos.units * fill - exit_fee
                pnl = pos.units * (fill - pos.entry_price) - pos.entry_fee - exit_fee
                trades.append({
                    "entrada": pos.entry_time, "salida": t, "regimen": lab_pos,
                    "precio_entrada": pos.entry_price, "precio_salida": fill,
                    "stop_inicial": pos.entry_price - p_pos.sl_atr * pos.atr_at_entry,
                    "take_profit": pos.take_profit, "unidades": pos.units,
                    "nocional": pos.units * pos.entry_price,
                    "costos": pos.entry_fee + exit_fee, "pnl": pnl,
                    "retorno_%": 100 * pnl / (pos.units * pos.entry_price),
                    "R": pnl / (pos.units * p_pos.sl_atr * pos.atr_at_entry),
                    "motivo": reason, "barras": idx.get_loc(t) - idx.get_loc(pos.entry_time) + 1,
                })
                pos = None
            else:
                update_break_even(pos, h[i], p_pos)

        units = pos.units if pos is not None else 0.0
        equity = cash + units * c[i]
        rows.append((t, cash, units, equity, pos is not None))
        equity_prev = equity

        if pos is None and bool(trig[i]):
            p_sig = params_by_regime.get(lab[i], None) if isinstance(lab[i], str) else None
            if p_sig is not None:
                pending, pending_lab = p_sig, lab[i]

    eq = pd.DataFrame(rows, columns=["datetime", "cash", "units", "equity", "en_posicion"]).set_index("datetime")
    return {"equity": eq, "trades": pd.DataFrame(trades), "open_position": None if pos is None else {"entrada": pos.entry_time},
            "params": default}


def gate_by_regime(data: pd.DataFrame, regime: str, label_col: str = "etiqueta") -> pd.DataFrame:
    """Copia de `data` donde solo se permiten entradas cuando la etiqueta filtrada es `regime`."""
    out = data.copy()
    out["entry_trigger"] = out["entry_trigger"].astype(bool) & (out[label_col] == regime)
    return out


def optimize_by_regime(train: pd.DataFrame, theta_unico: dict, label_col: str = "etiqueta",
                       n_trials: int = 150, seed: int = 42, base: Params = Params(),
                       min_trades: int = 2) -> pd.DataFrame:
    """θ*_j = arg max_θ J( backtest(train | s_t = j, θ) ) para cada régimen j.

    Decisión por régimen (solo con train):
      * menos de `min_trades` señales en ese régimen -> no hay datos: se usa el θ* único
      * J*_j <= 0                                     -> el régimen pierde aun optimizado: no se opera
      * el θ* único da mejor J que el de Optuna       -> se conserva el θ* único
      * en otro caso                                  -> se usa θ*_j
    """
    rows = []
    for k, reg in enumerate(REGIMENES):
        d_j = gate_by_regime(train, reg, label_col)
        n_sig = int(d_j["entry_trigger"].sum())
        j_unico = objetivo(d_j, theta_unico, base, min_trades) if n_sig else PENALTY
        if n_sig < min_trades:
            th, j_star, decision = dict(theta_unico), np.nan, "θ* único (sin señales suficientes)"
        else:
            th, j_star, _ = optimizar_optuna(d_j, n_trials, seed + k, base, min_trades)
            if j_star <= PENALTY:
                th, j_star, decision = dict(theta_unico), np.nan, "θ* único (sin operaciones suficientes)"
            elif j_unico > j_star:              # Optuna no superó al θ* único: se conserva el único
                th, j_star, decision = dict(theta_unico), j_unico, "θ* único (ya era el mejor en este régimen)"
                if j_star <= 0:
                    decision = "No operar (J* <= 0)"
            elif j_star <= 0:
                decision = "No operar (J* <= 0)"
            else:
                decision = "θ* del régimen"
        n_ops = len(backtest(d_j, theta_to_params(th, base))["trades"])
        rows.append({"Régimen": reg, "Barras en train": int((train[label_col] == reg).sum()),
                     "Señales en train": n_sig, "sl_atr": th["sl_atr"], "tp_atr": th["tp_atr"], "be_atr": th["be_atr"],
                     "J* (Calmar)": j_star, "J con θ* único": j_unico if j_unico > PENALTY else np.nan,
                     "Operaciones con θ elegido": n_ops, "Decisión": decision})
    return pd.DataFrame(rows).set_index("Régimen")


def params_from_table(tabla: pd.DataFrame, base: Params = Params()) -> dict:
    """Convierte la tabla de optimize_by_regime en el diccionario que usa backtest_regimes."""
    out = {}
    for reg, r in tabla.iterrows():
        out[reg] = None if r["Decisión"].startswith("No operar") else theta_to_params(
            {"sl_atr": r["sl_atr"], "tp_atr": r["tp_atr"], "be_atr": r["be_atr"]}, base)
    return out
