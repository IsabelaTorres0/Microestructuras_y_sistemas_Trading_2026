import numpy as np
import pandas as pd

def _future_move(df, pos, lookahead):
    price_now = df["close"].iloc[pos]
    price_future = df["close"].iloc[pos + lookahead]
    return (price_future - price_now) / price_now * 100

def find_crossover_failures(df, fast, slow, lookahead=288, min_move_pct=0.5):
    """Cruces entre dos líneas (ej. sma_10/sma_100 o macd/macd_signal)
    donde el precio se movió en contra de lo que el cruce sugería."""
    diff = df[fast] - df[slow]
    prev_sign, curr_sign = np.sign(diff.shift(1)), np.sign(diff)
    cross_up = (prev_sign < 0) & (curr_sign > 0)   # señal alcista
    cross_down = (prev_sign > 0) & (curr_sign < 0) # señal bajista

    n = len(df)
    rows = []
    for pos in np.where(cross_up.values)[0]:
        if pos + lookahead >= n: continue
        move = _future_move(df, pos, lookahead)
        if move < -min_move_pct:
            rows.append({"date": df.index[pos], "signal": f"{fast}_cruza_arriba_de_{slow}", "pct_move_after": move})
    for pos in np.where(cross_down.values)[0]:
        if pos + lookahead >= n: continue
        move = _future_move(df, pos, lookahead)
        if move > min_move_pct:
            rows.append({"date": df.index[pos], "signal": f"{fast}_cruza_abajo_de_{slow}", "pct_move_after": move})
    return pd.DataFrame(rows).sort_values("pct_move_after", key=abs, ascending=False)

def find_rsi_failures(df, lookahead=288, min_move_pct=0.5):
    """RSI entra a sobrecompra/sobreventa pero el precio sigue igual en vez de revertir."""
    rsi = df["rsi_14"]
    overbought = (rsi > 70) & (rsi.shift(1) <= 70)
    oversold = (rsi < 30) & (rsi.shift(1) >= 30)

    n = len(df)
    rows = []
    for pos in np.where(overbought.values)[0]:
        if pos + lookahead >= n: continue
        move = _future_move(df, pos, lookahead)
        if move > min_move_pct:
            rows.append({"date": df.index[pos], "signal": "rsi_sobrecompra", "pct_move_after": move})
    for pos in np.where(oversold.values)[0]:
        if pos + lookahead >= n: continue
        move = _future_move(df, pos, lookahead)
        if move < -min_move_pct:
            rows.append({"date": df.index[pos], "signal": "rsi_sobreventa", "pct_move_after": move})
    return pd.DataFrame(rows).sort_values("pct_move_after", key=abs, ascending=False)

def find_bollinger_failures(df, lookahead=288, min_move_pct=0.5):
    """Precio toca banda alta/baja pero en vez de revertir, sigue caminando la banda."""
    touch_high = (df["close"] >= df["bb_high"]) & (df["close"].shift(1) < df["bb_high"].shift(1))
    touch_low = (df["close"] <= df["bb_low"]) & (df["close"].shift(1) > df["bb_low"].shift(1))

    n = len(df)
    rows = []
    for pos in np.where(touch_high.values)[0]:
        if pos + lookahead >= n: continue
        move = _future_move(df, pos, lookahead)
        if move > min_move_pct:
            rows.append({"date": df.index[pos], "signal": "bb_rompe_banda_alta", "pct_move_after": move})
    for pos in np.where(touch_low.values)[0]:
        if pos + lookahead >= n: continue
        move = _future_move(df, pos, lookahead)
        if move < -min_move_pct:
            rows.append({"date": df.index[pos], "signal": "bb_rompe_banda_baja", "pct_move_after": move})
    return pd.DataFrame(rows).sort_values("pct_move_after", key=abs, ascending=False)
    