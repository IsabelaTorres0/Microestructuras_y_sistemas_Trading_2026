"""
Calibrate a Geometric Brownian Motion (GBM) from realized 1-minute log
returns and simulate terminal-price paths.

GBM: dS_t = mu * S_t dt + sigma * S_t dW_t
  => S_T = S_0 * exp( (mu - sigma^2/2) T + sigma * sqrt(T) * Z ),  Z ~ N(0,1)

Identities being verified:
  E[S_T]      = S_0 * exp(mu * T)
  median(S_T) = S_0 * exp( (mu - sigma^2/2) * T )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GBMParams:
    mu: float      # annualized drift (so that E[S_T] = S0 * exp(mu*T))
    sigma: float   # annualized volatility
    dt: float      # length of one simulation step, in years


def realized_log_returns(close: pd.Series) -> pd.Series:
    """1-bar log returns from a series of close prices."""
    return np.log(close / close.shift(1)).dropna().rename("log_return")


def calibrate_gbm(
    log_returns: pd.Series,
    bars_per_year: float,
) -> GBMParams:
    """
    Calibrate annualized (mu, sigma) from a series of per-bar log returns.

    bars_per_year: number of bars of this frequency in a trading year,
        e.g. 390 (minutes/day) * 252 (days/year) for 1-minute bars.
    """
    if len(log_returns) < 2:
        raise ValueError("Need at least 2 log returns to calibrate GBM.")

    dt = 1.0 / bars_per_year
    per_bar_mean = float(log_returns.mean())
    per_bar_var = float(log_returns.var(ddof=1))

    sigma = np.sqrt(per_bar_var / dt)
    # per_bar_mean = (mu - sigma^2/2) * dt  =>  mu = per_bar_mean/dt + sigma^2/2
    mu = per_bar_mean / dt + 0.5 * sigma ** 2

    return GBMParams(mu=mu, sigma=sigma, dt=dt)


def simulate_gbm_paths(
    s0: float,
    params: GBMParams,
    n_steps: int,
    n_paths: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate `n_paths` GBM paths over `n_steps` steps of length params.dt.
    Returns an array of shape (n_paths, n_steps + 1) including S_0 at t=0.
    """
    rng = np.random.default_rng(seed)
    dt = params.dt
    drift_term = (params.mu - 0.5 * params.sigma ** 2) * dt
    diffusion_scale = params.sigma * np.sqrt(dt)

    z = rng.standard_normal(size=(n_paths, n_steps))
    log_increments = drift_term + diffusion_scale * z
    log_paths = np.cumsum(log_increments, axis=1)

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = s0
    paths[:, 1:] = s0 * np.exp(log_paths)
    return paths


def verify_mean_median_identity(
    paths: np.ndarray,
    s0: float,
    params: GBMParams,
) -> Dict[str, float]:
    """
    Compare simulated terminal mean/median of S_T against the closed-form
    GBM identities. Returns a dict with theoretical vs simulated values and
    relative errors.
    """
    n_steps = paths.shape[1] - 1
    T = n_steps * params.dt
    terminal = paths[:, -1]

    theo_mean = s0 * np.exp(params.mu * T)
    theo_median = s0 * np.exp((params.mu - 0.5 * params.sigma ** 2) * T)

    sim_mean = float(terminal.mean())
    sim_median = float(np.median(terminal))

    return {
        "T_years": T,
        "theoretical_mean": theo_mean,
        "simulated_mean": sim_mean,
        "mean_rel_error": abs(sim_mean - theo_mean) / theo_mean,
        "theoretical_median": theo_median,
        "simulated_median": sim_median,
        "median_rel_error": abs(sim_median - theo_median) / theo_median,
    }
