"""
Factor-timing portfolios built from the rolling-Lasso predicted returns.

At each OOS date d, the mean-variance weight vector is

    w_d = Sigma_d^-1 . r_hat_d

where r_hat_d is the Lasso's predicted return for d (a forecast built only
from information available before d -- see lasso.py) and Sigma_d is the
sample covariance of realized factor returns over the same trailing
`rolling_window_periods` window the Lasso trained on, [d-window, d-1]
(also known before d, so no lookahead is introduced).

Raw weights are then scaled to a target annualized vol. Since
w_d' Sigma_d w_d = r_hat_d' Sigma_d^-1 r_hat_d for these weights, the
ex-ante portfolio variance has a closed form and the vol-targeting scale
factor follows directly -- still no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config

_RIDGE = 1e-10  # numerical safety only, not part of the methodology


def _rolling_covariance_windows(dataset: pd.DataFrame, factors: list[str], dates: pd.DatetimeIndex,
                                 window: int) -> dict:
    """Trailing `window`-period sample covariance of `factors`, for [d-window, d-1]."""
    idx = dataset.index
    cov_by_date = {}
    for d in dates:
        t = idx.get_loc(d)
        train = dataset[factors].iloc[t - window: t]
        cov_by_date[d] = train.cov().to_numpy() + _RIDGE * np.eye(len(factors))
    return cov_by_date


def _expanding_covariance_windows(dataset: pd.DataFrame, factors: list[str], dates: pd.DatetimeIndex) -> dict:
    """Expanding sample covariance of `factors`, for [start, d-1]."""
    idx = dataset.index
    cov_by_date = {}
    for d in dates:
        t = idx.get_loc(d)
        train = dataset[factors].iloc[:t]
        cov_by_date[d] = train.cov().to_numpy() + _RIDGE * np.eye(len(factors))
    return cov_by_date


def _vol_targeted_weights(predictions: pd.DataFrame, factors: list[str], cov_by_date: dict, cfg: Config) -> pd.DataFrame:
    """Vol-targeted mean-variance weights w_d = Sigma_d^-1 . r_hat_d, one row per date in `cov_by_date`.

    `predictions` is the wide multi-index frame from a *_predictions.csv
    (columns are (factor, 'actual'/'predicted'[/...])).
    """
    dates = list(cov_by_date.keys())
    r_hat = pd.concat({f: predictions[(f, "predicted")] for f in factors}, axis=1)[factors]
    target_vol_period = cfg.target_vol_annual / np.sqrt(cfg.periods_per_year)

    rows = []
    for d in dates:
        r_hat_d = r_hat.loc[d].to_numpy()
        sigma_d = cov_by_date[d]
        raw_w = np.linalg.solve(sigma_d, r_hat_d)
        ex_ante_var = float(raw_w @ sigma_d @ raw_w)
        if ex_ante_var < 1e-14:
            w = np.zeros_like(raw_w)
        else:
            w = raw_w * (target_vol_period / np.sqrt(ex_ante_var))
        rows.append(w)

    return pd.DataFrame(rows, index=dates, columns=factors)


def build_timing_weights(dataset: pd.DataFrame, predictions: pd.DataFrame, factors: list[str],
                          cfg: Config) -> pd.DataFrame:
    """Weights with Sigma_d from the trailing `rolling_window_periods` window (used for the
    rolling-Lasso portfolio, matching the window the Lasso itself trained on)."""
    cov_by_date = _rolling_covariance_windows(dataset, factors, predictions.index, cfg.rolling_window_periods)
    return _vol_targeted_weights(predictions, factors, cov_by_date, cfg)


def build_expanding_timing_weights(dataset: pd.DataFrame, predictions: pd.DataFrame, factors: list[str],
                                    cfg: Config) -> pd.DataFrame:
    """Weights with Sigma_d from the expanding window [start, d-1] (used for the Melone
    static/Kalman portfolios, matching the expanding window those methods re-estimate on)."""
    cov_by_date = _expanding_covariance_windows(dataset, factors, predictions.index)
    return _vol_targeted_weights(predictions, factors, cov_by_date, cfg)


def equal_weight_benchmark(dates: pd.DatetimeIndex, factors: list[str]) -> pd.DataFrame:
    """Static 1/N weights on `factors`, rebalanced back to equal weight every period."""
    w = np.full(len(factors), 1.0 / len(factors))
    return pd.DataFrame(np.tile(w, (len(dates), 1)), index=dates, columns=factors)


def sharpe_ratio(returns: pd.Series, cfg: Config) -> float:
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(cfg.periods_per_year))


def certainty_equivalent_return(returns: pd.Series, gamma: float, cfg: Config) -> float:
    """Annualized CER = mean(ann.) - gamma/2 * var(ann.)."""
    mean_ann = returns.mean() * cfg.periods_per_year
    var_ann = returns.var(ddof=1) * cfg.periods_per_year
    return float(mean_ann - 0.5 * gamma * var_ann)


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown_series(returns).min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod() - 1


def period_turnover(weights: pd.DataFrame, actual_returns: pd.DataFrame, port_returns: pd.Series) -> pd.Series:
    """Mean absolute weight change per period, drift-adjusting last period's
    weights for the return they earned before comparing them to this
    period's rebalanced weights."""
    drifted = weights.shift(1) * (1 + actual_returns.shift(1))
    drifted = drifted.div(1 + port_returns.shift(1), axis=0)
    return (weights - drifted).abs().sum(axis=1).dropna()


def summarize_portfolio(name: str, weights: pd.DataFrame, dataset: pd.DataFrame, cfg: Config) -> dict:
    actual = dataset.loc[weights.index, weights.columns]
    returns = (weights * actual).sum(axis=1).rename(name)
    turnover = period_turnover(weights, actual, returns)
    metrics = {
        "Portfolio": name,
        "Sharpe (ann.)": sharpe_ratio(returns, cfg),
        "CER (%, ann., gamma=5)": certainty_equivalent_return(returns, cfg.cer_gamma, cfg) * 100,
        "Max drawdown (%)": max_drawdown(returns) * 100,
        "Avg quarterly turnover": float(turnover.mean()),
    }
    return {"returns": returns, "weights": weights, "turnover": turnover, "metrics": metrics}
