"""
TSMOM signal generation and position sizing (Moskowitz, Ooi & Pedersen 2012, Eq. 5).

    r^{TSMOM,i}_{t,t+1} = sign(r^i_{t-12,t}) * (40% / sigma^i_t) * r^i_{t,t+1}

All quantities are computed at month-end rebalance dates. The signal and
the volatility scalar used to size the position at date t are both built
exclusively from information available through t, so the only unknown
quantity is the realized return over the following month, r_{t,t+1}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config


def month_end_dates(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available trading day of each calendar month in the index."""
    s = pd.Series(daily_index, index=daily_index)
    return s.groupby([daily_index.year, daily_index.month]).max().sort_values().values


def compute_monthly_excess_returns(excess_returns: pd.DataFrame, m_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Compound daily excess returns into monthly excess returns.

    The value at index t is the compounded return over the period
    (m_ends[k-1], m_ends[k]], i.e. the return *ending* at t.

    Grouped directly by (year, month) rather than via `resample`, since
    `resample` labels bins with the calendar month-end date, which does not
    always coincide with the last *trading* day in `m_ends` (e.g. when the
    calendar month-end falls on a weekend/holiday) -- that mismatch would
    silently drop the reindex to all-NaN.
    """
    def _compound(x: pd.DataFrame) -> pd.Series:
        return (1.0 + x).prod(min_count=1) - 1.0

    idx = excess_returns.index
    monthly = excess_returns.groupby([idx.year, idx.month]).apply(_compound)
    monthly.index = pd.DatetimeIndex(m_ends)
    return monthly


def compute_past_12m_return(monthly_excess_returns: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Trailing `lookback`-month compounded cumulative return, ending at t."""
    def _cum(x: np.ndarray) -> float:
        if np.any(np.isnan(x)):
            return np.nan
        return np.prod(1.0 + x) - 1.0

    return monthly_excess_returns.rolling(lookback, min_periods=lookback).apply(_cum, raw=True)


def compute_sigma_at_month_end(sigma_daily_lagged: pd.DataFrame, m_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Sample the (already 1-day-lagged) daily ex-ante vol at month-end dates."""
    return sigma_daily_lagged.reindex(pd.DatetimeIndex(m_ends), method="ffill")


def build_tsmom_signals(
    excess_returns_daily: pd.DataFrame,
    sigma_daily_lagged: pd.DataFrame,
    cfg: Config,
    long_only: bool = False,
) -> dict:
    """Construct TSMOM signals, position sizes and instrument-level returns.

    If `long_only` is True, the sign signal is replaced by a constant +1
    (still requiring 12m of history to exist, to keep the investable
    universe identical to the TSMOM strategy) -- this produces the
    "passive long" benchmark with the same volatility scaling as TSMOM.

    Returns a dict with:
      - monthly_excess_returns: monthly excess return ending at each date
      - past_12m_return: trailing cumulative excess return as of each date
      - signal: sign(past_12m_return), or +1 everywhere if long_only
      - sigma: ex-ante annualized vol sampled at month-end (no look-ahead)
      - position_size: vol_target / sigma
      - instrument_returns: realized TSMOM return per instrument, indexed
        by the date the return is *realized* (end of holding month), so it
        is directly comparable to monthly_excess_returns.
    """
    m_ends = month_end_dates(excess_returns_daily.index)

    monthly_excess_returns = compute_monthly_excess_returns(excess_returns_daily, m_ends)
    past_12m_return = compute_past_12m_return(monthly_excess_returns, cfg.lookback_months)
    if long_only:
        signal = past_12m_return.where(past_12m_return.isna(), 1.0)
    else:
        signal = np.sign(past_12m_return)

    sigma = compute_sigma_at_month_end(sigma_daily_lagged, m_ends)
    position_size = cfg.vol_target / sigma
    position_size = position_size.replace([np.inf, -np.inf], np.nan)
    if cfg.max_position_size is not None:
        position_size = position_size.clip(upper=cfg.max_position_size)

    # forward_return.loc[t] = r_{t, t+1}, the return earned holding from t to t+1
    forward_return = monthly_excess_returns.shift(-1)

    raw_strategy_return = signal * position_size * forward_return

    # Re-index so the return sits on the date it is realized (t+1), matching
    # the convention used by monthly_excess_returns (return ending at index).
    instrument_returns = raw_strategy_return.shift(1)

    return {
        "monthly_excess_returns": monthly_excess_returns,
        "past_12m_return": past_12m_return,
        "signal": signal,
        "sigma": sigma,
        "position_size": position_size,
        "instrument_returns": instrument_returns,
    }


def build_xsmom_signals(
    excess_returns_daily: pd.DataFrame,
    sigma_daily_lagged: pd.DataFrame,
    cfg: Config,
) -> dict:
    """Cross-sectional momentum: rank by past 12m return, long top half /
    short bottom half, vol-scaled, equal-weighted within each leg.
    """
    m_ends = month_end_dates(excess_returns_daily.index)

    monthly_excess_returns = compute_monthly_excess_returns(excess_returns_daily, m_ends)
    past_12m_return = compute_past_12m_return(monthly_excess_returns, cfg.xsmom_lookback_months)

    sigma = compute_sigma_at_month_end(sigma_daily_lagged, m_ends)
    position_size = cfg.vol_target / sigma
    position_size = position_size.replace([np.inf, -np.inf], np.nan)
    if cfg.max_position_size is not None:
        position_size = position_size.clip(upper=cfg.max_position_size)

    forward_return = monthly_excess_returns.shift(-1)

    def _xs_weights(row: pd.Series) -> pd.Series:
        valid = row.dropna()
        n = len(valid)
        weights = pd.Series(np.nan, index=row.index)
        if n < 4:  # need at least a couple of names on each side
            return weights
        ranks = valid.rank(method="first")
        half = n / 2.0
        long_mask = ranks > (n - np.floor(n / 2))
        short_mask = ranks <= np.floor(n / 2)
        n_long = long_mask.sum()
        n_short = short_mask.sum()
        w = pd.Series(0.0, index=valid.index)
        w[long_mask] = 1.0 / n_long
        w[short_mask] = -1.0 / n_short
        weights.loc[valid.index] = w
        return weights

    xs_weights = past_12m_return.apply(_xs_weights, axis=1)

    raw_strategy_return = xs_weights * position_size * forward_return
    instrument_returns = raw_strategy_return.shift(1)

    return {
        "monthly_excess_returns": monthly_excess_returns,
        "past_12m_return": past_12m_return,
        "weights": xs_weights,
        "sigma": sigma,
        "position_size": position_size,
        "instrument_returns": instrument_returns,
    }
