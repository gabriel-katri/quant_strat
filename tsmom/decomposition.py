"""
TSMOM vs. cross-sectional momentum (XSMOM) decomposition
(Moskowitz, Ooi & Pedersen 2012, Eq. 6-7).

The paper shows that the expected return of a time-series momentum
strategy can be decomposed into three sources of predictability across the
universe of vol-normalized returns x_{i,t} = r_{i,t} / sigma_{i,t-1}:

  1. Auto-covariance component  -- Cov(x_{i,t-1}, x_{i,t}), averaged over i.
     This is each instrument's own serial correlation: the part that a
     time-series strategy captures but a cross-sectional (relative-value)
     strategy does not.
  2. Cross-serial component     -- Cov(x_{i,t-1}, x_{j,t}) for i != j,
     averaged over all ordered pairs. This is the lead-lag effect across
     different instruments, which is what a cross-sectional momentum
     strategy is positioned to exploit (long past winners, short past
     losers, regardless of the *sign* of the market's own past return).
  3. Mean component             -- E[x_{i,t}]^2, averaged over i. This
     captures a persistent, unconditional drift in an instrument's return
     (a "mean" or risk-premium effect): a strategy that is always long that
     instrument benefits from it whether or not there's genuine serial
     dependence.

TSMOM (which always takes a directional bet based on each instrument's own
past sign) loads on components (1) and (3). XSMOM (long winners / short
losers *relative to the cross-section*, net dollar-neutral) loads on
component (2) and largely cancels (3), since it is a relative-value,
zero-net strategy across instruments.

This module estimates the three components empirically from the observed
panel of vol-normalized returns and reports which one dominates -- in the
original paper, the auto-covariance component (1) is by far the largest,
which is the main empirical justification for why TSMOM outperforms XSMOM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
import signals as sig_mod
import backtest as bt


def compute_normalized_returns(
    excess_returns_daily: pd.DataFrame,
    sigma_daily_lagged: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Vol-normalized monthly returns x_{i,t} = r_{i,t,t+1} / sigma_{i,t}.

    Indexed by the realization date (end of the holding month), matching
    the convention used throughout backtest.py / signals.py.
    """
    m_ends = sig_mod.month_end_dates(excess_returns_daily.index)
    monthly_excess_returns = sig_mod.compute_monthly_excess_returns(excess_returns_daily, m_ends)
    sigma = sig_mod.compute_sigma_at_month_end(sigma_daily_lagged, m_ends)

    forward_return = monthly_excess_returns.shift(-1)
    x_t = forward_return / sigma
    return x_t.shift(1)


def decompose_tsmom(normalized_returns: pd.DataFrame) -> dict:
    """Estimate the auto-covariance, cross-serial and mean components.

    Returns raw (monthly) and annualized (x12) component estimates, plus
    the full lag-1 cross-covariance matrix for inspection.
    """
    x = normalized_returns
    x_lag = x.shift(1)

    lag_cols = [f"{c}_lag" for c in x.columns]
    combined = pd.concat([x_lag.set_axis(lag_cols, axis=1), x], axis=1)
    cov_full = combined.cov(min_periods=24)

    cross_cov = cov_full.loc[lag_cols, x.columns]
    cross_cov.index = x.columns  # drop "_lag" suffix for readability
    cross_cov.columns = x.columns

    n = cross_cov.shape[0]
    diag_vals = np.diag(cross_cov.values)
    off_diag_mask = ~np.eye(n, dtype=bool)
    off_diag_vals = cross_cov.values[off_diag_mask]

    auto_cov_component = np.nanmean(diag_vals)
    cross_serial_component = np.nanmean(off_diag_vals)

    means = x.mean(skipna=True)
    mean_component = np.nanmean(means.values ** 2)

    months_per_year = 12
    result = {
        "cross_cov_matrix": cross_cov,
        "auto_covariance_component": auto_cov_component,
        "cross_serial_component": cross_serial_component,
        "mean_component": mean_component,
        "auto_covariance_annualized": auto_cov_component * months_per_year,
        "cross_serial_annualized": cross_serial_component * months_per_year,
        "mean_component_annualized": mean_component * months_per_year,
    }

    components = {
        "Auto-covariance": abs(result["auto_covariance_component"]),
        "Cross-serial": abs(result["cross_serial_component"]),
        "Mean": abs(result["mean_component"]),
    }
    result["dominant_component"] = max(components, key=components.get)
    return result


def run_xsmom_backtest(
    excess_returns_daily: pd.DataFrame,
    sigma_daily_lagged: pd.DataFrame,
    cfg: Config,
) -> dict:
    """Run the cross-sectional momentum benchmark strategy."""
    xsmom = sig_mod.build_xsmom_signals(excess_returns_daily, sigma_daily_lagged, cfg)
    xsmom_instrument_returns = xsmom["instrument_returns"]
    # equal-weighted long/short: weights already sum to 0 with 1/n_long, 1/n_short scaling
    xsmom_portfolio = xsmom_instrument_returns.sum(axis=1, skipna=True, min_count=1)
    n_available = xsmom_instrument_returns.notna().sum(axis=1)
    xsmom_portfolio = xsmom_portfolio.where(n_available >= cfg.min_instruments_for_portfolio)
    return {
        "xsmom": xsmom,
        "xsmom_instrument_returns": xsmom_instrument_returns,
        "xsmom_portfolio_returns": xsmom_portfolio.dropna(),
    }


def compare_tsmom_xsmom(tsmom_returns: pd.Series, xsmom_returns: pd.Series, cfg: Config) -> pd.DataFrame:
    """Side-by-side performance summary of TSMOM vs XSMOM."""
    tsmom_stats = bt.performance_summary(tsmom_returns, cfg)
    xsmom_stats = bt.performance_summary(xsmom_returns, cfg)
    return pd.DataFrame({"TSMOM": tsmom_stats, "XSMOM": xsmom_stats})
