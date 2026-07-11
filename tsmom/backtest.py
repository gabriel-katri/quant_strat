"""
Portfolio construction, return aggregation, and performance metrics.

The diversified TSMOM (and passive-long benchmark) portfolios are simple
equal-weighted averages of the available instrument-level strategy returns
each month -- instruments without a valid signal/vol estimate that month
are excluded from the average, not treated as zero.

Transaction costs are modeled as a simple bps charge on position turnover
(currently defaulted to 0 via config), applied to each instrument's
month-over-month change in position size at the point returns are
aggregated, so a cost model can be swapped in later without touching the
signal generation logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config
import signals as sig_mod


def apply_transaction_costs(
    instrument_returns: pd.DataFrame,
    position_size: pd.DataFrame,
    signal: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    """Subtract a simple turnover-based transaction cost from instrument returns.

    Cost is charged proportional to the absolute change in signed position
    (|w_t - w_{t-1}|) at cost_bps / 10000 per unit of turnover. With
    cost_bps = 0 (the default) this is a no-op.
    """
    if cost_bps == 0:
        return instrument_returns

    signed_position = (signal * position_size).reindex(instrument_returns.index)
    turnover = signed_position.diff().abs()
    cost = turnover.shift(1) * (cost_bps / 10000.0)
    return instrument_returns - cost.reindex_like(instrument_returns).fillna(0.0)


def diversified_portfolio_return(instrument_returns: pd.DataFrame, min_instruments: int = 1) -> pd.Series:
    """Equal-weighted average return across all instruments with a valid
    signal that month (NaN instruments are excluded from the average, not
    zeroed out).

    Months where fewer than `min_instruments` instruments have a valid
    return are set to NaN: with too few names, "diversified" is a misnomer
    and the portfolio is really an undiversified, highly-levered single- or
    few-name bet driven by whichever ETFs happened to exist yet.
    """
    avg = instrument_returns.mean(axis=1, skipna=True)
    n_available = instrument_returns.notna().sum(axis=1)
    return avg.where(n_available >= min_instruments)


def run_tsmom_backtest(
    excess_returns_daily: pd.DataFrame,
    sigma_daily_lagged: pd.DataFrame,
    cfg: Config,
) -> dict:
    """Run the full TSMOM backtest and the passive-long benchmark.

    Returns a dict with instrument-level detail plus the two portfolio
    return series ('tsmom' and 'passive_long').
    """
    tsmom = sig_mod.build_tsmom_signals(excess_returns_daily, sigma_daily_lagged, cfg, long_only=False)
    passive = sig_mod.build_tsmom_signals(excess_returns_daily, sigma_daily_lagged, cfg, long_only=True)

    tsmom_instrument_returns = apply_transaction_costs(
        tsmom["instrument_returns"], tsmom["position_size"], tsmom["signal"], cfg.transaction_cost_bps
    )
    passive_instrument_returns = apply_transaction_costs(
        passive["instrument_returns"], passive["position_size"], passive["signal"], cfg.transaction_cost_bps
    )

    tsmom_portfolio = diversified_portfolio_return(tsmom_instrument_returns, cfg.min_instruments_for_portfolio)
    passive_portfolio = diversified_portfolio_return(passive_instrument_returns, cfg.min_instruments_for_portfolio)

    return {
        "tsmom": tsmom,
        "passive": passive,
        "tsmom_instrument_returns": tsmom_instrument_returns,
        "passive_instrument_returns": passive_instrument_returns,
        "tsmom_portfolio_returns": tsmom_portfolio.dropna(),
        "passive_portfolio_returns": passive_portfolio.dropna(),
    }


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def cumulative_returns(monthly_returns: pd.Series) -> pd.Series:
    """Compound a monthly return series into a cumulative wealth index (base 1.0)."""
    return (1.0 + monthly_returns).cumprod()


def annualized_return(monthly_returns: pd.Series, months_per_year: int) -> float:
    n = len(monthly_returns)
    if n == 0:
        return np.nan
    total_growth = (1.0 + monthly_returns).prod()
    years = n / months_per_year
    return total_growth ** (1.0 / years) - 1.0


def annualized_volatility(monthly_returns: pd.Series, months_per_year: int) -> float:
    return monthly_returns.std(ddof=1) * np.sqrt(months_per_year)


def sharpe_ratio(monthly_returns: pd.Series, months_per_year: int) -> float:
    """Returns are already excess-of-cash, so Sharpe = mean / std (annualized)."""
    mu = monthly_returns.mean() * months_per_year
    sigma = annualized_volatility(monthly_returns, months_per_year)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return mu / sigma


def max_drawdown(monthly_returns: pd.Series) -> float:
    wealth = cumulative_returns(monthly_returns)
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return drawdown.min()


def calmar_ratio(monthly_returns: pd.Series, months_per_year: int) -> float:
    mdd = max_drawdown(monthly_returns)
    if mdd == 0 or np.isnan(mdd):
        return np.nan
    return annualized_return(monthly_returns, months_per_year) / abs(mdd)


def hit_rate(monthly_returns: pd.Series) -> float:
    return (monthly_returns > 0).mean()


def best_worst_month(monthly_returns: pd.Series) -> tuple[float, float]:
    return monthly_returns.max(), monthly_returns.min()


def best_worst_year(monthly_returns: pd.Series) -> tuple[float, float]:
    annual = (1.0 + monthly_returns).groupby(monthly_returns.index.year).prod() - 1.0
    return annual.max(), annual.min()


def drawdown_series(monthly_returns: pd.Series) -> pd.Series:
    wealth = cumulative_returns(monthly_returns)
    running_max = wealth.cummax()
    return wealth / running_max - 1.0


def rolling_sharpe(monthly_returns: pd.Series, window: int, months_per_year: int) -> pd.Series:
    roll_mean = monthly_returns.rolling(window).mean() * months_per_year
    roll_std = monthly_returns.rolling(window).std(ddof=1) * np.sqrt(months_per_year)
    return roll_mean / roll_std


def performance_summary(monthly_returns: pd.Series, cfg: Config) -> dict:
    """Compute the full set of headline performance metrics for a return series."""
    mpy = cfg.months_per_year
    best_m, worst_m = best_worst_month(monthly_returns)
    best_y, worst_y = best_worst_year(monthly_returns)
    return {
        "Annualized Return": annualized_return(monthly_returns, mpy),
        "Annualized Volatility": annualized_volatility(monthly_returns, mpy),
        "Sharpe Ratio": sharpe_ratio(monthly_returns, mpy),
        "Max Drawdown": max_drawdown(monthly_returns),
        "Calmar Ratio": calmar_ratio(monthly_returns, mpy),
        "Monthly Hit Rate": hit_rate(monthly_returns),
        "Best Month": best_m,
        "Worst Month": worst_m,
        "Best Year": best_y,
        "Worst Year": worst_y,
        "N Months": len(monthly_returns),
    }


def per_instrument_sharpe(instrument_returns: pd.DataFrame, cfg: Config) -> pd.Series:
    """Annualized Sharpe ratio for each instrument's TSMOM sub-strategy."""
    return instrument_returns.apply(lambda col: sharpe_ratio(col.dropna(), cfg.months_per_year))
