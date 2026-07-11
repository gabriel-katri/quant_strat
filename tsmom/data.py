"""
Data acquisition and cleaning.

Downloads daily adjusted prices for the instrument universe and the
risk-free proxy from yfinance, and derives daily excess returns for every
instrument. Each instrument keeps its own history (max available), and
returns are simply left as NaN before an instrument's inception -- all
downstream code must handle a ragged panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from config import Config


def _download_adj_close(tickers: list[str], start: str, end: str | None) -> pd.DataFrame:
    """Download daily adjusted close prices for a list of tickers.

    Returns a DataFrame indexed by date with one column per ticker.
    """
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,  # 'Close' already dividend/split adjusted
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        closes = {}
        for t in tickers:
            if t in raw.columns.get_level_values(0):
                closes[t] = raw[t]["Close"]
        prices = pd.DataFrame(closes)
    else:
        # Single ticker case
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    return prices


def download_prices(cfg: Config) -> pd.DataFrame:
    """Download adjusted close prices for the full instrument universe."""
    return _download_adj_close(cfg.tickers, cfg.data_start, cfg.data_end)


def download_risk_free(cfg: Config) -> pd.Series:
    """Download the 3-month T-bill rate (^IRX) and convert to a daily rate.

    ^IRX is quoted as an annualized discount yield in percentage points
    (e.g. 5.25 means 5.25%/yr). We convert to a simple daily rate by
    dividing by the trading-day count, and forward-fill non-trading gaps.
    """
    raw = yf.download(
        cfg.risk_free_ticker,
        start=cfg.data_start,
        end=cfg.data_end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        irx = raw["Close"][cfg.risk_free_ticker]
    else:
        irx = raw["Close"]
    irx = irx.sort_index()
    irx.index = pd.to_datetime(irx.index).tz_localize(None)
    irx = irx.ffill()

    daily_rf = (irx / 100.0) / cfg.trading_days_per_year
    daily_rf.name = "rf"
    return daily_rf


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily percentage returns, NaN-safe across ragged histories."""
    return prices.pct_change()


def compute_excess_returns(returns: pd.DataFrame, daily_rf: pd.Series) -> pd.DataFrame:
    """Subtract the daily risk-free rate from each instrument's return.

    Only subtracts on dates where the instrument actually has a return
    (keeps NaN before inception / after delisting untouched).
    """
    rf_aligned = daily_rf.reindex(returns.index).ffill()
    excess = returns.sub(rf_aligned, axis=0)
    excess[returns.isna()] = np.nan
    return excess


def load_data(cfg: Config) -> dict:
    """End-to-end data load: prices, daily returns, excess returns, rf.

    Returns a dict with keys: prices, returns, excess_returns, daily_rf.
    """
    prices = download_prices(cfg)
    daily_rf = download_risk_free(cfg)
    returns = compute_daily_returns(prices)
    excess_returns = compute_excess_returns(returns, daily_rf)

    return {
        "prices": prices,
        "returns": returns,
        "excess_returns": excess_returns,
        "daily_rf": daily_rf,
    }
