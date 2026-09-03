"""
Step 1 (Favero, Melone & Tamoni 2022): build factor price levels and macro
driver levels from the already-assembled quarterly dataset
(results/rolling_lasso_dataset.csv).

Factor prices are cumulative log returns, ln F_t = sum_{s<=t} ln(1+r_s) --
an I(1) log-price index built from the I(0) quarterly factor returns. Macro
driver levels are the cumulative sum of each macro driver (OIL_WTI_LOGRET,
POT_GDP_GROWTH, TERM_SPREAD_10Y3M, LIQUIDITY_PS): the paper
cumulates the macro series itself to get a persistent "macro trend" driver,
regardless of whether the underlying series is a rate, a level, or already
a growth rate. Step 2 tests both sets of levels for cointegration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config


def build_factor_log_returns(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    return np.log1p(df[factors])


def build_factor_prices(df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    """ln F_t, one column per factor -- cumulative sum of log returns."""
    return build_factor_log_returns(df, factors).cumsum()


def build_macro_driver_levels(df: pd.DataFrame, macro_columns: list[str]) -> pd.DataFrame:
    """M_t, one column per macro signal -- cumulative sum of the raw series."""
    return df[macro_columns].cumsum()


def build_levels(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (factor_prices, macro_driver_levels), both indexed like `df`."""
    factor_prices = build_factor_prices(df, cfg.unconstrained_factors)
    macro_levels = build_macro_driver_levels(df, cfg.macro_columns)
    return factor_prices, macro_levels
