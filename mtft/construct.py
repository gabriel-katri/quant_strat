"""
Step 1: Construct factor "prices" and macro driver levels.

Factor prices are cumulative log returns (a random-walk-plus-drift price
index); macro drivers are cumulative sums of the quarterly macro series.
Both are treated as I(1) processes that Step 2 tests for cointegration.
"""

from __future__ import annotations

import pandas as pd


def build_factor_prices(factor_log_returns: pd.DataFrame) -> pd.DataFrame:
    return factor_log_returns.cumsum()


def build_macro_driver_levels(macro_series: pd.DataFrame) -> pd.DataFrame:
    return macro_series.cumsum()
