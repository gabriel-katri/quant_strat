"""
Step 1 (Melone static): out-of-sample evaluation, expanding window.

Fully real-time: at each quarter t, BOTH the long-run (ECT) regression and
the predictive regression are re-estimated using only data through t-1, so
the forecast for t has no look-ahead. Initial training window: 100 quarters
(1975Q1-1999Q4), same as the rolling-Lasso window (cfg.rolling_window_periods)
so the two methods are evaluated on the identical 2000Q1+ OOS period.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

import melone_ect as ect_mod
from config import Config


def run_oos_for_factor(factor_return: pd.Series, factor_price: pd.Series,
                        macro_levels: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    min_train = cfg.rolling_window_periods
    dates = factor_return.index

    rows = []
    for t in range(min_train, len(dates)):
        price_train = factor_price.iloc[:t]
        macro_train = macro_levels.iloc[:t]
        trend_train = ect_mod.build_trend(price_train.index)

        lr_model = ect_mod.estimate_long_run_regression(price_train, macro_train, trend=trend_train, hac=False)
        ect_train = pd.Series(lr_model.resid, index=price_train.index)

        ret_train = factor_return.iloc[1:t]
        x_train = ect_train.iloc[:-1]
        X = sm.add_constant(x_train.values)
        pred_model = sm.OLS(ret_train.values, X).fit()

        forecast = float(pred_model.params[0] + pred_model.params[1] * ect_train.iloc[-1])
        rows.append({"date": dates[t], "actual": float(factor_return.iloc[t]), "predicted": forecast})

    return pd.DataFrame(rows).set_index("date")


def run_all_oos(dataset: pd.DataFrame, factor_prices: pd.DataFrame, macro_levels: pd.DataFrame,
                 cfg: Config) -> dict:
    """Returns {factor: predictions_df}, predictions_df has 'actual'/'predicted' columns."""
    return {
        f: run_oos_for_factor(dataset[f], factor_prices[f], macro_levels, cfg)
        for f in cfg.unconstrained_factors
    }
