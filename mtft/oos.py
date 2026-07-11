"""
Step 5: Out-of-sample evaluation.

Fully real-time expanding window: at each quarter t, BOTH the long-run (ECT)
regression and the predictive regression are re-estimated using only data
through t-1, so the forecast for t has no look-ahead bias. (The prompt's
pseudocode notes this as the "fully real-time version" vs. a "fixed
parameter" version that reuses the full-sample ECT; we implement the
real-time version since it is the defensible true-OOS test.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from config import Config
from ect import build_trend

MIN_TRAIN_OBS = 20


def oos_r2_campbell_thompson(actual: np.ndarray, forecast_model: np.ndarray, forecast_mean: np.ndarray) -> float:
    """R^2_OOS = 1 - SSE(model) / SSE(historical mean), Campbell & Thompson (2008)."""
    sse_model = np.sum((actual - forecast_model) ** 2)
    sse_mean = np.sum((actual - forecast_mean) ** 2)
    return 1.0 - sse_model / sse_mean


def clark_west_test(actual: np.ndarray, forecast_model: np.ndarray, forecast_mean: np.ndarray):
    """Clark & West (2007) MSPE-adjusted test for equal predictive accuracy."""
    d = (actual - forecast_mean) ** 2 - ((actual - forecast_model) ** 2 - (forecast_model - forecast_mean) ** 2)
    t_stat = np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d)))
    p_value = 1.0 - stats.norm.cdf(t_stat)
    return t_stat, p_value


def _first_oos_index(index: pd.DatetimeIndex, start_date: str) -> int:
    pos = index.searchsorted(pd.Timestamp(start_date))
    return int(pos)


def run_oos_for_factor_and_start(
    factor_log_return: pd.Series,
    factor_price: pd.Series,
    macro_driver_levels: pd.DataFrame,
    start_date: str,
) -> pd.DataFrame:
    """Expanding-window real-time OOS forecasts for a single factor from `start_date`."""
    full_trend = build_trend(factor_log_return.index)
    first_t = max(_first_oos_index(factor_log_return.index, start_date), MIN_TRAIN_OBS)

    dates, actuals, forecasts_model, forecasts_mean = [], [], [], []
    for t in range(first_t, len(factor_log_return)):
        trend_train = full_trend.iloc[:t]
        price_train = factor_price.iloc[:t]
        macro_train = macro_driver_levels.iloc[:t]

        X_lr = sm.add_constant(pd.concat([trend_train, macro_train], axis=1))
        lr_model = sm.OLS(price_train, X_lr).fit()
        ect_train = lr_model.resid

        Y_train = factor_log_return.iloc[1:t]
        X_train = sm.add_constant(ect_train.iloc[:-1])
        pred_model = sm.OLS(Y_train.values, X_train.values).fit()

        forecast = pred_model.params[0] + pred_model.params[1] * ect_train.iloc[-1]

        dates.append(factor_log_return.index[t])
        actuals.append(factor_log_return.iloc[t])
        forecasts_model.append(forecast)
        forecasts_mean.append(Y_train.mean())

    return pd.DataFrame(
        {"actual": actuals, "forecast_model": forecasts_model, "forecast_mean": forecasts_mean}, index=dates
    )


def run_all_oos_evaluations(
    factor_log_returns: pd.DataFrame, factor_prices: pd.DataFrame, macro_driver_levels: pd.DataFrame, cfg: Config
) -> dict:
    """Returns {start_date: {factor: forecast_df}}."""
    out = {}
    for start_date in cfg.oos_start_dates:
        out[start_date] = {
            j: run_oos_for_factor_and_start(factor_log_returns[j], factor_prices[j], macro_driver_levels, start_date)
            for j in cfg.factors
        }
    return out


def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def build_table3_panel_b(oos_results: dict, cfg: Config) -> pd.DataFrame:
    """Table 3 Panel B: OOS R^2 (%) with Clark-West significance stars."""
    row_labels = {d: f"From {pd.Timestamp(d).year}" for d in cfg.oos_start_dates}
    rows = {}
    for start_date in cfg.oos_start_dates:
        row = {}
        for j in cfg.factors:
            df = oos_results[start_date][j]
            r2 = oos_r2_campbell_thompson(df["actual"].values, df["forecast_model"].values, df["forecast_mean"].values)
            _, p = clark_west_test(df["actual"].values, df["forecast_model"].values, df["forecast_mean"].values)
            row[cfg.factor_labels[j]] = f"{r2 * 100:.2f}{_stars(p)}"
        rows[row_labels[start_date]] = row
    return pd.DataFrame(rows).T
