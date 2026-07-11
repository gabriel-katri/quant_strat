"""
Step 3: Estimate the Error Correction Term (ECT) via the long-run regression.

For each factor j: factor_price[j] = alpha_0 + alpha_1 * t + beta' * M_t + ECT[j]_t
Estimated by OLS with HAC (Newey-West, automatic bandwidth) standard errors.
The ECT is the regression residual: the factor's deviation from its
macro-implied long-run equilibrium.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import Config


def build_trend(index: pd.Index) -> pd.Series:
    """Calendar-consistent trend: position within the FULL sample, not a subsample."""
    return pd.Series(np.arange(len(index)), index=index, name="trend")


def estimate_long_run_regression(
    factor_price: pd.Series, macro_driver_levels: pd.DataFrame, trend: pd.Series | None = None, hac: bool = True
):
    """OLS of factor_price on [const, trend, macro_driver_levels], HAC SEs by default.

    `trend` may be passed in (e.g. a slice of the full-sample trend, for
    recursive/expanding-window re-estimation in Step 5) so the trend stays
    calendar-consistent instead of resetting to 0 on every subsample.
    """
    Y = factor_price
    if trend is None:
        trend = build_trend(Y.index)
    X = sm.add_constant(pd.concat([trend, macro_driver_levels], axis=1))
    if hac:
        model = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": None})
    else:
        model = sm.OLS(Y, X).fit()
    return model


def estimate_all_ects(factor_prices: pd.DataFrame, macro_driver_levels: pd.DataFrame, cfg: Config) -> dict:
    """Returns {factor: statsmodels OLS results}. ECT[j] = results.resid."""
    return {j: estimate_long_run_regression(factor_prices[j], macro_driver_levels) for j in cfg.factors}


def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def build_table_a1(long_run_models: dict, cfg: Config) -> pd.DataFrame:
    """Table A.1: long-run regression coefficients (coef, SE, stars) per factor."""
    coef_rows = ["const", "trend"] + list(cfg.macro_drivers)
    columns = {}
    for j in cfg.factors:
        model = long_run_models[j]
        col = []
        for name in coef_rows:
            coef = model.params[name]
            se = model.bse[name]
            p = model.pvalues[name]
            col.append(f"{coef:.4f}{_stars(p)}\n({se:.4f})")
        col.append(f"{model.rsquared:.4f}")
        col.append(f"{int(model.nobs)}")
        columns[cfg.factor_labels[j]] = col
    index = coef_rows + ["R^2", "Observations"]
    return pd.DataFrame(columns, index=index)


def apply_fitted_long_run(model, factor_price: pd.Series, macro_driver_levels: pd.DataFrame, trend: pd.Series) -> pd.Series:
    """Compute ECT_t = factor_price_t - fitted(t) using a model's coefficients
    applied out-of-window to a (typically longer) series -- e.g. coefficients
    fit on a training subsample, applied to the full sample for Step 6's OOS
    split test or Step 7's fixed-coefficient extended backtest."""
    X = sm.add_constant(pd.concat([trend, macro_driver_levels], axis=1))
    X = X[model.params.index]  # match column order or use dict cast below
    fitted = model.predict(X)
    return factor_price - fitted


def get_ects(long_run_models: dict, cfg: Config) -> pd.DataFrame:
    """ECT[j]_t = residual of the long-run regression for factor j.

    Keyed by raw FF factor name (e.g. "Mkt-RF"), matching factor_log_returns
    / factor_prices, for consistency across all downstream modules. Use
    cfg.factor_labels only when building display tables.
    """
    return pd.DataFrame({j: long_run_models[j].resid for j in cfg.factors})
