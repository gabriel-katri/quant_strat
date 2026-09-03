"""
Step 1 (Melone static): long-run regression and the static ECT.

For each factor: ln F_t = alpha_0 + alpha_1 * t + beta' * M_t + w_t,
estimated by OLS (Newey-West HAC SEs, automatic bandwidth) on the WHOLE
sample. w_t (the regression residual) is the static error-correction term
-- the factor's deviation from its macro-implied long-run equilibrium.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import Config


def build_trend(index: pd.Index) -> pd.Series:
    """Position within the sample passed in (0, 1, 2, ...), not calendar time."""
    return pd.Series(np.arange(len(index)), index=index, name="trend", dtype=float)


def estimate_long_run_regression(factor_price: pd.Series, macro_levels: pd.DataFrame,
                                  trend: pd.Series | None = None, hac: bool = True):
    """OLS of factor_price on [const, trend, macro_levels...]."""
    if trend is None:
        trend = build_trend(factor_price.index)
    X = sm.add_constant(pd.concat([trend, macro_levels], axis=1))
    if hac:
        return sm.OLS(factor_price.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": None})
    return sm.OLS(factor_price.values, X.values).fit()


def estimate_all_long_run(factor_prices: pd.DataFrame, macro_levels: pd.DataFrame, cfg: Config) -> dict:
    """Returns {factor: statsmodels OLS results}."""
    return {f: estimate_long_run_regression(factor_prices[f], macro_levels) for f in cfg.unconstrained_factors}


def get_static_ects(models: dict, factor_prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Static ECT_t = residual of the full-sample long-run regression, one column per factor."""
    return pd.DataFrame({f: pd.Series(models[f].resid, index=factor_prices.index) for f in cfg.unconstrained_factors})


def build_beta_table(models: dict, cfg: Config) -> pd.DataFrame:
    """Coefficients (const, trend, macro betas), R^2 and N obs, one column per factor."""
    coef_names = ["const", "trend"] + list(cfg.macro_columns)
    rows = {}
    for f in cfg.unconstrained_factors:
        m = models[f]
        rows[f] = list(m.params) + [m.rsquared, int(m.nobs)]
    index = coef_names + ["R2", "N obs"]
    return pd.DataFrame(rows, index=index)


def get_beta_static_series(models: dict, cfg: Config) -> dict:
    """{factor: pd.Series} of just the coefficients (const, trend, macro betas),
    indexed by name -- the anchor Melone's Kalman mean-reverting variant
    (melone_kalman.py) reverts each beta_t toward."""
    coef_names = ["const", "trend"] + list(cfg.macro_columns)
    return {f: pd.Series(models[f].params, index=coef_names) for f in cfg.unconstrained_factors}
