"""
Step 2: Johansen cointegration test.

For each factor j, tests whether [factor_price_j, macro_driver_1, ...,
macro_driver_k] share a common stochastic trend (are cointegrated). The VAR
lag order is chosen by AIC on the levels VAR, then used as k_ar_diff in the
Johansen VECM, exactly as specified.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config import Config


def select_var_lag(data: pd.DataFrame, maxlags: int) -> int:
    model = VAR(data)
    selection = model.select_order(maxlags=maxlags)
    return max(int(selection.aic), 0)


def run_johansen_test(factor_price: pd.Series, macro_driver_levels: pd.DataFrame, cfg: Config):
    """Run the Johansen test on [factor_price, macro_driver_levels...]."""
    data = pd.concat([factor_price, macro_driver_levels], axis=1).dropna()
    optimal_lag = select_var_lag(data, cfg.johansen_maxlags)
    result = coint_johansen(data.values, cfg.johansen_det_order, optimal_lag)
    return result, optimal_lag, len(data)


def run_all_johansen_tests(factor_prices: pd.DataFrame, macro_driver_levels: pd.DataFrame, cfg: Config) -> dict:
    """Run the Johansen test for every factor. Returns {factor: (result, lag, nobs)}."""
    return {
        j: run_johansen_test(factor_prices[j], macro_driver_levels, cfg)
        for j in cfg.factors
    }


def build_table1(johansen_results: dict, cfg: Config) -> dict:
    """Build Table 1: Trace and L-max panels, rows r=0..r<=n-1, cols = factors + 95% CV."""
    n_vars = 1 + len(cfg.macro_drivers)
    row_labels = ["r = 0"] + [f"r <= {k}" for k in range(1, n_vars)]

    factor_names = [cfg.factor_labels[j] for j in cfg.factors]

    trace_data = {cfg.factor_labels[j]: johansen_results[j][0].lr1 for j in cfg.factors}
    lmax_data = {cfg.factor_labels[j]: johansen_results[j][0].lr2 for j in cfg.factors}

    trace_df = pd.DataFrame(trace_data, index=row_labels)
    lmax_df = pd.DataFrame(lmax_data, index=row_labels)

    # Critical values depend only on (n_vars, det_order), identical across factors.
    any_result = johansen_results[cfg.factors[0]][0]
    trace_df["95% CV"] = any_result.cvt[:, 1]
    lmax_df["95% CV"] = any_result.cvm[:, 1]

    return {"trace": trace_df, "lmax": lmax_df, "factor_names": factor_names}


def summarize_cointegration_rank(johansen_results: dict, cfg: Config) -> pd.DataFrame:
    """For each factor, determine the cointegrating rank via the trace test at 95%."""
    rows = []
    for j in cfg.factors:
        result, lag, nobs = johansen_results[j]
        trace_stats = result.lr1
        cv95 = result.cvt[:, 1]
        rank = 0
        for stat, cv in zip(trace_stats, cv95):
            if stat > cv:
                rank += 1
            else:
                break
        rows.append({
            "Factor": cfg.factor_labels[j],
            "VAR lag (AIC)": lag,
            "N obs": nobs,
            "Reject r=0 @95%": trace_stats[0] > cv95[0],
            "Cointegrating rank": rank,
        })
    return pd.DataFrame(rows).set_index("Factor")
