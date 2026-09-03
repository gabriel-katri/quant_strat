"""
Step 1 (Melone static): Johansen cointegration test.

For each factor j, tests whether [ln F_j_t, M_t (the 4 macro driver
levels)] share a common stochastic trend (are cointegrated). The VAR lag
order is chosen by AIC on the levels VAR, then used as k_ar_diff in the
Johansen VECM. det_order=1 includes a linear trend in the cointegrating
relation, matching the long-run regression in melone_ect.py.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from config import Config


def select_var_lag(data: pd.DataFrame, maxlags: int) -> int:
    selection = VAR(data).select_order(maxlags=maxlags)
    return max(int(selection.aic), 1)


def run_johansen_test(factor_price: pd.Series, macro_levels: pd.DataFrame, cfg: Config):
    """Run the Johansen test on [factor_price, macro_levels...]. Returns (result, lag, nobs)."""
    data = pd.concat([factor_price, macro_levels], axis=1).dropna()
    lag = select_var_lag(data, cfg.johansen_maxlags)
    result = coint_johansen(data.values, cfg.johansen_det_order, lag)
    return result, lag, len(data)


def run_all_johansen_tests(factor_prices: pd.DataFrame, macro_levels: pd.DataFrame, cfg: Config) -> dict:
    """Returns {factor: (result, lag, nobs)}."""
    return {f: run_johansen_test(factor_prices[f], macro_levels, cfg) for f in cfg.unconstrained_factors}


def build_trace_lmax_tables(results: dict, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trace and L-max panels: rows r=0..r<=n-1, cols = factors + 95% CV."""
    n_vars = 1 + len(cfg.macro_columns)
    row_labels = ["r = 0"] + [f"r <= {k}" for k in range(1, n_vars)]

    trace = pd.DataFrame({f: results[f][0].lr1 for f in cfg.unconstrained_factors}, index=row_labels)
    lmax = pd.DataFrame({f: results[f][0].lr2 for f in cfg.unconstrained_factors}, index=row_labels)

    any_result = results[cfg.unconstrained_factors[0]][0]  # critical values depend only on (n_vars, det_order)
    trace["95% CV"] = any_result.cvt[:, 1]
    lmax["95% CV"] = any_result.cvm[:, 1]
    return trace, lmax


def summarize_cointegration_rank(results: dict, cfg: Config) -> pd.DataFrame:
    """For each factor, the cointegrating rank via the sequential trace test at 95%."""
    rows = []
    for f in cfg.unconstrained_factors:
        result, lag, nobs = results[f]
        trace_stats, cv95 = result.lr1, result.cvt[:, 1]
        rank = 0
        for stat, cv in zip(trace_stats, cv95):
            if stat > cv:
                rank += 1
            else:
                break
        rows.append({
            "Factor": f,
            "VAR lag (AIC)": lag,
            "N obs": nobs,
            "Reject r=0 @95%": bool(trace_stats[0] > cv95[0]),
            "Cointegrating rank": rank,
        })
    return pd.DataFrame(rows).set_index("Factor")
