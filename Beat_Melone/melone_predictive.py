"""
Step 1 (Melone static): predictive regression r_{t+1} = a + delta * ECT_t + v_{t+1}.

HAC (Newey-West, automatic bandwidth) SEs. The target is the factor's
*simple* return (not the log return used to build ln F_t and the ECT),
keeping return units consistent with the rest of the project (rolling
Lasso, portfolio P&L) so the final Static/Kalman/LASSO/Benchmark comparison
isn't mixing return conventions.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from config import Config


def run_predictive_regression(factor_return: pd.Series, ect: pd.Series):
    """OLS of r_{t+1} on [const, ECT_t]. Returns (model, y, x_ect), all aligned by position."""
    y = factor_return.iloc[1:]
    x_ect = ect.iloc[:-1]
    X = sm.add_constant(x_ect.values)
    model = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": None})
    return model, y, x_ect


def run_all_predictive(dataset: pd.DataFrame, ects: pd.DataFrame, cfg: Config) -> dict:
    """Returns {factor: {'model', 'y', 'x_ect'}}."""
    out = {}
    for f in cfg.unconstrained_factors:
        model, y, x_ect = run_predictive_regression(dataset[f], ects[f])
        out[f] = {"model": model, "y": y, "x_ect": x_ect}
    return out


def build_predictive_table(results: dict, cfg: Config) -> pd.DataFrame:
    rows = []
    for f in cfg.unconstrained_factors:
        model = results[f]["model"]
        rows.append({
            "Factor": f,
            "delta": model.params[1],
            "delta_se (HAC)": model.bse[1],
            "p-value (HAC)": model.pvalues[1],
            "R2": model.rsquared,
            "N obs": int(model.nobs),
        })
    return pd.DataFrame(rows).set_index("Factor")
