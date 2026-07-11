"""
Step 4: In-sample predictive regression.

f_{j,t+1} = alpha + delta * ECT_{j,t} + e_{t+1}

ECT_t (the deviation from long-run equilibrium) should negatively predict
next-quarter's factor return: delta < 0 (mean reversion toward the
macro-implied price).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import Config


def run_predictive_regression(factor_log_return: pd.Series, ect: pd.Series):
    """OLS of f_{t+1} on [const, ECT_t], HAC (Newey-West, automatic bandwidth) SEs."""
    Y = factor_log_return.iloc[1:]
    X_ect = ect.iloc[:-1]
    X = sm.add_constant(X_ect)
    model = sm.OLS(Y.values, X.values).fit(cov_type="HAC", cov_kwds={"maxlags": None})
    return model, Y, X


def run_all_predictive_regressions(factor_log_returns: pd.DataFrame, ects: pd.DataFrame, cfg: Config) -> dict:
    """Keyed by raw FF factor name, matching `ects` and `factor_log_returns`."""
    results = {}
    for j in cfg.factors:
        model, Y, X = run_predictive_regression(factor_log_returns[j], ects[j])
        results[j] = {"model": model, "Y": Y, "X": X}
    return results


def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def build_table3_panel_a(predictive_results: dict, cfg: Config) -> pd.DataFrame:
    """Table 3 Panel A: ECT(-1) coefficient, constant, N, R^2, sigma[E_t]/E."""
    columns = {}
    for j in cfg.factors:
        r = predictive_results[j]
        model, Y = r["model"], r["Y"]
        delta, const = model.params[1], model.params[0]
        se_delta, se_const = model.bse[1], model.bse[0]
        p_delta, p_const = model.pvalues[1], model.pvalues[0]

        fitted = model.fittedvalues
        sigma_ratio = np.std(fitted) / np.abs(np.mean(Y))

        col = [
            f"{delta:.4f}{_stars(p_delta)}\n({se_delta:.4f})",
            f"{const:.4f}{_stars(p_const)}\n({se_const:.4f})",
            f"{int(model.nobs)}",
            f"{model.rsquared:.4f}",
            f"{sigma_ratio:.4f}",
        ]
        columns[cfg.factor_labels[j]] = col

    index = ["ECT(-1)", "Constant", "Observations", "R^2", "sigma[E_t]/E"]
    return pd.DataFrame(columns, index=index)
