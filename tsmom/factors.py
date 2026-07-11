"""
Factor regression analysis (Moskowitz, Ooi & Pedersen 2012, Eq. 4).

Regresses TSMOM monthly returns on the Fama-French/Carhart four factors
(MKT, SMB, HML, UMD) downloaded from Ken French's data library, and
separately on MKT + MKT^2 to test for the option-like ("smile") payoff
documented in the paper.

Standard errors are Newey-West (HAC) to account for the serial correlation
that TSMOM returns exhibit by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas_datareader.famafrench import FamaFrenchReader

from config import Config

NW_MAXLAGS = 6


def download_ff_factors(start: str, end: str | None) -> pd.DataFrame:
    """Download monthly MKT, SMB, HML (Fama-French 3) and UMD (momentum).

    Returns a DataFrame in decimal (not percent) units, indexed by
    month-end Timestamp, with columns ['MKT', 'SMB', 'HML', 'UMD', 'RF'].
    """
    ff3 = FamaFrenchReader("F-F_Research_Data_Factors", start=start, end=end).read()[0]
    ff3.columns = [c.strip() for c in ff3.columns]
    ff3 = ff3.rename(columns={"Mkt-RF": "MKT"})

    mom = FamaFrenchReader("F-F_Momentum_Factor", start=start, end=end).read()[0]
    mom.columns = [c.strip() for c in mom.columns]
    mom = mom.rename(columns={mom.columns[0]: "UMD"})

    factors = ff3.join(mom, how="inner") / 100.0

    factors.index = factors.index.to_timestamp(how="end").normalize()
    return factors


def _align_to_month_end(monthly_returns: pd.Series, factors: pd.DataFrame) -> pd.DataFrame:
    """Align a strategy return series (indexed by trading-day month-ends)
    with FF factors (indexed by calendar month-end) via a period key.
    """
    ret = monthly_returns.copy()
    ret.index = pd.PeriodIndex(ret.index, freq="M")
    fac = factors.copy()
    fac.index = pd.PeriodIndex(fac.index, freq="M")
    merged = pd.concat([ret.rename("strategy"), fac], axis=1, join="inner")
    return merged


def run_four_factor_regression(monthly_returns: pd.Series, factors: pd.DataFrame) -> dict:
    """OLS of strategy returns on MKT, SMB, HML, UMD with Newey-West SEs."""
    data = _align_to_month_end(monthly_returns, factors)
    y = data["strategy"]
    X = sm.add_constant(data[["MKT", "SMB", "HML", "UMD"]])
    model = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": NW_MAXLAGS})
    return _pack_results(model, data)


def run_smile_regression(monthly_returns: pd.Series, factors: pd.DataFrame) -> dict:
    """OLS of strategy returns on MKT and MKT^2 -- tests for the straddle-like
    (convex) payoff profile relative to the equity market.
    """
    data = _align_to_month_end(monthly_returns, factors)
    data = data.copy()
    data["MKT2"] = data["MKT"] ** 2
    y = data["strategy"]
    X = sm.add_constant(data[["MKT", "MKT2"]])
    model = sm.OLS(y, X, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": NW_MAXLAGS})
    return _pack_results(model, data)


def _pack_results(model, data: pd.DataFrame) -> dict:
    return {
        "model": model,
        "params": model.params,
        "tstats": model.tvalues,
        "pvalues": model.pvalues,
        "r_squared": model.rsquared,
        "n_obs": int(model.nobs),
        "data": data,
    }


def summarize_regression(results: dict, months_per_year: int, label: str) -> pd.DataFrame:
    """Tabulate coefficients, annualized alpha, t-stats and R^2."""
    params = results["params"]
    tstats = results["tstats"]
    rows = []
    for name in params.index:
        annualized = params[name] * months_per_year if name == "const" else params[name]
        rows.append({
            "Regression": label,
            "Term": "alpha" if name == "const" else name,
            "Coefficient": params[name],
            "Annualized (if alpha)": annualized if name == "const" else np.nan,
            "t-stat": tstats[name],
        })
    df = pd.DataFrame(rows)
    df.attrs["r_squared"] = results["r_squared"]
    df.attrs["n_obs"] = results["n_obs"]
    return df
