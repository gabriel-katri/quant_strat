"""
Step 8: Risk management -- predicted return distributions around crises.

Compares two models for the Value portfolio's long leg (average of the two
High-B/M size portfolios) around the 2008-09 GFC and the 2020-21 COVID
crash/recovery:

  CER (Constant Expected Return):  H_{t+1} = mu_0 + eps_{t+1}
  Macro-FECM:                      H_{t+1} = alpha + delta_MKT*ECT_MKT_t
                                              + delta_HML*ECT_HML_t + v_{t+1}
    (both MKT and HML ECTs are used because the long value leg carries
    market risk as well as value risk)

Each model is estimated on data through the year before the crisis, then
residuals are bootstrapped to build a predicted annual-return distribution
for the crisis year; the year is then folded into the training data and the
next year's distribution is generated the same way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import ect as ect_mod
from config import Config
from data import ff5_monthly_to_quarterly_log_returns


def build_value_high_leg(size_bm_monthly: pd.DataFrame) -> pd.Series:
    """Quarterly log return of the 'High' B/M leg: avg(SMALL HiBM, BIG HiBM)."""
    avg_high = ((size_bm_monthly["SMALL HiBM"] + size_bm_monthly["BIG HiBM"]) / 2.0).to_frame("H")
    quarterly_log = ff5_monthly_to_quarterly_log_returns(avg_high)
    return quarterly_log["H"]


def _fixed_ect(factor_prices: pd.DataFrame, macro_driver_levels: pd.DataFrame, factor_col: str, train_cutoff: str):
    train = factor_prices.index <= train_cutoff
    trend_full = ect_mod.build_trend(factor_prices.index)
    lr_model = ect_mod.estimate_long_run_regression(
        factor_prices.loc[train, factor_col], macro_driver_levels.loc[train], hac=False
    )
    ect_full = ect_mod.apply_fitted_long_run(lr_model, factor_prices[factor_col], macro_driver_levels, trend_full)
    return ect_full


def fit_cer(H: pd.Series, train_cutoff: str):
    train = H.loc[H.index <= train_cutoff]
    mu0 = train.mean()
    resid = train - mu0
    return {"mu0": mu0, "resid": resid}


def fit_fecm(H: pd.Series, ect_mkt: pd.Series, ect_hml: pd.Series, train_cutoff: str):
    idx = H.index[H.index <= train_cutoff]
    Y = H.loc[idx].iloc[1:]
    X_df = pd.DataFrame({"ECT_MKT": ect_mkt, "ECT_HML": ect_hml}).loc[idx].iloc[:-1]
    X = sm.add_constant(X_df)
    model = sm.OLS(Y.values, X.values).fit()
    resid = pd.Series(model.resid, index=Y.index)
    return {"model": model, "resid": resid}


def _year_quarters(index: pd.DatetimeIndex, year: int) -> pd.DatetimeIndex:
    return index[index.year == year]


def bootstrap_cer_year(cer_fit: dict, H_index: pd.DatetimeIndex, year: int, cfg: Config) -> np.ndarray:
    quarters = _year_quarters(H_index, year)
    rng = np.random.default_rng(cfg.bootstrap_seed + year)
    resid_pool = cer_fit["resid"].values
    draws = rng.choice(resid_pool, size=(cfg.bootstrap_n, len(quarters)), replace=True)
    simulated = cer_fit["mu0"] + draws
    return simulated.sum(axis=1)


def bootstrap_fecm_year(
    fecm_fit: dict, ect_mkt: pd.Series, ect_hml: pd.Series, year: int, cfg: Config
) -> np.ndarray:
    quarters = _year_quarters(ect_mkt.index, year)
    alpha, d_mkt, d_hml = fecm_fit["model"].params
    quarterly_forecast = alpha + d_mkt * ect_mkt.loc[quarters].values + d_hml * ect_hml.loc[quarters].values

    rng = np.random.default_rng(cfg.bootstrap_seed + year)
    resid_pool = fecm_fit["resid"].values
    draws = rng.choice(resid_pool, size=(cfg.bootstrap_n, len(quarters)), replace=True)
    simulated = quarterly_forecast[None, :] + draws
    return simulated.sum(axis=1)


def realized_annual_return(H: pd.Series, year: int) -> float:
    return H.loc[H.index.year == year].sum()


def run_crisis_year_analysis(
    H: pd.Series, factor_prices: pd.DataFrame, macro_driver_levels: pd.DataFrame, years: list[int], cfg: Config
) -> dict:
    """For a sequence of years (e.g. [2008, 2009]), fit CER + FECM on data through
    year-1, bootstrap the predicted distribution for `year`, then include `year`
    in training for the next iteration."""
    H = H.loc[H.index.isin(factor_prices.index)]
    results = {}
    for year in years:
        train_cutoff = f"{year - 1}-12-31"

        ect_mkt = _fixed_ect(factor_prices, macro_driver_levels, "Mkt-RF", train_cutoff)
        ect_hml = _fixed_ect(factor_prices, macro_driver_levels, "HML", train_cutoff)

        cer_fit = fit_cer(H, train_cutoff)
        fecm_fit = fit_fecm(H, ect_mkt, ect_hml, train_cutoff)

        cer_dist = bootstrap_cer_year(cer_fit, H.index, year, cfg)
        fecm_dist = bootstrap_fecm_year(fecm_fit, ect_mkt, ect_hml, year, cfg)
        realized = realized_annual_return(H, year)

        results[year] = {
            "cer_dist": cer_dist,
            "fecm_dist": fecm_dist,
            "realized": realized,
            "cer_var10": np.percentile(cer_dist, cfg.var_quantile * 100),
            "fecm_var10": np.percentile(fecm_dist, cfg.var_quantile * 100),
        }
    return results


def build_var_table(results_by_period: dict) -> pd.DataFrame:
    """results_by_period: {label: {year: {...}}} -> flat summary table."""
    rows = []
    for label, results in results_by_period.items():
        for year, r in results.items():
            rows.append({
                "Period": label,
                "Year": year,
                "Realized": r["realized"],
                "CER 10% VaR": r["cer_var10"],
                "FECM 10% VaR": r["fecm_var10"],
            })
    return pd.DataFrame(rows)
