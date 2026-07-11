"""
Step 9: Robustness checks with alternative macro drivers.

Three variants, each re-running the full Johansen -> ECT -> predictive
regression -> OOS pipeline on its own aligned sample:

  1. Gold instead of oil (GLD ETF as a proxy; FRED's LBMA gold series was
     discontinued, and Stooq now gates its CSV export behind a JS
     proof-of-work challenge that isn't scriptable here).
  2. Corporate spread (BAA - AAA) instead of the term spread.
  3. The VIX-liquidity 4-driver extended panel built in Step 0 (already
     computed there; included here for a side-by-side comparison since it
     covers a different -- shorter -- sample than the 3-driver baseline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

import cointegration as coint_mod
import construct
import ect as ect_mod
import oos as oos_mod
import predictive as pred_mod
from config import Config, FRED_CSV_TEMPLATE, FRED_SERIES
from data import _cached, _download_fred_series


def build_gold_driver() -> pd.Series:
    """Gold log returns via GLD ETF (proxy for spot gold; see module docstring)."""

    def _fetch() -> pd.Series:
        prices = yf.download("GLD", start="2000-01-01", progress=False, auto_adjust=True)["Close"]["GLD"]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        return prices

    prices = _cached("gld_prices", _fetch)
    monthly = prices.resample("ME").last()
    log_ret = np.log(monthly / monthly.shift(1)).dropna()
    return log_ret.resample("QE").sum().rename("gold")


def build_corp_spread_driver() -> pd.Series:
    """Corporate credit spread: Moody's Baa - Aaa yield, quarterly end-of-quarter."""
    baa = _download_fred_series(FRED_SERIES["baa"]).resample("QE").last()
    aaa = _download_fred_series(FRED_SERIES["aaa"]).resample("QE").last()
    return (baa - aaa).dropna().rename("corp_spread")


def _align(factor_log_returns: pd.DataFrame, macro: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = pd.concat([factor_log_returns, macro], axis=1).dropna(how="any")
    aligned = aligned[aligned.index >= cfg.sample_start]
    return aligned[factor_log_returns.columns], aligned[macro.columns]


def _pick_oos_start(index: pd.DatetimeIndex) -> list[str]:
    """A single OOS start date at ~65% into whatever sample is available."""
    pos = int(len(index) * 0.65)
    return [index[pos].strftime("%Y-%m-%d")]


def run_variant_pipeline(factor_log_returns: pd.DataFrame, macro: pd.DataFrame, cfg: Config) -> dict:
    """Run Johansen -> ECT -> predictive regression -> OOS for one macro-driver variant."""
    macro_cols = list(macro.columns)
    ret, macro_aligned = _align(factor_log_returns, macro, cfg)

    variant_cfg = Config(macro_drivers=macro_cols)
    prices = construct.build_factor_prices(ret)
    levels = construct.build_macro_driver_levels(macro_aligned)

    johansen_results = coint_mod.run_all_johansen_tests(prices, levels, variant_cfg)
    rank_summary = coint_mod.summarize_cointegration_rank(johansen_results, variant_cfg)

    lr_models = ect_mod.estimate_all_ects(prices, levels, variant_cfg)
    ects = ect_mod.get_ects(lr_models, variant_cfg)

    pred_results = pred_mod.run_all_predictive_regressions(ret, ects, variant_cfg)
    in_sample_r2 = {j: pred_results[j]["model"].rsquared for j in cfg.factors}

    oos_starts = _pick_oos_start(ret.index)
    variant_cfg.oos_start_dates = oos_starts
    oos_results = oos_mod.run_all_oos_evaluations(ret, prices, levels, variant_cfg)
    oos_r2 = {
        j: oos_mod.oos_r2_campbell_thompson(
            oos_results[oos_starts[0]][j]["actual"].values,
            oos_results[oos_starts[0]][j]["forecast_model"].values,
            oos_results[oos_starts[0]][j]["forecast_mean"].values,
        )
        for j in cfg.factors
    }

    return {
        "macro_drivers": macro_cols,
        "sample_start": ret.index.min(),
        "sample_end": ret.index.max(),
        "n_obs": len(ret),
        "rank_summary": rank_summary,
        "in_sample_r2": in_sample_r2,
        "oos_start": oos_starts[0],
        "oos_r2": oos_r2,
    }


def build_robustness_summary(variants: dict, cfg: Config) -> pd.DataFrame:
    """variants: {variant_name: variant_result_dict} -> flat comparison table."""
    rows = []
    for name, v in variants.items():
        for j in cfg.factors:
            rows.append({
                "Variant": name,
                "Factor": cfg.factor_labels[j],
                "Sample": f"{v['sample_start'].date()} to {v['sample_end'].date()} (N={v['n_obs']})",
                "Coint. Rank": v["rank_summary"].loc[cfg.factor_labels[j], "Cointegrating rank"],
                "IS R^2 (%)": v["in_sample_r2"][j] * 100,
                "OOS Start": v["oos_start"][:4],
                "OOS R^2 (%)": v["oos_r2"][j] * 100,
            })
    return pd.DataFrame(rows)
