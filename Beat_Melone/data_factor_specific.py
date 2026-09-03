"""
Download and assemble the quarterly dataset for the factor-specific-driver
model: the original 6 ad hoc FRED macro signals (CPI_YoY, UNRATE, VIX,
PMI_proxy_INDPRO_YoY, TERM_SPREAD_10Y2Y, REAL_M2_YoY), aggregated from
monthly to quarterly (end-of-quarter value of each signal) and lagged one
quarter relative to the FF5 factor returns -- same lag convention as
data.py's Melone-driver dataset. Each factor later picks its own 3-driver
subset from these 6 (see config.FACTOR_SPECIFIC_DRIVERS); this module only
builds the full 6-column quarterly panel.

Reuses download_ff5_monthly/ff5_monthly_to_quarterly and the FRED
cache/download plumbing from data.py.
"""

from __future__ import annotations

import pandas as pd

from config import Config
from data import _MONTH_END, _QUARTER_END, _download_fred_series, download_ff5_monthly, ff5_monthly_to_quarterly


def build_macro_signals_monthly(cfg: Config) -> pd.DataFrame:
    """The original 6 monthly macro signals, raw (not yet lagged)."""
    cpi = _download_fred_series(cfg.fred_series_factor_specific["cpi"]).resample(_MONTH_END).last()
    unrate = _download_fred_series(cfg.fred_series_factor_specific["unrate"]).resample(_MONTH_END).last()
    vix = _download_fred_series(cfg.fred_series_factor_specific["vix"]).resample(_MONTH_END).mean()
    indpro = _download_fred_series(cfg.fred_series_factor_specific["indpro"]).resample(_MONTH_END).last()
    term_spread = _download_fred_series(cfg.fred_series_factor_specific["term_spread"]).resample(_MONTH_END).last()
    m2 = _download_fred_series(cfg.fred_series_factor_specific["m2"]).resample(_MONTH_END).last()

    cpi_yoy = cpi.pct_change(12) * 100.0
    pmi_proxy_yoy = indpro.pct_change(12) * 100.0
    real_m2_level = m2 / (cpi / 100.0)  # M2 deflated by the CPI index (CPIAUCSL base = 1982-84=100)
    real_m2_yoy = real_m2_level.pct_change(12) * 100.0

    return pd.concat(
        [cpi_yoy.rename("CPI_YoY"), unrate.rename("UNRATE"), vix.rename("VIX"),
         pmi_proxy_yoy.rename("PMI_proxy_INDPRO_YoY"), term_spread.rename("TERM_SPREAD_10Y2Y"),
         real_m2_yoy.rename("REAL_M2_YoY")],
        axis=1,
    )


def build_macro_signals_quarterly(cfg: Config) -> pd.DataFrame:
    """End-of-quarter value of each of the 6 monthly signals."""
    monthly = build_macro_signals_monthly(cfg)
    return monthly.resample(_QUARTER_END).last()


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Assemble the final quarterly dataset: FF5 quarterly returns + one-quarter-lagged macro signals."""
    ff5_monthly = download_ff5_monthly()
    ff5 = ff5_monthly_to_quarterly(ff5_monthly[cfg.factor_columns])
    macro = build_macro_signals_quarterly(cfg)
    macro_lagged = macro.shift(1)

    df = pd.concat([ff5, macro_lagged[cfg.factor_specific_macro_columns]], axis=1)
    df = df[df.index >= cfg.sample_start]
    if cfg.sample_end is not None:
        df = df[df.index <= cfg.sample_end]
    df = df.dropna(how="any")
    return df
