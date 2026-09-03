"""
Diagnostic (read-only -- does NOT modify config.py, data.py, or any pipeline
file): real Pastor-Stambaugh liquidity + sample-start sensitivity.

Step 1: builds the real LIQ_V traded liquidity factor (Chicago Booth
Fama-Miller mirror, https://research.chicagobooth.edu/.../liq_data_1962_2019.txt),
replacing the -TEDRATE proxy, and reports the earliest quarter where oil,
GDPPOT, term spread, real liquidity, and all 5 FF factors are simultaneously
available.

Step 2: runs the Johansen test for all 5 factors at 3 candidate sample
starts (earliest full-coverage quarter, 1975Q1, 1990Q1 [current default]),
holding the END date fixed at the real-liquidity cutoff (2019Q4) across all
three so only the start varies -- isolating the effect of the start date
from the effect of sample length. Reports trace stat / 95% CV / distance
for each candidate, to check whether pre-1975 data degrades cointegration
evidence here the way it did in the sibling mtft project (thin/survivorship
-biased Compustat coverage pre-1975 for CMA/HML; administratively-set,
non-market WTI oil prices pre-mid-1970s for all 5 factors via the shared
oil driver).

Run with: python diagnose_sample_start.py
"""

from __future__ import annotations

import re
import sys
import warnings

import numpy as np
import pandas as pd
import requests

import data as data_mod
import melone_cointegration as coint_mod
from config import Config

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v: .3f}")

LIQ_URL = "https://research.chicagobooth.edu/-/media/research/famamiller/data/liq_data_1962_2019.txt"


def _section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def fetch_real_liquidity_quarterly() -> pd.Series:
    """LIQ_V (Pastor & Stambaugh 2003's traded liquidity factor), quarterly sum
    of the monthly series. -99 is the file's missing-value sentinel for months
    before the traded factor's construction (pre-1968)."""

    def _fetch() -> pd.Series:
        resp = requests.get(LIQ_URL, timeout=30)
        resp.raise_for_status()
        lines = [ln for ln in resp.text.splitlines() if ln.strip() and not ln.startswith("%")]
        rows = [re.split(r"\s+", ln.strip()) for ln in lines]
        df = pd.DataFrame(rows, columns=["month", "agg_liq", "innov_liq", "liq_v"])
        df["liq_v"] = pd.to_numeric(df["liq_v"], errors="coerce")
        df["month"] = pd.to_datetime(df["month"], format="%Y%m") + pd.offsets.MonthEnd(0)
        s = df.set_index("month")["liq_v"]
        return s[s != -99].dropna()

    monthly = data_mod._cached("liq_v_monthly", _fetch)
    return monthly.resample(data_mod._QUARTER_END).sum().rename("LIQUIDITY_real_PS")


def main() -> None:
    cfg = Config()

    _section("Step 1: real liquidity (LIQ_V) coverage")
    liq_q = fetch_real_liquidity_quarterly()
    print(f"LIQUIDITY_real_PS: {liq_q.index.min().date()} to {liq_q.index.max().date()}  ({len(liq_q)} quarters)")

    oil_q = data_mod.build_oil_driver(cfg)
    pot_q = data_mod.build_potential_output_driver(cfg)
    term_q = data_mod.build_term_spread_driver(cfg)
    ff5_q = data_mod.ff5_monthly_to_quarterly(data_mod.download_ff5_monthly())[cfg.factor_columns]

    for name, s in [("OIL_WTI_LOGRET", oil_q), ("POT_GDP_GROWTH", pot_q),
                     ("TERM_SPREAD_10Y3M", term_q), ("LIQUIDITY_real_PS", liq_q)]:
        print(f"{name}: {s.index.min().date()} to {s.index.max().date()}")
    print(f"FF5 (all 5 factors): {ff5_q.dropna().index.min().date()} to {ff5_q.dropna().index.max().date()}")

    macro_full = pd.concat([oil_q, pot_q, term_q, liq_q], axis=1)
    common = pd.concat([ff5_q, macro_full], axis=1).dropna(how="any")
    earliest_common = common.index.min()
    latest_common = common.index.max()
    print(f"\nEarliest quarter with ALL 4 drivers + all 5 FF factors: {earliest_common.date()}")
    print(f"Latest quarter with ALL 4 drivers + all 5 FF factors:   {latest_common.date()} "
          f"(bounded by LIQUIDITY_real_PS's Dec-2019 cutoff)")

    _section("Step 2: Johansen test sensitivity to sample start (end fixed at real-liquidity cutoff)")
    fixed_end = liq_q.index.max()
    candidates = {
        f"Earliest full coverage ({earliest_common.date()})": earliest_common,
        "1975Q1": pd.Timestamp("1975-01-01"),
        "1990Q1 (current default)": pd.Timestamp("1990-01-01"),
    }

    for label, start in candidates.items():
        _section(f"Candidate: {label}  ->  window [{start.date()}, {fixed_end.date()}]")

        oil_w = oil_q[(oil_q.index >= start) & (oil_q.index <= fixed_end)]
        pot_w = pot_q[(pot_q.index >= start) & (pot_q.index <= fixed_end)]
        term_w = term_q[(term_q.index >= start) & (term_q.index <= fixed_end)]
        liq_w = liq_q[(liq_q.index >= start) & (liq_q.index <= fixed_end)]
        ff5_w = ff5_q[(ff5_q.index >= start) & (ff5_q.index <= fixed_end)]

        macro_levels_w = pd.concat([oil_w, pot_w, term_w, liq_w], axis=1).dropna(how="any").cumsum()
        factor_prices_w = pd.concat({f: np.log1p(ff5_w[f]) for f in cfg.unconstrained_factors}, axis=1)
        factor_prices_w = factor_prices_w.reindex(macro_levels_w.index).cumsum()

        print(f"N obs = {len(macro_levels_w)}")

        joh_results = {
            f: coint_mod.run_johansen_test(factor_prices_w[f], macro_levels_w, cfg)
            for f in cfg.unconstrained_factors
        }
        rows = []
        for f in cfg.unconstrained_factors:
            result, lag, nobs = joh_results[f]
            trace0, cv95_0 = result.lr1[0], result.cvt[0, 1]
            rank = int((result.lr1 > result.cvt[:, 1]).sum())
            rows.append({"Factor": f, "VAR lag (AIC)": lag, "N obs": nobs,
                          "Trace stat (r=0)": trace0, "95% CV": cv95_0,
                          "Distance to CV": trace0 - cv95_0, "Reject r=0 @95%": bool(trace0 > cv95_0),
                          "Rank": rank})
        table = pd.DataFrame(rows).set_index("Factor")
        print(table.round(2).to_string())
        n_reject = int(table["Reject r=0 @95%"].sum())
        print(f"-> {n_reject}/5 factors reject r=0 at 95%")


if __name__ == "__main__":
    sys.exit(main())
