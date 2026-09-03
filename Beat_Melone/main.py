"""
Assemble the quarterly Fama-French 5 factors + Melone (2022) macro-driver
dataset for the rolling-Lasso project (Jan 1990 - latest available).

Run with: python main.py
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

import data as data_mod
from config import Config

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v: .4f}")


def _section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.results_dir, exist_ok=True)

    _section("Downloading FF5 factors + macro drivers")
    df = data_mod.load_dataset(cfg)
    print(f"Assembled {len(df)} quarterly observations, {df.index.min().date()} to {df.index.max().date()}")
    print(f"Columns: {list(df.columns)}")
    print("Note: macro columns are lagged 1 quarter vs. factor returns "
          "(row t pairs r_t with x_{t-1}); LIQUIDITY_PS is real Pastor-Stambaugh "
          "traded liquidity (LIQ_V) through 2019Q4, spliced with a -TEDRATE proxy "
          "for 2020Q1 on -- see build_liquidity_driver() in data.py.")

    _section("First rows")
    print(df.head(10).to_string())

    _section("Descriptive statistics")
    print(df.describe().to_string())

    _section("NaN check")
    nan_counts = df.isna().sum()
    if nan_counts.sum() == 0:
        print("No NaNs in the final assembled dataset.")
    else:
        print(nan_counts[nan_counts > 0].to_string())

    _section("Macro signal correlation matrix (redundancy check)")
    macro_corr = df[cfg.macro_columns].corr()
    print(macro_corr.round(3).to_string())

    out_path = os.path.join(cfg.results_dir, cfg.output_csv)
    df.to_csv(out_path)
    print(f"\nSaved dataset to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    sys.exit(main())
