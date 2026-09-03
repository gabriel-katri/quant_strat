"""
Assemble the quarterly factor-specific-driver dataset (FF5 factors + the 6
original ad hoc macro signals, one-quarter-lagged).

Run with: python main_factor_specific.py
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

import data_factor_specific as data_fs_mod
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

    _section("Downloading FF5 factors + 6 factor-specific macro signals")
    df = data_fs_mod.load_dataset(cfg)
    print(f"Assembled {len(df)} quarterly observations, {df.index.min().date()} to {df.index.max().date()}")
    print(f"Columns: {list(df.columns)}")
    print("Driver mapping (config.FACTOR_SPECIFIC_DRIVERS):")
    for f, drivers in cfg.factor_specific_drivers.items():
        print(f"  {f}: {drivers}")

    _section("First rows")
    print(df.head(6).to_string())

    _section("NaN check")
    nan_counts = df.isna().sum()
    print("No NaNs." if nan_counts.sum() == 0 else nan_counts[nan_counts > 0].to_string())

    out_path = os.path.join(cfg.results_dir, cfg.factor_specific_output_csv)
    df.to_csv(out_path)
    print(f"\nSaved dataset to {os.path.abspath(out_path)}")


if __name__ == "__main__":
    sys.exit(main())
