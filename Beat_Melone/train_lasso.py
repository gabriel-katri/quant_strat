"""
Rolling-window Lasso factor return prediction, from the already-assembled
quarterly dataset (results/rolling_lasso_dataset.csv -- run main.py first
if it doesn't exist yet).

Run with: python train_lasso.py
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

import lasso as lasso_mod
import plots as plots_mod
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
    dataset_path = os.path.join(cfg.results_dir, cfg.output_csv)
    if not os.path.exists(dataset_path):
        sys.exit(f"Dataset not found at {dataset_path} -- run `python main.py` first to build it.")

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    n_train_windows = len(df) - cfg.rolling_window_periods
    print(f"Loaded {len(df)} quarterly observations, {df.index.min().date()} to {df.index.max().date()}")
    print(f"Rolling window: {cfg.rolling_window_periods} quarters -> {n_train_windows} OOS predictions per factor "
          f"({df.index[cfg.rolling_window_periods].date()} to {df.index.max().date()})")
    if n_train_windows <= 0:
        sys.exit("Not enough observations for even one rolling window.")

    _section("Running rolling-window Lasso (LassoCV, TimeSeriesSplit, standardized in-window)")
    results = lasso_mod.run_all_factors(df, cfg)
    for factor in cfg.lasso_target_factors:
        print(f"  {factor}: {len(results[factor]['predictions'])} predictions done")

    _section("Out-of-sample R^2 by factor")
    r2_table = lasso_mod.build_r2_table(results, cfg)
    print(r2_table.round(3).to_string())

    _section("Hit rate by factor (% of quarters sign(predicted) == sign(actual))")
    hit_rate_table = lasso_mod.build_hit_rate_table(results, cfg)
    print(hit_rate_table.round(3).to_string())

    _section("Sample of predictions vs. actuals (MKT_RF, last 10 rows)")
    print(results["MKT_RF"]["predictions"].tail(10).to_string())

    _section("Sample of coefficients (MKT_RF, last 5 rows)")
    print(results["MKT_RF"]["coefficients"].tail(5).to_string())

    _section("Generating plots")
    coef_plot_path = plots_mod.plot_coefficients_over_time(results, cfg)
    pred_plot_path = plots_mod.plot_predicted_vs_actual(results, cfg)
    print(f"Saved {coef_plot_path}")
    print(f"Saved {pred_plot_path}")

    _section("Saving results")
    os.makedirs(cfg.results_dir, exist_ok=True)

    all_predictions = pd.concat(
        {factor: results[factor]["predictions"] for factor in cfg.lasso_target_factors}, axis=1
    )
    all_coefficients = pd.concat(
        {factor: results[factor]["coefficients"] for factor in cfg.lasso_target_factors}, axis=1
    )

    pred_path = os.path.join(cfg.results_dir, "lasso_predictions.csv")
    coef_path = os.path.join(cfg.results_dir, "lasso_coefficients.csv")
    r2_path = os.path.join(cfg.results_dir, "lasso_oos_r2.csv")
    hit_rate_path = os.path.join(cfg.results_dir, "lasso_hit_rate.csv")

    all_predictions.to_csv(pred_path)
    all_coefficients.to_csv(coef_path)
    r2_table.to_csv(r2_path)
    hit_rate_table.to_csv(hit_rate_path)

    print(f"Saved {pred_path}")
    print(f"Saved {coef_path}")
    print(f"Saved {r2_path}")
    print(f"Saved {hit_rate_path}")


if __name__ == "__main__":
    sys.exit(main())
