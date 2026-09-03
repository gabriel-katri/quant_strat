"""
Build the beta-neutral and unconstrained factor-timing portfolios from the
rolling-Lasso predicted returns, plus a static equal-weight benchmark, and
compare Sharpe ratio / CER / max drawdown / turnover.

Requires results/rolling_lasso_dataset.csv and results/lasso_predictions.csv
-- run main.py then train_lasso.py first if they don't exist yet.

Benchmark note: the equal-weight benchmark uses the same 5-factor universe
as the Unconstrained portfolio (MKT_RF, SMB, HML, RMW, CMA), rebalanced
back to 1/N every quarter.

Run with: python run_factor_timing.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

import factor_timing as ft_mod
import plots as plots_mod
from config import Config

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def _section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    cfg = Config()
    dataset_path = os.path.join(cfg.results_dir, cfg.output_csv)
    pred_path = os.path.join(cfg.results_dir, "lasso_predictions.csv")
    for path in (dataset_path, pred_path):
        if not os.path.exists(path):
            sys.exit(f"{path} not found -- run `python main.py` and `python train_lasso.py` first.")

    dataset = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    predictions = pd.read_csv(pred_path, header=[0, 1], index_col=0, parse_dates=True)
    print(f"Loaded {len(predictions)} OOS quarters, {predictions.index.min().date()} to {predictions.index.max().date()}")

    portfolios = {}

    _section(f"Beta-neutral portfolio ({', '.join(cfg.beta_neutral_factors)})")
    w_bn = ft_mod.build_timing_weights(dataset, predictions, cfg.beta_neutral_factors, cfg)
    portfolios["Beta-neutral"] = ft_mod.summarize_portfolio("Beta-neutral", w_bn, dataset, cfg)

    _section(f"Unconstrained portfolio ({', '.join(cfg.unconstrained_factors)})")
    w_unc = ft_mod.build_timing_weights(dataset, predictions, cfg.unconstrained_factors, cfg)
    portfolios["Unconstrained"] = ft_mod.summarize_portfolio("Unconstrained", w_unc, dataset, cfg)

    _section(f"Benchmark: equal-weight, quarterly rebalanced ({', '.join(cfg.unconstrained_factors)})")
    w_bench = ft_mod.equal_weight_benchmark(predictions.index, cfg.unconstrained_factors)
    portfolios["Benchmark (EW)"] = ft_mod.summarize_portfolio("Benchmark (EW)", w_bench, dataset, cfg)

    _section("Summary table")
    summary = pd.DataFrame([p["metrics"] for p in portfolios.values()]).set_index("Portfolio")
    print(summary.round(3).to_string())

    _section("Generating plots")
    returns_by_portfolio = {name: p["returns"] for name, p in portfolios.items()}
    cum_path = plots_mod.plot_cumulative_returns(returns_by_portfolio, cfg)
    dd_path = plots_mod.plot_drawdown(returns_by_portfolio, cfg)
    print(f"Saved {cum_path}")
    print(f"Saved {dd_path}")

    _section("Saving results")
    os.makedirs(cfg.results_dir, exist_ok=True)

    summary_path = os.path.join(cfg.results_dir, "factor_timing_summary.csv")
    returns_path = os.path.join(cfg.results_dir, "factor_timing_returns.csv")
    weights_path = os.path.join(cfg.results_dir, "factor_timing_weights.csv")

    summary.to_csv(summary_path)
    pd.concat(returns_by_portfolio, axis=1).to_csv(returns_path)
    pd.concat({name: p["weights"] for name, p in portfolios.items()}, axis=1).to_csv(weights_path)

    print(f"Saved {summary_path}")
    print(f"Saved {returns_path}")
    print(f"Saved {weights_path}")


if __name__ == "__main__":
    sys.exit(main())
