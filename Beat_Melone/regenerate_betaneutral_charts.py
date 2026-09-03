"""
Regenerates results/factor_timing_cumulative_returns.png and
results/factor_timing_drawdown.png from the exact same code path/results
objects as Table 3 in generate_paper_tables.py.

Explicitly does NOT use run_factor_specific_comparison.py or its outputs
(factor_specific_cumulative_returns.png, factor_specific_drawdown.png) --
those are a separate, stale (Jul 21, predates the Jul 27 dataset rebuild),
unrelated experiment (per-factor macro drivers) and are out of scope here.

Matches Table 3's exact setup: Beta-neutral universe (excludes MKT_RF),
full 2000Q1-2026Q2 OOS period, main pipeline's default window (expanding,
1975Q1-1999Q4 in-sample), gamma=5. Five series: Static, Benchmark (EW),
LASSO (matched Sigma -- expanding-window covariance, same convention as
Static/Kalman, isolating the covariance-window confound), Kalman (RW),
Kalman (MR).

Before plotting, sanity-checks Static - Unconstrained's max drawdown from
this same code path against the paper's verified Table 1 number (-24.23%,
trough 2002-09-30) and aborts without plotting anything if it doesn't
match within tolerance.

Overwrites the two filenames in place so old bookmarks/references still
resolve to current results.

Requires results/rolling_lasso_dataset.csv and results/lasso_predictions.csv.

Run with: python regenerate_betaneutral_charts.py
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

import factor_timing as ft_mod
import melone_construct as construct_mod
import melone_ect as ect_mod
import melone_kalman as kalman_mod
import melone_oos_static as oos_static_mod
import melone_portfolio as port_mod
import plots as plots_mod
from config import Config

warnings.filterwarnings("ignore")

EXPECTED_STATIC_UNCONSTRAINED_MDD_PCT = -24.23
EXPECTED_STATIC_UNCONSTRAINED_TROUGH = "2002-09-30"
MDD_TOLERANCE_PCT = 0.05


def main() -> None:
    cfg = Config()
    dataset_path = os.path.join(cfg.results_dir, cfg.output_csv)
    lasso_pred_path = os.path.join(cfg.results_dir, "lasso_predictions.csv")
    for path in (dataset_path, lasso_pred_path):
        if not os.path.exists(path):
            sys.exit(f"{path} not found -- run main.py and train_lasso.py first.")

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    lasso_predictions = pd.read_csv(lasso_pred_path, header=[0, 1], index_col=0, parse_dates=True)
    factor_prices, macro_levels = construct_mod.build_levels(df, cfg)

    static_oos = oos_static_mod.run_all_oos(df, factor_prices, macro_levels, cfg)
    static_predictions = port_mod.to_wide_predictions(static_oos, cfg.unconstrained_factors)

    lr_models = ect_mod.estimate_all_long_run(factor_prices, macro_levels, cfg)
    beta_static_by_factor = ect_mod.get_beta_static_series(lr_models, cfg)

    kalman_rw = kalman_mod.run_all_kalman(factor_prices, macro_levels, cfg)
    kalman_rw_oos = kalman_mod.run_all_oos(df, kalman_rw, cfg)
    kalman_rw_predictions = port_mod.to_wide_predictions(kalman_rw_oos, cfg.unconstrained_factors)

    kalman_mr = kalman_mod.run_all_kalman_mean_reverting(factor_prices, macro_levels, beta_static_by_factor, cfg)
    kalman_mr_oos = kalman_mod.run_all_oos(df, kalman_mr, cfg)
    kalman_mr_predictions = port_mod.to_wide_predictions(kalman_mr_oos, cfg.unconstrained_factors)

    # ---------------------------------------------------------- Sanity check (Unconstrained, before touching Beta-neutral)
    w_static_unc = ft_mod.build_expanding_timing_weights(df, static_predictions, cfg.unconstrained_factors, cfg)
    static_unc = ft_mod.summarize_portfolio("Static - Unconstrained", w_static_unc, df, cfg)
    dd = ft_mod.drawdown_series(static_unc["returns"])
    mdd_pct, trough_date = dd.min() * 100, dd.idxmin().date()

    print(f"Sanity check -- Static - Unconstrained max drawdown (this code path): "
          f"{mdd_pct:.2f}% on {trough_date}")
    print(f"Expected (Table 1 / paper Results section): "
          f"{EXPECTED_STATIC_UNCONSTRAINED_MDD_PCT}% on {EXPECTED_STATIC_UNCONSTRAINED_TROUGH}")

    if abs(mdd_pct - EXPECTED_STATIC_UNCONSTRAINED_MDD_PCT) > MDD_TOLERANCE_PCT or \
            str(trough_date) != EXPECTED_STATIC_UNCONSTRAINED_TROUGH:
        sys.exit("Sanity check FAILED -- underlying data disagrees with the paper's verified Table 1 number. "
                 "Aborting without plotting anything; investigate before re-running.")
    print("Sanity check PASSED -- proceeding to build the Beta-neutral charts.\n")

    factors = cfg.beta_neutral_factors
    w_static = ft_mod.build_expanding_timing_weights(df, static_predictions, factors, cfg)
    w_kalman_rw = ft_mod.build_expanding_timing_weights(df, kalman_rw_predictions, factors, cfg)
    w_kalman_mr = ft_mod.build_expanding_timing_weights(df, kalman_mr_predictions, factors, cfg)
    w_lasso_matched = ft_mod.build_expanding_timing_weights(df, lasso_predictions, factors, cfg)
    w_bench = ft_mod.equal_weight_benchmark(lasso_predictions.index, factors)

    headline = {
        "Static": ft_mod.summarize_portfolio("Static", w_static, df, cfg)["returns"],
        "Benchmark (EW)": ft_mod.summarize_portfolio("Benchmark (EW)", w_bench, df, cfg)["returns"],
        "LASSO (matched Sigma)": ft_mod.summarize_portfolio(
            "LASSO (matched Sigma)", w_lasso_matched, df, cfg)["returns"],
        "Kalman (RW)": ft_mod.summarize_portfolio("Kalman (RW)", w_kalman_rw, df, cfg)["returns"],
        "Kalman (MR)": ft_mod.summarize_portfolio("Kalman (MR)", w_kalman_mr, df, cfg)["returns"],
    }
    colors = [plots_mod.STRATEGY_COLORS[name] for name in headline]

    cum_path = plots_mod.plot_cumulative_returns(
        headline, cfg, colors=colors, filename="factor_timing_cumulative_returns",
        title="Static vs. Kalman (RW/MR) vs. LASSO (matched Sigma) vs. Benchmark: Cumulative Returns (Beta-neutral)")
    dd_path = plots_mod.plot_drawdown(
        headline, cfg, colors=colors, filename="factor_timing_drawdown",
        title="Static vs. Kalman (RW/MR) vs. LASSO (matched Sigma) vs. Benchmark: Drawdown (Beta-neutral)")
    print(f"Saved {cum_path}")
    print(f"Saved {dd_path}")


if __name__ == "__main__":
    sys.exit(main())
