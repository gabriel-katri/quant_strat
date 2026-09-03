"""
Kalman portfolio backtest: sensitivity to q (state noise).

Rebuilds the Step 2 Kalman expanding-window OOS forecasts and the
Unconstrained/Beta-neutral portfolios for three specifications of q (the
shared state-noise variance): the MLE estimate from calibrate_q_r, and q
manually divided by 10 and by 100. diagnose_kalman.py showed the MLE q
tracks ln F_t so closely the ECT is nearly flat for 4 of 5 factors, and
that dividing q recovers a static-ECT-like signal -- this script checks
whether that translates into a portfolio difference. r and the per-column
design scale stay fixed at their MLE-calibrated values throughout; only q
changes.

Requires results/rolling_lasso_dataset.csv -- run main.py first if it
doesn't exist.

Run with: python backtest_kalman_q_sensitivity.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

import factor_timing as ft_mod
import lasso as lasso_mod
import melone_construct as construct_mod
import melone_kalman as kalman_mod
import melone_oos_static as oos_static_mod
import melone_portfolio as port_mod
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


def _mean_hit_rate(predictions_wide: pd.DataFrame, factors: list[str]) -> float:
    """Unweighted average, across factors, of the sign-match hit rate."""
    rates = [lasso_mod.hit_rate(predictions_wide[(f, "actual")], predictions_wide[(f, "predicted")]) for f in factors]
    return float(np.mean(rates)) * 100.0


def main() -> None:
    cfg = Config()
    dataset_path = os.path.join(cfg.results_dir, cfg.output_csv)
    if not os.path.exists(dataset_path):
        sys.exit(f"{dataset_path} not found -- run `python main.py` first.")
    os.makedirs(cfg.results_dir, exist_ok=True)

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    factor_prices, macro_levels = construct_mod.build_levels(df, cfg)
    universes = {"Unconstrained": cfg.unconstrained_factors, "Beta-neutral": cfg.beta_neutral_factors}

    _section("Calibrating Kalman (r, q via MLE) and re-filtering with q/10, q/100")
    kalman_results = kalman_mod.run_all_kalman(factor_prices, macro_levels, cfg)
    ect_variants = {"Kalman (q MLE)": {}, "Kalman (q/10)": {}, "Kalman (q/100)": {}}
    for f in cfg.unconstrained_factors:
        params, scale = kalman_results[f]["params"], kalman_results[f]["scale"]
        ect_variants["Kalman (q MLE)"][f] = kalman_results[f]["ect"]
        for divisor, label in [(10.0, "Kalman (q/10)"), (100.0, "Kalman (q/100)")]:
            _beta_t, ect = kalman_mod.filter_with_q_divisor(
                factor_prices[f], macro_levels, params, scale, divisor, cfg)
            ect_variants[label][f] = ect
        print(f"{f}: q_MLE={kalman_results[f]['q']:.6f}  q/10={kalman_results[f]['q'] / 10:.6f}  "
              f"q/100={kalman_results[f]['q'] / 100:.6f}")

    _section("Expanding-window predictive OOS for each q variant + Static")
    predictions_by_method = {}
    for label, ect_by_factor in ect_variants.items():
        oos_by_factor = {f: kalman_mod.run_oos_for_factor(df[f], ect_by_factor[f], cfg)
                          for f in cfg.unconstrained_factors}
        predictions_by_method[label] = port_mod.to_wide_predictions(oos_by_factor, cfg.unconstrained_factors)
        print(f"{label}: done ({len(predictions_by_method[label])} OOS quarters)")

    static_oos = oos_static_mod.run_all_oos(df, factor_prices, macro_levels, cfg)
    predictions_by_method["Static"] = port_mod.to_wide_predictions(static_oos, cfg.unconstrained_factors)
    print(f"Static: done ({len(predictions_by_method['Static'])} OOS quarters)")

    _section("Building portfolios (expanding-window covariance, vol-targeted)")
    portfolios = {}
    for method in ["Kalman (q MLE)", "Kalman (q/10)", "Kalman (q/100)", "Static"]:
        preds = predictions_by_method[method]
        for uni_name, factors in universes.items():
            name = f"{method} - {uni_name}"
            w = ft_mod.build_expanding_timing_weights(df, preds, factors, cfg)
            summary = ft_mod.summarize_portfolio(name, w, df, cfg)
            summary["metrics"]["Hit rate (%)"] = _mean_hit_rate(preds, factors)
            portfolios[name] = summary

    oos_dates = predictions_by_method["Static"].index
    for uni_name, factors in universes.items():
        name = f"Benchmark (EW) - {uni_name}"
        w_bench = ft_mod.equal_weight_benchmark(oos_dates, factors)
        summary = ft_mod.summarize_portfolio(name, w_bench, df, cfg)
        summary["metrics"]["Hit rate (%)"] = float("nan")
        portfolios[name] = summary

    _section("Comparison table: Kalman (q MLE / q/10 / q/100) vs Static vs Benchmark")
    comparison = pd.DataFrame([p["metrics"] for p in portfolios.values()]).set_index("Portfolio")
    comparison = comparison[["Sharpe (ann.)", "CER (%, ann., gamma=5)", "Max drawdown (%)",
                              "Hit rate (%)", "Avg quarterly turnover"]]
    print(comparison.round(3).to_string())
    comp_path = os.path.join(cfg.results_dir, "melone_kalman_q_sensitivity_comparison.csv")
    comparison.to_csv(comp_path)
    print(f"\nSaved {comp_path}")

    _section("Plot: cumulative returns, 3 Kalman q-variants (Unconstrained)")
    headline = {
        "q (MLE)": portfolios["Kalman (q MLE) - Unconstrained"]["returns"],
        "q/10": portfolios["Kalman (q/10) - Unconstrained"]["returns"],
        "q/100": portfolios["Kalman (q/100) - Unconstrained"]["returns"],
    }
    colors = [plots_mod.Q_VARIANT_COLORS[name] for name in headline]
    cum_path = plots_mod.plot_cumulative_returns(
        headline, cfg, colors=colors, filename="melone_kalman_q_sensitivity_cumulative_returns",
        title="Kalman Portfolio: Cumulative Returns by q (Unconstrained)")
    print(f"Saved {cum_path}")

    returns_wide = pd.concat({name: p["returns"] for name, p in portfolios.items()}, axis=1)
    returns_path = os.path.join(cfg.results_dir, "melone_kalman_q_sensitivity_returns.csv")
    returns_wide.to_csv(returns_path)
    print(f"Saved {returns_path}")


if __name__ == "__main__":
    sys.exit(main())
