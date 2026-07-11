"""
End-to-end TSMOM pipeline: download data, build signals, backtest, run
factor regressions and the TSMOM/XSMOM decomposition, generate plots, and
print a full summary of results.

Run with: python main.py
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

import backtest as bt
import data as data_mod
import decomposition as decomp
import factors as fac_mod
import plots as plots_mod
import signals as sig_mod
from config import Config
from volatility import ewma_ex_ante_vol

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v: .4f}")


def _section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.results_dir, exist_ok=True)

    _section("1/6  Downloading data from yfinance")
    dataset = data_mod.load_data(cfg)
    excess_returns = dataset["excess_returns"]
    n_instruments = excess_returns.notna().any().sum()
    print(f"Downloaded {n_instruments}/{len(cfg.tickers)} instruments, "
          f"{excess_returns.index.min().date()} to {excess_returns.index.max().date()}")

    _section("2/6  Estimating ex-ante volatility (EWMA, Eq. 1)")
    sigma_daily_lagged = ewma_ex_ante_vol(excess_returns, cfg)
    print(f"Ex-ante vol estimated for {sigma_daily_lagged.notna().any().sum()} instruments.")

    _section("3/6  Running TSMOM backtest (Eq. 5) + passive-long benchmark")
    results = bt.run_tsmom_backtest(excess_returns, sigma_daily_lagged, cfg)
    tsmom_returns = results["tsmom_portfolio_returns"]
    passive_returns = results["passive_portfolio_returns"]
    print(f"TSMOM portfolio: {len(tsmom_returns)} monthly observations "
          f"({tsmom_returns.index.min().date()} to {tsmom_returns.index.max().date()})")

    tsmom_stats = bt.performance_summary(tsmom_returns, cfg)
    passive_stats = bt.performance_summary(passive_returns, cfg)

    _section("4/6  Factor regressions (Eq. 4): TSMOM ~ MKT + SMB + HML + UMD")
    try:
        ff_factors = fac_mod.download_ff_factors(cfg.data_start, cfg.data_end)
        four_factor = fac_mod.run_four_factor_regression(tsmom_returns, ff_factors)
        smile_reg = fac_mod.run_smile_regression(tsmom_returns, ff_factors)

        four_factor_table = fac_mod.summarize_regression(four_factor, cfg.months_per_year, "4-Factor")
        smile_table = fac_mod.summarize_regression(smile_reg, cfg.months_per_year, "MKT+MKT^2")

        print("\nFour-factor regression (Newey-West t-stats):")
        print(four_factor_table.to_string(index=False))
        print(f"R^2 = {four_factor['r_squared']:.4f}   N = {four_factor['n_obs']}")

        print("\nMKT + MKT^2 (smile) regression (Newey-West t-stats):")
        print(smile_table.to_string(index=False))
        print(f"R^2 = {smile_reg['r_squared']:.4f}   N = {smile_reg['n_obs']}")
    except Exception as e:
        print(f"[WARN] Factor regression skipped -- could not download Fama-French data: {e}")
        four_factor_table = None
        smile_table = None

    _section("5/6  TSMOM vs XSMOM decomposition (Eq. 6-7)")
    normalized_returns = decomp.compute_normalized_returns(excess_returns, sigma_daily_lagged, cfg)
    decomposition_result = decomp.decompose_tsmom(normalized_returns)

    print(f"Auto-covariance component (monthly):  {decomposition_result['auto_covariance_component']:.6f}  "
          f"(annualized: {decomposition_result['auto_covariance_annualized']:.6f})")
    print(f"Cross-serial component (monthly):     {decomposition_result['cross_serial_component']:.6f}  "
          f"(annualized: {decomposition_result['cross_serial_annualized']:.6f})")
    print(f"Mean component (monthly):             {decomposition_result['mean_component']:.6f}  "
          f"(annualized: {decomposition_result['mean_component_annualized']:.6f})")
    print(f"\nDominant component: {decomposition_result['dominant_component']}")

    xsmom_results = decomp.run_xsmom_backtest(excess_returns, sigma_daily_lagged, cfg)
    xsmom_returns = xsmom_results["xsmom_portfolio_returns"]
    comparison = decomp.compare_tsmom_xsmom(tsmom_returns, xsmom_returns, cfg)
    print("\nTSMOM vs XSMOM performance comparison:")
    print(comparison.to_string())

    _section("6/6  Generating plots -> results/")
    benchmark_monthly_excess = results["tsmom"]["monthly_excess_returns"][cfg.benchmark_ticker]
    plot_outputs = plots_mod.generate_all_plots(
        normalized_returns=normalized_returns,
        tsmom_returns=tsmom_returns,
        passive_returns=passive_returns,
        benchmark_monthly_excess_returns=benchmark_monthly_excess,
        tsmom_instrument_returns=results["tsmom_instrument_returns"],
        cfg=cfg,
    )
    print(f"Saved 6 plots to {os.path.abspath(cfg.results_dir)}/")

    _section("SUMMARY: TSMOM vs Passive Long")
    summary_table = pd.DataFrame({"TSMOM": tsmom_stats, "Passive Long": passive_stats})
    print(summary_table.to_string())

    _section("SUMMARY: Per-Instrument Sharpe Ratio (TSMOM)")
    print(plot_outputs["sharpe_by_instrument"].sort_values(ascending=False).to_string())

    summary_table.to_csv(os.path.join(cfg.results_dir, "summary_metrics.csv"))
    plot_outputs["sharpe_by_instrument"].to_csv(os.path.join(cfg.results_dir, "sharpe_by_instrument.csv"))
    comparison.to_csv(os.path.join(cfg.results_dir, "tsmom_vs_xsmom.csv"))
    if four_factor_table is not None:
        four_factor_table.to_csv(os.path.join(cfg.results_dir, "four_factor_regression.csv"), index=False)
        smile_table.to_csv(os.path.join(cfg.results_dir, "smile_regression.csv"), index=False)

    print(f"\nAll results saved to {os.path.abspath(cfg.results_dir)}/")


if __name__ == "__main__":
    sys.exit(main())
