"""
Factor-specific-driver model: each factor cointegrated with its OWN 3-driver
subset of the 6 original ad hoc macro signals (config.FACTOR_SPECIFIC_DRIVERS),
instead of Melone's shared 4-driver panel. Builds both a Static and a Kalman
version, exactly as in run_melone.py / diagnose_kalman.py, but re-run per
factor with that factor's own macro_levels slice -- the underlying single-
factor regression/cointegration/Kalman functions are already generic on
whatever `macro_levels` frame is passed in, so no changes were needed there.

Final comparison: Melone Static, Melone Kalman, Factor-specific Static,
Factor-specific Kalman, LASSO (all already trained/available), and the
equal-weight benchmark -- Beta-neutral and Unconstrained universes, full
OOS period plus the 2000-2019 / 2020-2026 sub-periods.

Requires:
  results/rolling_lasso_dataset.csv   (main.py)
  results/lasso_predictions.csv       (train_lasso.py)
  results/factor_specific_dataset.csv (main_factor_specific.py)

Run with: python run_factor_specific_comparison.py
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
import melone_cointegration as coint_mod
import melone_ect as ect_mod
import melone_kalman as kalman_mod
import melone_oos_static as oos_static_mod
import melone_portfolio as port_mod
import melone_predictive as pred_mod
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


def _save_csv(df, name: str, cfg: Config) -> str:
    path = os.path.join(cfg.results_dir, f"{name}.csv")
    df.to_csv(path)
    print(f"Saved {path}")
    return path


def _mean_hit_rate(predictions_wide: pd.DataFrame, factors: list[str]) -> float:
    rates = [lasso_mod.hit_rate(predictions_wide[(f, "actual")], predictions_wide[(f, "predicted")]) for f in factors]
    return float(np.mean(rates)) * 100.0


def run_factor_specific_static(df_fs: pd.DataFrame, factor_prices_fs: pd.DataFrame, macro_levels_fs: pd.DataFrame,
                                cfg: Config) -> tuple[dict, dict, dict]:
    """Per-factor long-run regression + Johansen + predictive regression + expanding OOS,
    each on that factor's OWN 3-driver subset. Returns (beta_rows, rank_rows, oos_by_factor)."""
    beta_rows, rank_rows, oos_by_factor = [], [], {}
    for f in cfg.unconstrained_factors:
        drivers = cfg.factor_specific_drivers[f]
        macro_f = macro_levels_fs[drivers]

        lr_model = ect_mod.estimate_long_run_regression(factor_prices_fs[f], macro_f)
        beta_rows.append({"Factor": f, "Drivers": ", ".join(drivers),
                           **dict(zip(["const", "trend"] + drivers, lr_model.params)),
                           "R2": lr_model.rsquared, "N obs": int(lr_model.nobs)})

        joh_result, lag, nobs = coint_mod.run_johansen_test(factor_prices_fs[f], macro_f, cfg)
        trace0, cv95_0 = joh_result.lr1[0], joh_result.cvt[0, 1]
        rank = int(np.sum(joh_result.lr1 > joh_result.cvt[:, 1]))
        rank_rows.append({"Factor": f, "Drivers": ", ".join(drivers), "VAR lag (AIC)": lag, "N obs": nobs,
                           "Trace stat (r=0)": trace0, "95% CV": cv95_0, "Reject r=0 @95%": bool(trace0 > cv95_0),
                           "Cointegrating rank": rank})

        oos_by_factor[f] = oos_static_mod.run_oos_for_factor(df_fs[f], factor_prices_fs[f], macro_f, cfg)

    return beta_rows, rank_rows, oos_by_factor


def run_factor_specific_kalman(df_fs: pd.DataFrame, factor_prices_fs: pd.DataFrame, macro_levels_fs: pd.DataFrame,
                                cfg: Config) -> tuple[dict, dict]:
    """Per-factor Kalman calibration/filter + expanding predictive OOS, each on
    that factor's own 3-driver subset. Returns (kalman_results, oos_by_factor)."""
    kalman_results, oos_by_factor = {}, {}
    for f in cfg.unconstrained_factors:
        drivers = cfg.factor_specific_drivers[f]
        macro_f = macro_levels_fs[drivers]
        kalman_results[f] = kalman_mod.run_kalman_for_factor(factor_prices_fs[f], macro_f, cfg)
        oos_by_factor[f] = kalman_mod.run_oos_for_factor(df_fs[f], kalman_results[f]["ect"], cfg)
    return kalman_results, oos_by_factor


def main() -> None:
    cfg = Config()
    melone_path = os.path.join(cfg.results_dir, cfg.output_csv)
    lasso_pred_path = os.path.join(cfg.results_dir, "lasso_predictions.csv")
    fs_path = os.path.join(cfg.results_dir, cfg.factor_specific_output_csv)
    for path in (melone_path, lasso_pred_path, fs_path):
        if not os.path.exists(path):
            sys.exit(f"{path} not found -- run main.py, train_lasso.py, and main_factor_specific.py first.")
    os.makedirs(cfg.results_dir, exist_ok=True)

    df_melone = pd.read_csv(melone_path, index_col=0, parse_dates=True)
    lasso_predictions = pd.read_csv(lasso_pred_path, header=[0, 1], index_col=0, parse_dates=True)
    df_fs = pd.read_csv(fs_path, index_col=0, parse_dates=True)
    print(f"Melone dataset: {len(df_melone)} quarters, {df_melone.index.min().date()} to {df_melone.index.max().date()}")
    print(f"Factor-specific dataset: {len(df_fs)} quarters, {df_fs.index.min().date()} to {df_fs.index.max().date()}")

    factor_prices, macro_levels = construct_mod.build_levels(df_melone, cfg)
    factor_prices_fs = construct_mod.build_factor_prices(df_fs, cfg.unconstrained_factors)
    macro_levels_fs = construct_mod.build_macro_driver_levels(df_fs, cfg.factor_specific_macro_columns)

    # ---------------------------------------------------------- Melone (already-known methodology, re-run fresh)
    _section("Melone (shared 4-driver panel): static + Kalman OOS")
    melone_static_oos = oos_static_mod.run_all_oos(df_melone, factor_prices, macro_levels, cfg)
    melone_kalman_results = kalman_mod.run_all_kalman(factor_prices, macro_levels, cfg)
    melone_kalman_oos = kalman_mod.run_all_oos(df_melone, melone_kalman_results, cfg)
    melone_static_preds = port_mod.to_wide_predictions(melone_static_oos, cfg.unconstrained_factors)
    melone_kalman_preds = port_mod.to_wide_predictions(melone_kalman_oos, cfg.unconstrained_factors)
    print("Done.")

    # ------------------------------------------------------------------- Factor-specific: static
    _section("Factor-specific drivers: static long-run regression + Johansen")
    beta_rows, rank_rows, fs_static_oos = run_factor_specific_static(df_fs, factor_prices_fs, macro_levels_fs, cfg)
    beta_table_fs = pd.DataFrame(beta_rows).set_index("Factor")
    rank_table_fs = pd.DataFrame(rank_rows).set_index("Factor")
    print(beta_table_fs.round(4).to_string())
    print()
    print(rank_table_fs.round(2).to_string())
    _save_csv(beta_table_fs, "factor_specific_static_beta_table", cfg)
    _save_csv(rank_table_fs, "factor_specific_johansen_rank", cfg)
    fs_static_preds = port_mod.to_wide_predictions(fs_static_oos, cfg.unconstrained_factors)

    fs_static_r2 = lasso_mod.build_r2_table({f: {"predictions": fs_static_oos[f]} for f in cfg.unconstrained_factors}, cfg)
    fs_static_hit = lasso_mod.build_hit_rate_table({f: {"predictions": fs_static_oos[f]} for f in cfg.unconstrained_factors}, cfg)
    _section("Factor-specific static: OOS R^2 and hit rate")
    print(fs_static_r2.round(3).to_string())
    print(fs_static_hit.round(3).to_string())
    _save_csv(fs_static_r2, "factor_specific_static_oos_r2", cfg)
    _save_csv(fs_static_hit, "factor_specific_static_oos_hitrate", cfg)

    # ------------------------------------------------------------------- Factor-specific: Kalman
    _section("Factor-specific drivers: Kalman calibration + OOS")
    fs_kalman_results, fs_kalman_oos = run_factor_specific_kalman(df_fs, factor_prices_fs, macro_levels_fs, cfg)
    kalman_params_fs = pd.DataFrame(
        {f: {"r (obs var)": fs_kalman_results[f]["r"], "q (state var)": fs_kalman_results[f]["q"],
             "Drivers": ", ".join(cfg.factor_specific_drivers[f])} for f in cfg.unconstrained_factors}
    ).T
    print(kalman_params_fs.to_string())
    _save_csv(kalman_params_fs, "factor_specific_kalman_params", cfg)
    fs_kalman_preds = port_mod.to_wide_predictions(fs_kalman_oos, cfg.unconstrained_factors)

    fs_kalman_r2 = lasso_mod.build_r2_table({f: {"predictions": fs_kalman_oos[f]} for f in cfg.unconstrained_factors}, cfg)
    fs_kalman_hit = lasso_mod.build_hit_rate_table({f: {"predictions": fs_kalman_oos[f]} for f in cfg.unconstrained_factors}, cfg)
    _section("Factor-specific Kalman: OOS R^2 and hit rate")
    print(fs_kalman_r2.round(3).to_string())
    print(fs_kalman_hit.round(3).to_string())
    _save_csv(fs_kalman_r2, "factor_specific_kalman_oos_r2", cfg)
    _save_csv(fs_kalman_hit, "factor_specific_kalman_oos_hitrate", cfg)

    # ----------------------------------------------------------------------------------- Portfolios
    _section("Building portfolios: Melone Static/Kalman, Factor-specific Static/Kalman, LASSO, Benchmark")
    universes = {"Unconstrained": cfg.unconstrained_factors, "Beta-neutral": cfg.beta_neutral_factors}
    expanding_methods = {
        "Melone Static": (df_melone, melone_static_preds),
        "Melone Kalman": (df_melone, melone_kalman_preds),
        "FS Static": (df_fs, fs_static_preds),
        "FS Kalman": (df_fs, fs_kalman_preds),
    }

    portfolios = {}
    for method, (dataset, preds) in expanding_methods.items():
        for uni_name, factors in universes.items():
            name = f"{method} - {uni_name}"
            w = ft_mod.build_expanding_timing_weights(dataset, preds, factors, cfg)
            summary = ft_mod.summarize_portfolio(name, w, dataset, cfg)
            summary["metrics"]["Hit rate (%)"] = _mean_hit_rate(preds, factors)
            portfolios[name] = summary

    for uni_name, factors in universes.items():
        name = f"LASSO - {uni_name}"
        w = ft_mod.build_timing_weights(df_melone, lasso_predictions, factors, cfg)
        summary = ft_mod.summarize_portfolio(name, w, df_melone, cfg)
        summary["metrics"]["Hit rate (%)"] = _mean_hit_rate(lasso_predictions, factors)
        portfolios[name] = summary

        bench_name = f"Benchmark (EW) - {uni_name}"
        w_bench = ft_mod.equal_weight_benchmark(lasso_predictions.index, factors)
        summary_bench = ft_mod.summarize_portfolio(bench_name, w_bench, df_melone, cfg)
        summary_bench["metrics"]["Hit rate (%)"] = float("nan")
        portfolios[bench_name] = summary_bench

    _section("Comparison table (full OOS period)")
    comparison = pd.DataFrame([p["metrics"] for p in portfolios.values()]).set_index("Portfolio")
    comparison = comparison[["Sharpe (ann.)", "CER (%, ann., gamma=5)", "Max drawdown (%)",
                              "Hit rate (%)", "Avg quarterly turnover"]]
    print(comparison.round(3).to_string())
    _save_csv(comparison, "factor_specific_comparison_full", cfg)

    subperiod_tables = port_mod.build_subperiod_tables(portfolios, cfg)
    for label, table in subperiod_tables.items():
        table = table.reindex(columns=["Sharpe (ann.)", "CER (%, ann., gamma=5)", "Max drawdown (%)",
                                        "Avg quarterly turnover"])
        _section(f"Comparison table: {label}")
        print(table.round(3).to_string())
        _save_csv(table, f"factor_specific_comparison_{label.replace('-', '_')}", cfg)

    returns_wide = pd.concat({name: p["returns"] for name, p in portfolios.items()}, axis=1)
    weights_wide = pd.concat({name: p["weights"] for name, p in portfolios.items()}, axis=1)
    _save_csv(returns_wide, "factor_specific_portfolio_returns", cfg)
    _save_csv(weights_wide, "factor_specific_portfolio_weights", cfg)

    _section("Generating plots (Unconstrained universe)")
    headline = {
        "Melone Static": portfolios["Melone Static - Unconstrained"]["returns"],
        "Melone Kalman": portfolios["Melone Kalman - Unconstrained"]["returns"],
        "FS Static": portfolios["FS Static - Unconstrained"]["returns"],
        "FS Kalman": portfolios["FS Kalman - Unconstrained"]["returns"],
        "LASSO": portfolios["LASSO - Unconstrained"]["returns"],
        "Benchmark (EW)": portfolios["Benchmark (EW) - Unconstrained"]["returns"],
    }
    colors = [plots_mod.CATEGORICAL[i] for i in range(len(headline))]
    cum_path = plots_mod.plot_cumulative_returns(
        headline, cfg, colors=colors, filename="factor_specific_cumulative_returns",
        title="Melone vs. Factor-Specific Drivers vs. LASSO vs. Benchmark: Cumulative Returns (Unconstrained)")
    dd_path = plots_mod.plot_drawdown(
        headline, cfg, colors=colors, filename="factor_specific_drawdown",
        title="Melone vs. Factor-Specific Drivers vs. LASSO vs. Benchmark: Drawdown (Unconstrained)")
    print(f"Saved {cum_path}")
    print(f"Saved {dd_path}")


if __name__ == "__main__":
    sys.exit(main())
