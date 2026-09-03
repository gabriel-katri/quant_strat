"""
Melone (Favero, Melone & Tamoni 2022) factor-timing pipeline, quarterly.

Prep:    factor price levels (ln F_t) and macro driver levels (M_t) from
         the already-assembled quarterly dataset.
Step 1:  static cointegration -- full-sample long-run regression (beta,
         static ECT), Johansen test, in-sample predictive regression,
         expanding-window OOS evaluation, and the resulting portfolio.
Step 2:  time-varying cointegration via a Kalman filter, two variants --
         (RW) unconstrained random walk, (MR) mean-reverting toward the
         static beta -- both estimated and reported side by side (neither
         is selected as a "winner"). beta_t, ECT_t, expanding-window
         predictive OOS, portfolio, for each variant.
Compare: Static / Kalman (RW) / Kalman (MR) / LASSO (already trained, see
         train_lasso.py) / equal-weight-benchmark portfolios, full period +
         2000-2019 / 2020-2026 sub-periods.

Requires results/rolling_lasso_dataset.csv and results/lasso_predictions.csv
-- run main.py then train_lasso.py first if they don't exist yet.

Run with: python run_melone.py
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

import lasso as lasso_mod
import melone_cointegration as coint_mod
import melone_construct as construct_mod
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


def main() -> None:
    cfg = Config()
    dataset_path = os.path.join(cfg.results_dir, cfg.output_csv)
    lasso_pred_path = os.path.join(cfg.results_dir, "lasso_predictions.csv")
    for path in (dataset_path, lasso_pred_path):
        if not os.path.exists(path):
            sys.exit(f"{path} not found -- run `python main.py` and `python train_lasso.py` first.")
    os.makedirs(cfg.results_dir, exist_ok=True)

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    lasso_predictions = pd.read_csv(lasso_pred_path, header=[0, 1], index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} quarterly observations, {df.index.min().date()} to {df.index.max().date()}")

    # ------------------------------------------------------------------ Prep
    _section("Prep: factor price levels (ln F_t) and macro driver levels (M_t)")
    factor_prices, macro_levels = construct_mod.build_levels(df, cfg)
    print(f"Factors: {list(factor_prices.columns)} | Macro drivers: {list(macro_levels.columns)}")
    levels_path = plots_mod.plot_melone_levels(factor_prices, macro_levels, cfg)
    print(f"Saved {levels_path}")

    # ------------------------------------------------------- Step 1: static
    _section("Step 1 (static): long-run regression, beta coefficients")
    lr_models = ect_mod.estimate_all_long_run(factor_prices, macro_levels, cfg)
    beta_table = ect_mod.build_beta_table(lr_models, cfg)
    print(beta_table.round(4).to_string())
    static_ects = ect_mod.get_static_ects(lr_models, factor_prices, cfg)
    _save_csv(beta_table, "melone_static_beta_table", cfg)
    _save_csv(static_ects, "melone_static_ect", cfg)

    _section("Step 1 (static): Johansen cointegration test")
    johansen_results = coint_mod.run_all_johansen_tests(factor_prices, macro_levels, cfg)
    trace_table, lmax_table = coint_mod.build_trace_lmax_tables(johansen_results, cfg)
    rank_table = coint_mod.summarize_cointegration_rank(johansen_results, cfg)
    print("Trace statistic vs 95% critical value:")
    print(trace_table.round(2).to_string())
    print("\nL-max statistic vs 95% critical value:")
    print(lmax_table.round(2).to_string())
    print("\nCointegrating rank (sequential trace test @95%):")
    print(rank_table.to_string())
    _save_csv(trace_table, "melone_johansen_trace", cfg)
    _save_csv(lmax_table, "melone_johansen_lmax", cfg)
    _save_csv(rank_table, "melone_johansen_rank", cfg)

    _section("Step 1 (static): predictive regression r_{t+1} = a + delta*ECT_t + v")
    predictive_results = pred_mod.run_all_predictive(df, static_ects, cfg)
    predictive_table = pred_mod.build_predictive_table(predictive_results, cfg)
    print(predictive_table.round(4).to_string())
    _save_csv(predictive_table, "melone_predictive_static", cfg)

    _section(f"Step 1 (static): out-of-sample evaluation (expanding window, {cfg.rolling_window_periods}Q initial train)")
    static_oos = oos_static_mod.run_all_oos(df, factor_prices, macro_levels, cfg)
    static_r2_table = lasso_mod.build_r2_table(
        {f: {"predictions": static_oos[f]} for f in cfg.unconstrained_factors}, cfg)
    static_hit_table = lasso_mod.build_hit_rate_table(
        {f: {"predictions": static_oos[f]} for f in cfg.unconstrained_factors}, cfg)
    print("OOS R^2:")
    print(static_r2_table.round(3).to_string())
    print("\nHit rate:")
    print(static_hit_table.round(3).to_string())
    _save_csv(static_r2_table, "melone_static_oos_r2", cfg)
    _save_csv(static_hit_table, "melone_static_oos_hitrate", cfg)
    static_predictions = port_mod.to_wide_predictions(static_oos, cfg.unconstrained_factors)

    # ------------------------------------------------ Step 2: Kalman (both variants)
    beta_static_by_factor = ect_mod.get_beta_static_series(lr_models, cfg)

    _section(f"Step 2 (Kalman, RW): calibrating (r, q) on the initial {cfg.rolling_window_periods}Q window")
    kalman_rw = kalman_mod.run_all_kalman(factor_prices, macro_levels, cfg)
    kalman_rw_params = pd.DataFrame(
        {f: {"r (obs var)": kalman_rw[f]["r"], "q (state var)": kalman_rw[f]["q"]}
         for f in cfg.unconstrained_factors}
    ).T
    print(kalman_rw_params.to_string())
    _save_csv(kalman_rw_params, "melone_kalman_rw_params", cfg)

    _section(f"Step 2 (Kalman, MR): calibrating (r, q, phi) on the initial {cfg.rolling_window_periods}Q window")
    kalman_mr = kalman_mod.run_all_kalman_mean_reverting(factor_prices, macro_levels, beta_static_by_factor, cfg)
    kalman_mr_params = pd.DataFrame(
        {f: {"r (obs var)": kalman_mr[f]["r"], "q (state var)": kalman_mr[f]["q"], "phi": kalman_mr[f]["phi"]}
         for f in cfg.unconstrained_factors}
    ).T
    print(kalman_mr_params.to_string())
    _save_csv(kalman_mr_params, "melone_kalman_mr_params", cfg)

    for label, results in [("rw", kalman_rw), ("mr", kalman_mr)]:
        betas_wide = pd.concat({f: results[f]["beta_t"] for f in cfg.unconstrained_factors}, axis=1)
        ect_wide = pd.concat({f: results[f]["ect"] for f in cfg.unconstrained_factors}, axis=1)
        _save_csv(betas_wide, f"melone_kalman_{label}_betas", cfg)
        _save_csv(ect_wide, f"melone_kalman_{label}_ect", cfg)

    betas_plot_path = plots_mod.plot_kalman_betas(kalman_rw, cfg)
    ect_plot_path = plots_mod.plot_ect_static_vs_kalman(static_ects, kalman_rw, cfg)
    print(f"Saved {betas_plot_path} (RW variant)")
    print(f"Saved {ect_plot_path} (RW variant)")

    _section("Step 2 (Kalman): out-of-sample evaluation, both variants (expanding-window predictive regression)")
    kalman_rw_oos = kalman_mod.run_all_oos(df, kalman_rw, cfg)
    kalman_mr_oos = kalman_mod.run_all_oos(df, kalman_mr, cfg)
    for label, oos in [("RW", kalman_rw_oos), ("MR", kalman_mr_oos)]:
        r2_table = lasso_mod.build_r2_table({f: {"predictions": oos[f]} for f in cfg.unconstrained_factors}, cfg)
        hit_table = lasso_mod.build_hit_rate_table({f: {"predictions": oos[f]} for f in cfg.unconstrained_factors}, cfg)
        print(f"\nKalman ({label}) OOS R^2:")
        print(r2_table.round(3).to_string())
        print(f"Kalman ({label}) hit rate:")
        print(hit_table.round(3).to_string())
        _save_csv(r2_table, f"melone_kalman_{label.lower()}_oos_r2", cfg)
        _save_csv(hit_table, f"melone_kalman_{label.lower()}_oos_hitrate", cfg)

    kalman_rw_predictions = port_mod.to_wide_predictions(kalman_rw_oos, cfg.unconstrained_factors)
    kalman_mr_predictions = port_mod.to_wide_predictions(kalman_mr_oos, cfg.unconstrained_factors)

    # --------------------------------------------------- Portfolio compare
    _section("Portfolios: Static vs Kalman (RW) vs Kalman (MR) vs LASSO vs Benchmark (Beta-neutral & Unconstrained)")
    expanding_methods = {
        "Static": static_predictions,
        "Kalman (RW)": kalman_rw_predictions,
        "Kalman (MR)": kalman_mr_predictions,
    }
    portfolios = port_mod.build_all_portfolios(df, expanding_methods, lasso_predictions, cfg)

    universes = {"Unconstrained": cfg.unconstrained_factors, "Beta-neutral": cfg.beta_neutral_factors}
    predictions_by_method = {"Static": static_predictions, "Kalman (RW)": kalman_rw_predictions,
                              "Kalman (MR)": kalman_mr_predictions, "LASSO": lasso_predictions}
    for name, p in portfolios.items():
        method, uni_name = name.rsplit(" - ", 1)
        if method in predictions_by_method:
            preds = predictions_by_method[method]
            factors = universes[uni_name]
            rates = [lasso_mod.hit_rate(preds[(f, "actual")], preds[(f, "predicted")]) for f in factors]
            p["metrics"]["Hit rate (%)"] = float(pd.Series(rates).mean()) * 100.0
        else:
            p["metrics"]["Hit rate (%)"] = float("nan")

    comparison_table = port_mod.build_comparison_table(portfolios)
    comparison_table = comparison_table[["Sharpe (ann.)", "CER (%, ann., gamma=5)", "Max drawdown (%)",
                                          "Hit rate (%)", "Avg quarterly turnover"]]
    print(comparison_table.round(3).to_string())
    _save_csv(comparison_table, "melone_portfolio_comparison", cfg)

    subperiod_tables = port_mod.build_subperiod_tables(portfolios, cfg)
    for label, table in subperiod_tables.items():
        _section(f"Portfolios: {label}")
        print(table.round(3).to_string())
        _save_csv(table, f"melone_portfolio_comparison_{label.replace('-', '_')}", cfg)

    returns_wide = pd.concat({name: p["returns"] for name, p in portfolios.items()}, axis=1)
    weights_wide = pd.concat({name: p["weights"] for name, p in portfolios.items()}, axis=1)
    _save_csv(returns_wide, "melone_portfolio_returns", cfg)
    _save_csv(weights_wide, "melone_portfolio_weights", cfg)

    _section("Generating comparison plots (Unconstrained universe)")
    headline = {
        "Static": portfolios["Static - Unconstrained"]["returns"],
        "Kalman (RW)": portfolios["Kalman (RW) - Unconstrained"]["returns"],
        "Kalman (MR)": portfolios["Kalman (MR) - Unconstrained"]["returns"],
        "LASSO": portfolios["LASSO - Unconstrained"]["returns"],
        "Benchmark (EW)": portfolios["Benchmark (EW) - Unconstrained"]["returns"],
    }
    colors = [plots_mod.STRATEGY_COLORS[name] for name in headline]
    cum_path = plots_mod.plot_cumulative_returns(
        headline, cfg, colors=colors, filename="melone_cumulative_returns",
        title="Melone Static vs. Kalman (RW/MR) vs. LASSO vs. Benchmark: Cumulative Returns (Unconstrained)")
    dd_path = plots_mod.plot_drawdown(
        headline, cfg, colors=colors, filename="melone_drawdown",
        title="Melone Static vs. Kalman (RW/MR) vs. LASSO vs. Benchmark: Drawdown (Unconstrained)")
    print(f"Saved {cum_path}")
    print(f"Saved {dd_path}")


if __name__ == "__main__":
    sys.exit(main())
