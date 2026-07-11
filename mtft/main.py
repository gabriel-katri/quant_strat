"""
End-to-end Macro Trends and Factor Timing (MTFT) pipeline, replicating
Favero, Melone & Tamoni (2022). Downloads data, tests cointegration,
estimates the ECT and predictive regressions, evaluates OOS forecasts,
backtests factor-timing portfolios, runs an extended 2020-2026 backtest,
bootstraps crisis-period risk distributions, and checks robustness to
alternative macro drivers.

Run with: python main.py
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

import backtest_2020 as bt2020
import cointegration as coint_mod
import construct
import data as data_mod
import ect as ect_mod
import oos as oos_mod
import plots as plots_mod
import portfolio as port_mod
import predictive as pred_mod
import risk as risk_mod
import robustness as rb_mod
from config import Config

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def _section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    cfg = Config()
    os.makedirs(cfg.results_dir, exist_ok=True)

    # ------------------------------------------------------------------
    _section("STEP 0/9  Downloading data (FF5 + macro drivers)")
    d = data_mod.load_data(cfg)
    factor_log_returns = d["factor_log_returns"]
    macro_series = d["macro_series"]
    print(f"Liquidity driver source: {d['liquidity_source']}")
    print(f"Core sample: {factor_log_returns.index.min().date()} to {factor_log_returns.index.max().date()} "
          f"({len(factor_log_returns)} quarters, macro drivers: {cfg.macro_drivers})")
    print(f"Extended (VIX-liquidity) sample: {d['factor_log_returns_ext'].index.min().date()} to "
          f"{d['factor_log_returns_ext'].index.max().date()} ({len(d['factor_log_returns_ext'])} quarters)")

    # ------------------------------------------------------------------
    _section("STEP 1/9  Constructing factor prices & macro driver levels + Figure 2")
    factor_prices = construct.build_factor_prices(factor_log_returns)
    macro_driver_levels = construct.build_macro_driver_levels(macro_series)
    fig2_path = plots_mod.plot_figure2(macro_driver_levels, factor_prices, cfg)
    print(f"Saved {fig2_path}")

    # ------------------------------------------------------------------
    _section("STEP 2/9  Johansen cointegration test (Table 1)")
    johansen_results = coint_mod.run_all_johansen_tests(factor_prices, macro_driver_levels, cfg)
    table1 = coint_mod.build_table1(johansen_results, cfg)
    print("\nTrace statistics (95% CV in last column):")
    print(table1["trace"].round(3).to_string())
    print("\nL-max statistics (95% CV in last column):")
    print(table1["lmax"].round(3).to_string())

    rank_summary = coint_mod.summarize_cointegration_rank(johansen_results, cfg)
    print("\nCointegrating rank by factor (trace test @95%):")
    print(rank_summary.to_string())
    n_rank1 = (rank_summary["Cointegrating rank"] == 1).sum()
    print(f"\nConclusion: {n_rank1}/{len(cfg.factors)} factors show exactly 1 cointegrating relationship "
          f"with the macro drivers at the 95% level.")

    # ------------------------------------------------------------------
    _section("STEP 3/9  Long-run regression & ECT (Table A.1, Figure 1a)")
    long_run_models = ect_mod.estimate_all_ects(factor_prices, macro_driver_levels, cfg)
    table_a1 = ect_mod.build_table_a1(long_run_models, cfg)
    print("\nTable A.1: Long-run regression coefficients (HAC SE in parentheses)")
    print(table_a1.to_string())

    ects = ect_mod.get_ects(long_run_models, cfg)
    fig1a_path = plots_mod.plot_figure1a(
        factor_prices["HML"], long_run_models["HML"].fittedvalues, "HML", cfg
    )
    print(f"\nSaved {fig1a_path}")

    # ------------------------------------------------------------------
    _section("STEP 4/9  In-sample predictive regression (Table 3 Panel A, Figure 1b)")
    predictive_results = pred_mod.run_all_predictive_regressions(factor_log_returns, ects, cfg)
    table3a = pred_mod.build_table3_panel_a(predictive_results, cfg)
    print("\nTable 3 Panel A: In-sample predictive regression, f(t+1) ~ ECT(t)")
    print(table3a.to_string())

    fig1b_path = plots_mod.plot_figure1b(predictive_results["HML"], "HML", cfg)
    print(f"\nSaved {fig1b_path}")

    scatter_path = plots_mod.plot_predicted_vs_realized(predictive_results, cfg)
    print(f"Saved {scatter_path}")

    # ------------------------------------------------------------------
    _section("STEP 5/9  Out-of-sample evaluation (Table 3 Panel B)")
    oos_results = oos_mod.run_all_oos_evaluations(factor_log_returns, factor_prices, macro_driver_levels, cfg)
    table3b = oos_mod.build_table3_panel_b(oos_results, cfg)
    print("\nTable 3 Panel B: OOS R^2 (%), Clark-West stars")
    print(table3b.to_string())

    # ------------------------------------------------------------------
    _section("STEP 6/9  Portfolio construction & backtest (Table 5)")
    is_results = port_mod.run_all_strategies(factor_log_returns, ects, predictive_results, cfg)
    oos_split_returns = port_mod.run_oos_split_strategies(factor_log_returns, factor_prices, macro_driver_levels, cfg)
    table5 = port_mod.build_table5(is_results, oos_split_returns, cfg)
    print(f"\nTable 5: Strategy performance (gamma={cfg.risk_aversion}, IS/OOS split at {cfg.is_oos_split_date})")
    print(table5.round(4).to_string())

    # ------------------------------------------------------------------
    _section("STEP 7/9  Extended backtest, 2020Q1 - latest")
    fixed_bt = bt2020.fixed_parameter_backtest(factor_log_returns, factor_prices, macro_driver_levels, cfg)
    expanding_bt = bt2020.expanding_window_backtest(factor_log_returns, factor_prices, macro_driver_levels, cfg)

    print("\nFixed-parameter backtest (coefficients frozen at 2019Q4):")
    print(bt2020.summarize_extended_backtest(fixed_bt["returns"], cfg).round(3).to_string())
    print("\nExpanding-window backtest (re-estimated every quarter):")
    print(bt2020.summarize_extended_backtest(expanding_bt["returns"], cfg).round(3).to_string())

    fi_static_2020 = fixed_bt["returns"]["FI"]
    print(f"\nStatic FI over 2020-2026: mean={fi_static_2020.mean()*4*100:.2f}%/yr, "
          f"vol={fi_static_2020.std(ddof=1)*2*100:.2f}%/yr, "
          f"Sharpe={bt2020.annualized_sharpe(fi_static_2020, cfg):.3f}")

    wealth_path = plots_mod.plot_cumulative_wealth_2020(
        fixed_bt["returns"], ["FI", "FT", "AT", "BN"], cfg, "cumulative_wealth_2020_2026"
    )
    print(f"Saved {wealth_path}")

    ect_zoom_path = plots_mod.plot_ect_timeseries(ects, cfg, zoom_start=cfg.extended_backtest_start)
    print(f"Saved {ect_zoom_path}")

    # ------------------------------------------------------------------
    _section("STEP 8/9  Risk management: predicted distributions around crises")
    H = risk_mod.build_value_high_leg(d["size_bm_monthly"])
    gfc_results = risk_mod.run_crisis_year_analysis(H, factor_prices, macro_driver_levels, [2008, 2009], cfg)
    covid_results = risk_mod.run_crisis_year_analysis(H, factor_prices, macro_driver_levels, [2020, 2021], cfg)
    var_table = risk_mod.build_var_table({"GFC": gfc_results, "COVID": covid_results})
    print("\n10% VaR and realized returns, CER vs Macro-FECM:")
    print(var_table.round(4).to_string(index=False))

    fig5_gfc = plots_mod.plot_crisis_distributions(gfc_results, [2008, 2009], "2008-2009 GFC", cfg)
    fig5_covid = plots_mod.plot_crisis_distributions(covid_results, [2020, 2021], "2020-2021 COVID", cfg)
    print(f"Saved {fig5_gfc}")
    print(f"Saved {fig5_covid}")

    # ------------------------------------------------------------------
    _section("STEP 9/9  Robustness: alternative macro drivers")
    gold = rb_mod.build_gold_driver()
    corp_spread = rb_mod.build_corp_spread_driver()

    variant_gold = macro_series[["pot_output", "term_spread"]].join(gold, how="inner")
    variant_corp = macro_series[["oil", "pot_output"]].join(corp_spread, how="inner")

    v_gold = rb_mod.run_variant_pipeline(factor_log_returns, variant_gold, cfg)
    v_corp = rb_mod.run_variant_pipeline(factor_log_returns, variant_corp, cfg)
    v_liq = rb_mod.run_variant_pipeline(d["factor_log_returns_ext"], d["macro_series_ext"], cfg)

    robustness_summary = rb_mod.build_robustness_summary(
        {"Gold-for-Oil": v_gold, "CorpSpread-for-Term": v_corp, "VIX-Liquidity-Ext": v_liq}, cfg
    )
    print("\nRobustness summary (baseline drivers: oil, potential output, term spread, 1968-2026):")
    print(robustness_summary.round(3).to_string(index=False))

    # ------------------------------------------------------------------
    _section("SUMMARY")
    strat_returns_df = pd.DataFrame({s: is_results[s]["returns"] for s in port_mod.STRATEGIES})
    summary_stats = pd.DataFrame({
        "Ann. Mean (%)": strat_returns_df.mean() * cfg.quarters_per_year * 100,
        "Ann. Vol (%)": strat_returns_df.std(ddof=1) * (cfg.quarters_per_year ** 0.5) * 100,
        "Sharpe": strat_returns_df.apply(lambda r: port_mod.annualized_sharpe(r, cfg)),
        "Max Drawdown (%)": strat_returns_df.apply(lambda r: bt2020.max_drawdown(r) * 100),
    })
    print("\nFull-sample strategy summary statistics:")
    print(summary_stats.round(3).to_string())

    print("\nCorrelation matrix of strategy returns:")
    print(strat_returns_df.corr().round(3).to_string())

    breakdown_flags = rank_summary[rank_summary["Cointegrating rank"] != 1]
    if len(breakdown_flags):
        print(f"\n[FLAG] Factors where the trace test does NOT indicate exactly 1 cointegrating "
              f"relationship: {list(breakdown_flags.index)}")
    else:
        print("\nAll 5 factors show exactly 1 cointegrating relationship with the macro drivers.")

    # ------------------------------------------------------------------
    table1["trace"].to_csv(os.path.join(cfg.results_dir, "table1_trace.csv"))
    table1["lmax"].to_csv(os.path.join(cfg.results_dir, "table1_lmax.csv"))
    table_a1.to_csv(os.path.join(cfg.results_dir, "table_a1_long_run_regression.csv"))
    table3a.to_csv(os.path.join(cfg.results_dir, "table3_panel_a_predictive.csv"))
    table3b.to_csv(os.path.join(cfg.results_dir, "table3_panel_b_oos.csv"))
    table5.to_csv(os.path.join(cfg.results_dir, "table5_portfolio_performance.csv"))
    var_table.to_csv(os.path.join(cfg.results_dir, "table_var_crisis_periods.csv"), index=False)
    robustness_summary.to_csv(os.path.join(cfg.results_dir, "table_robustness.csv"), index=False)
    summary_stats.to_csv(os.path.join(cfg.results_dir, "summary_strategy_stats.csv"))

    print(f"\nAll tables and plots saved to {os.path.abspath(cfg.results_dir)}/")


if __name__ == "__main__":
    sys.exit(main())
