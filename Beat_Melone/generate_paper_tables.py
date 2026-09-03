"""
Paper table refresh, gamma=5, main pipeline's default window (expanding,
1975Q1-1999Q4 in-sample, 2000Q1-2026Q2 out-of-sample) -- this script does
NOT touch that window, it only reads results off it.

Six portfolios throughout: Static, Benchmark (EW), LASSO (matched Sigma --
LASSO's own predictions with the expanding-window covariance, isolating
the covariance-window confound per the earlier diagnostic), LASSO (own
rolling Sigma -- the original), Kalman (RW), Kalman (MR).

Produces three tables:
  1. Full OOS period (2000Q1-2026Q2), Unconstrained universe.
  2. Same six portfolios/metrics, split into 2000Q1-2019Q4 and
     2020Q1-2026Q2, Unconstrained universe.
  3. Same six portfolios/metrics, Beta-neutral universe, full period +
     the same two sub-periods.

All three include Hit rate (%), computed (and re-sliced per sub-period)
from each method's own predictions -- not available for Benchmark (EW),
which has no forecast.

Requires results/rolling_lasso_dataset.csv and results/lasso_predictions.csv
-- run main.py and train_lasso.py first if they don't exist.

Run with: python generate_paper_tables.py
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
import melone_ect as ect_mod
import melone_kalman as kalman_mod
import melone_oos_static as oos_static_mod
import melone_portfolio as port_mod
from config import Config

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

METRIC_COLS = ["Sharpe (ann.)", "CER (%, ann., gamma=5)", "Max drawdown (%)", "Hit rate (%)", "Avg quarterly turnover"]
PORTFOLIO_ORDER = ["Static", "Benchmark (EW)", "LASSO (matched Sigma)", "LASSO (own rolling Sigma)",
                    "Kalman (RW)", "Kalman (MR)"]


def _section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def _mean_hit_rate(predictions: pd.DataFrame, factors: list[str], mask: pd.Series | None = None) -> float:
    rates = []
    for f in factors:
        actual, predicted = predictions[(f, "actual")], predictions[(f, "predicted")]
        if mask is not None:
            actual, predicted = actual[mask], predicted[mask]
        if len(actual) < 1:
            return float("nan")
        rates.append(lasso_mod.hit_rate(actual, predicted))
    return float(np.mean(rates)) * 100.0


def _metrics_row(name: str, returns: pd.Series, turnover: pd.Series,
                  predictions: pd.DataFrame | None, factors: list[str], cfg: Config) -> dict:
    if len(returns) < 2:
        return {"Portfolio": name, **{c: float("nan") for c in METRIC_COLS}}
    row = {
        "Portfolio": name,
        "Sharpe (ann.)": ft_mod.sharpe_ratio(returns, cfg),
        "CER (%, ann., gamma=5)": ft_mod.certainty_equivalent_return(returns, cfg.cer_gamma, cfg) * 100,
        "Max drawdown (%)": ft_mod.max_drawdown(returns) * 100,
        "Hit rate (%)": _mean_hit_rate(predictions, factors, mask=predictions.index.isin(returns.index))
        if predictions is not None else float("nan"),
        "Avg quarterly turnover": float(turnover.mean()) if len(turnover) else float("nan"),
    }
    return row


def to_markdown(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    header = "| Portfolio | " + " | ".join(cols) + " |"
    sep = "|---|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for idx, row in df.iterrows():
        vals = [idx] + [("" if pd.isna(v) else float_fmt.format(v)) for v in row]
        lines.append("| " + " | ".join(str(v) for v in vals) + " |")
    return "\n".join(lines)


def main() -> None:
    cfg = Config()
    dataset_path = os.path.join(cfg.results_dir, cfg.output_csv)
    lasso_pred_path = os.path.join(cfg.results_dir, "lasso_predictions.csv")
    for path in (dataset_path, lasso_pred_path):
        if not os.path.exists(path):
            sys.exit(f"{path} not found -- run main.py and train_lasso.py first.")

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    lasso_predictions = pd.read_csv(lasso_pred_path, header=[0, 1], index_col=0, parse_dates=True)
    print(f"Loaded {len(df)} quarterly obs, {df.index.min().date()} to {df.index.max().date()}. "
          f"cfg.cer_gamma = {cfg.cer_gamma}")

    factor_prices, macro_levels = construct_mod.build_levels(df, cfg)

    _section("Recomputing Static / Kalman (RW) / Kalman (MR) OOS forecasts")
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
    print("Done.")

    universes = {"Unconstrained": cfg.unconstrained_factors, "Beta-neutral": cfg.beta_neutral_factors}
    predictions_by_method = {
        "Static": static_predictions, "Kalman (RW)": kalman_rw_predictions, "Kalman (MR)": kalman_mr_predictions,
        "LASSO (matched Sigma)": lasso_predictions, "LASSO (own rolling Sigma)": lasso_predictions,
    }

    _section("Building all six portfolios x two universes")
    portfolios = {}
    expanding_methods = {"Static": static_predictions, "Kalman (RW)": kalman_rw_predictions,
                          "Kalman (MR)": kalman_mr_predictions, "LASSO (matched Sigma)": lasso_predictions}
    for method, preds in expanding_methods.items():
        for uni_name, factors in universes.items():
            name = f"{method} - {uni_name}"
            w = ft_mod.build_expanding_timing_weights(df, preds, factors, cfg)
            portfolios[name] = ft_mod.summarize_portfolio(name, w, df, cfg)

    for uni_name, factors in universes.items():
        name = f"LASSO (own rolling Sigma) - {uni_name}"
        w = ft_mod.build_timing_weights(df, lasso_predictions, factors, cfg)
        portfolios[name] = ft_mod.summarize_portfolio(name, w, df, cfg)

        bname = f"Benchmark (EW) - {uni_name}"
        w_bench = ft_mod.equal_weight_benchmark(lasso_predictions.index, factors)
        portfolios[bname] = ft_mod.summarize_portfolio(bname, w_bench, df, cfg)
    print(f"Built {len(portfolios)} portfolios.")

    def build_table(uni_name: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        factors = universes[uni_name]
        rows = []
        for method in PORTFOLIO_ORDER:
            name = f"{method} - {uni_name}"
            p = portfolios[name]
            returns, turnover = p["returns"], p["turnover"]
            r_mask = pd.Series(True, index=returns.index)
            t_mask = pd.Series(True, index=turnover.index)
            if start is not None:
                r_mask &= returns.index >= start
                t_mask &= turnover.index >= start
            if end is not None:
                r_mask &= returns.index <= end
                t_mask &= turnover.index <= end
            preds = predictions_by_method.get(method)
            rows.append(_metrics_row(method, returns[r_mask], turnover[t_mask], preds, factors, cfg))
        return pd.DataFrame(rows).set_index("Portfolio")[METRIC_COLS]

    sub1_start, sub1_end = cfg.oos_subperiods["2000-2019"]
    sub2_start, sub2_end = cfg.oos_subperiods["2020-2026"]

    table1 = build_table("Unconstrained")
    table2a = build_table("Unconstrained", sub1_start, sub1_end)
    table2b = build_table("Unconstrained", sub2_start, sub2_end)
    table3_full = build_table("Beta-neutral")
    table3a = build_table("Beta-neutral", sub1_start, sub1_end)
    table3b = build_table("Beta-neutral", sub2_start, sub2_end)

    _section("TABLE 1: Full OOS period (2000Q1-2026Q2), Unconstrained")
    print(table1.round(3).to_string())
    _section("TABLE 2a: 2000Q1-2019Q4, Unconstrained")
    print(table2a.round(3).to_string())
    _section("TABLE 2b: 2020Q1-2026Q2, Unconstrained")
    print(table2b.round(3).to_string())
    _section("TABLE 3 (full period): Beta-neutral")
    print(table3_full.round(3).to_string())
    _section("TABLE 3a: 2000Q1-2019Q4, Beta-neutral")
    print(table3a.round(3).to_string())
    _section("TABLE 3b: 2020Q1-2026Q2, Beta-neutral")
    print(table3b.round(3).to_string())

    _section("Saving CSVs + markdown")
    outputs = {
        "paper_table1_full_unconstrained": table1,
        "paper_table2a_2000_2019_unconstrained": table2a,
        "paper_table2b_2020_2026_unconstrained": table2b,
        "paper_table3_full_betaneutral": table3_full,
        "paper_table3a_2000_2019_betaneutral": table3a,
        "paper_table3b_2020_2026_betaneutral": table3b,
    }
    md_parts = []
    for fname, table in outputs.items():
        csv_path = os.path.join(cfg.results_dir, f"{fname}.csv")
        table.to_csv(csv_path)
        print(f"Saved {csv_path}")
        md_parts.append(f"### {fname}\n\n{to_markdown(table)}\n")

    md_path = os.path.join(cfg.results_dir, "paper_tables.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(md_parts))
    print(f"Saved {md_path}")


if __name__ == "__main__":
    sys.exit(main())
