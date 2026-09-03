"""
Two read-only diagnostics on the Beat_Melone backtest -- does NOT modify
config.py, factor_timing.py, melone_portfolio.py, or any pipeline file.

1. MDD date check: Static, Kalman (RW), and Kalman (MR) (Unconstrained) all
   report an identical -24.2% max drawdown. Reports the actual trough date
   for each, to distinguish "shared shock" from "coincidence worth
   investigating."

2. Controlled covariance-window comparison: rebuilds the LASSO portfolio
   using build_expanding_timing_weights (the same expanding-window Sigma
   Static/Kalman use) instead of LASSO's own rolling-window Sigma, with
   LASSO's predictions (lasso_predictions.csv) held fixed. Reports Sharpe /
   CER / MDD / turnover for "LASSO (expanding Sigma)" alongside the
   original five portfolios.

Requires results/rolling_lasso_dataset.csv, results/lasso_predictions.csv,
results/melone_portfolio_returns.csv -- run main.py, train_lasso.py, and
run_melone.py first if they don't exist.

Run with: python diagnose_mdd_and_lasso_sigma.py
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

import factor_timing as ft_mod
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
    lasso_pred_path = os.path.join(cfg.results_dir, "lasso_predictions.csv")
    returns_path = os.path.join(cfg.results_dir, "melone_portfolio_returns.csv")
    for path in (dataset_path, lasso_pred_path, returns_path):
        if not os.path.exists(path):
            sys.exit(f"{path} not found -- run main.py, train_lasso.py, and run_melone.py first.")

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    lasso_predictions = pd.read_csv(lasso_pred_path, header=[0, 1], index_col=0, parse_dates=True)
    portfolio_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    weights_path = os.path.join(cfg.results_dir, "melone_portfolio_weights.csv")
    portfolio_weights = pd.read_csv(weights_path, header=[0, 1], index_col=0, parse_dates=True)

    # ------------------------------------------------------------ Diagnostic 1
    _section("Diagnostic 1: max-drawdown trough date, Static / Kalman (RW) / Kalman (MR) -- Unconstrained")
    tied_names = ["Static - Unconstrained", "Kalman (RW) - Unconstrained", "Kalman (MR) - Unconstrained"]
    rows = []
    for name in tied_names:
        r = portfolio_returns[name].dropna()
        dd = ft_mod.drawdown_series(r)
        trough_date = dd.idxmin()
        trough_value = dd.min()
        # peak immediately preceding the trough (start of the drawdown episode)
        wealth = (1 + r).cumprod()
        peak_date = wealth.loc[:trough_date].idxmax()
        rows.append({"Portfolio": name, "Peak date": peak_date.date(), "Trough date": trough_date.date(),
                     "Max drawdown (%)": trough_value * 100})
    mdd_table = pd.DataFrame(rows).set_index("Portfolio")
    print(mdd_table.to_string())

    troughs = mdd_table["Trough date"]
    if troughs.nunique() == 1:
        print(f"\n-> All three troughs fall on the SAME date ({troughs.iloc[0]}): consistent with a shared-shock "
              f"explanation (a common episode where realized vol outran the expanding-window Sigma for all three).")
    else:
        same_quarter = len(set(pd.Timestamp(d).to_period("Q") for d in troughs)) == 1
        if same_quarter:
            print(f"\n-> Troughs fall in the same QUARTER but not the same exact date: {dict(troughs)}. "
                  f"Still consistent with a shared-shock explanation (quarterly rebalancing, so intra-quarter "
                  f"date differences don't imply a bug).")
        else:
            print(f"\n-> Troughs do NOT cluster: {dict(troughs)}. This is a genuine coincidence in the MAGNITUDE "
                  f"only, not the timing -- worth investigating further as a possible bug.")

    # ------------------------------------------------------------ Diagnostic 2
    _section("Diagnostic 2: LASSO with expanding-window Sigma (predictions unchanged)")
    universes = {"Unconstrained": cfg.unconstrained_factors, "Beta-neutral": cfg.beta_neutral_factors}
    lasso_expanding = {}
    for uni_name, factors in universes.items():
        name = f"LASSO (expanding Sigma) - {uni_name}"
        w = ft_mod.build_expanding_timing_weights(df, lasso_predictions, factors, cfg)
        lasso_expanding[name] = ft_mod.summarize_portfolio(name, w, df, cfg)

    orig_names = ["Static - Unconstrained", "Kalman (RW) - Unconstrained", "Kalman (MR) - Unconstrained",
                  "LASSO - Unconstrained", "Benchmark (EW) - Unconstrained",
                  "Static - Beta-neutral", "Kalman (RW) - Beta-neutral", "Kalman (MR) - Beta-neutral",
                  "LASSO - Beta-neutral", "Benchmark (EW) - Beta-neutral"]

    rows = []
    for name in orig_names:
        r = portfolio_returns[name].dropna()
        w = portfolio_weights[name].dropna(how="all")
        factors = list(w.columns)
        actual = df.loc[w.index, factors]
        turnover = ft_mod.period_turnover(w, actual, r.loc[w.index])
        rows.append({
            "Portfolio": name,
            "Sharpe (ann.)": ft_mod.sharpe_ratio(r, cfg),
            "CER (%, ann., gamma=5)": ft_mod.certainty_equivalent_return(r, cfg.cer_gamma, cfg) * 100,
            "Max drawdown (%)": ft_mod.max_drawdown(r) * 100,
            "Avg quarterly turnover": float(turnover.mean()),
        })
    for name, p in lasso_expanding.items():
        rows.append({
            "Portfolio": name,
            "Sharpe (ann.)": p["metrics"]["Sharpe (ann.)"],
            "CER (%, ann., gamma=5)": p["metrics"]["CER (%, ann., gamma=5)"],
            "Max drawdown (%)": p["metrics"]["Max drawdown (%)"],
            "Avg quarterly turnover": p["metrics"]["Avg quarterly turnover"],
        })

    table = pd.DataFrame(rows).set_index("Portfolio")
    print(table.round(3).to_string())

    out_path = os.path.join(cfg.results_dir, "diagnostic_lasso_expanding_sigma_comparison.csv")
    table.to_csv(out_path)
    print(f"\nSaved {out_path}")

    sharpe_orig = table.loc["LASSO - Unconstrained", "Sharpe (ann.)"]
    sharpe_new = table.loc["LASSO (expanding Sigma) - Unconstrained", "Sharpe (ann.)"]
    sharpe_static = table.loc["Static - Unconstrained", "Sharpe (ann.)"]
    print(f"\nLASSO Sharpe: {sharpe_orig:.3f} (own rolling Sigma) -> {sharpe_new:.3f} (expanding Sigma, matched to Static/Kalman)")
    print(f"Static Sharpe (reference, same Sigma convention): {sharpe_static:.3f}")


if __name__ == "__main__":
    sys.exit(main())
