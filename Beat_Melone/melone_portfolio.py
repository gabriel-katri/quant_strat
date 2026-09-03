"""
Steps 1 & 2 (Melone): assemble the Static / Kalman (both variants) / LASSO /
Benchmark portfolios (beta-neutral and unconstrained universes each), and
compute comparison metrics over the full OOS period and requested
sub-periods.

Static and Kalman weights use the expanding-window covariance (matching
the expanding window those methods re-estimate their forecasts on); LASSO
keeps its original trailing rolling-window covariance (matching the window
it was trained on) -- see factor_timing.py.
"""

from __future__ import annotations

import pandas as pd

import factor_timing as ft_mod
from config import Config


def to_wide_predictions(results_by_factor: dict, factors: list[str]) -> pd.DataFrame:
    """{factor: df(actual, predicted)} -> wide multi-index frame, columns (factor, field)."""
    return pd.concat({f: results_by_factor[f][["actual", "predicted"]] for f in factors}, axis=1)


def _metrics_row(name: str, returns: pd.Series, turnover: pd.Series, cfg: Config) -> dict:
    if len(returns) < 2:
        return {"Portfolio": name, "Sharpe (ann.)": float("nan"), "CER (%, ann., gamma=5)": float("nan"),
                "Max drawdown (%)": float("nan"), "Avg quarterly turnover": float("nan")}
    return {
        "Portfolio": name,
        "Sharpe (ann.)": ft_mod.sharpe_ratio(returns, cfg),
        "CER (%, ann., gamma=5)": ft_mod.certainty_equivalent_return(returns, cfg.cer_gamma, cfg) * 100,
        "Max drawdown (%)": ft_mod.max_drawdown(returns) * 100,
        "Avg quarterly turnover": float(turnover.mean()) if len(turnover) else float("nan"),
    }


def build_all_portfolios(dataset: pd.DataFrame, expanding_methods: dict, lasso_predictions: pd.DataFrame,
                          cfg: Config) -> dict:
    """Returns {name: {'returns', 'weights', 'turnover', 'metrics'}} for every
    (method x universe) combo, plus one equal-weight benchmark per universe.

    `expanding_methods` = {method_name: predictions_wide_df}, e.g. {"Static":
    ..., "Kalman (RW)": ..., "Kalman (MR)": ...} -- each built with the
    expanding-window covariance (see module docstring). LASSO keeps its own
    rolling-window covariance, so it's passed separately.
    """
    universes = {"Unconstrained": cfg.unconstrained_factors, "Beta-neutral": cfg.beta_neutral_factors}

    portfolios = {}
    for method, preds in expanding_methods.items():
        for uni_name, factors in universes.items():
            name = f"{method} - {uni_name}"
            w = ft_mod.build_expanding_timing_weights(dataset, preds, factors, cfg)
            portfolios[name] = ft_mod.summarize_portfolio(name, w, dataset, cfg)

    for uni_name, factors in universes.items():
        name = f"LASSO - {uni_name}"
        w = ft_mod.build_timing_weights(dataset, lasso_predictions, factors, cfg)
        portfolios[name] = ft_mod.summarize_portfolio(name, w, dataset, cfg)

        bench_name = f"Benchmark (EW) - {uni_name}"
        w_bench = ft_mod.equal_weight_benchmark(lasso_predictions.index, factors)
        portfolios[bench_name] = ft_mod.summarize_portfolio(bench_name, w_bench, dataset, cfg)

    return portfolios


def build_comparison_table(portfolios: dict) -> pd.DataFrame:
    return pd.DataFrame([p["metrics"] for p in portfolios.values()]).set_index("Portfolio")


def build_subperiod_tables(portfolios: dict, cfg: Config) -> dict:
    """Returns {subperiod_label: comparison_df}, metrics recomputed on the date slice."""
    out = {}
    for label, (start, end) in cfg.oos_subperiods.items():
        rows = []
        for name, p in portfolios.items():
            returns, turnover = p["returns"], p["turnover"]
            r_mask = returns.index >= start
            t_mask = turnover.index >= start
            if end is not None:
                r_mask &= returns.index <= end
                t_mask &= turnover.index <= end
            rows.append(_metrics_row(name, returns[r_mask], turnover[t_mask], cfg))
        out[label] = pd.DataFrame(rows).set_index("Portfolio")
    return out
