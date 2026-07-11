"""
Step 7: Extended out-of-sample backtest, 2020Q1 - latest available.

Two variants:
  - Fixed-parameter: long-run + predictive regressions estimated once on
    data through 2019Q4, coefficients frozen and applied to every quarter
    from 2020Q1 onward.
  - Expanding window: both regressions re-estimated every quarter using
    only data available up to that point (same real-time logic as Step 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

import ect as ect_mod
import predictive as pred_mod
from config import Config
from portfolio import STRATEGIES, compute_dynamic_expected_returns, compute_weights, portfolio_returns


def _clip_to_window(series_or_df, cfg: Config):
    """Restrict to [extended_backtest_start, extended_backtest_end]."""
    out = series_or_df[series_or_df.index >= cfg.extended_backtest_start]
    if cfg.extended_backtest_end is not None:
        out = out[out.index <= cfg.extended_backtest_end]
    return out


def fixed_parameter_backtest(
    factor_log_returns: pd.DataFrame, factor_prices: pd.DataFrame, macro_driver_levels: pd.DataFrame, cfg: Config
) -> dict:
    """Estimate on data through cfg.extended_train_cutoff; freeze coefficients; apply to 2020Q1+."""
    train = factor_prices.index <= cfg.extended_train_cutoff
    train_prices, train_macro, train_returns = factor_prices.loc[train], macro_driver_levels.loc[train], factor_log_returns.loc[train]

    lr_models = {j: ect_mod.estimate_long_run_regression(train_prices[j], train_macro, hac=False) for j in cfg.factors}
    train_ects = pd.DataFrame({j: lr_models[j].resid for j in cfg.factors})

    pred_models = {}
    for j in cfg.factors:
        model, _, _ = pred_mod.run_predictive_regression(train_returns[j], train_ects[j])
        pred_models[j] = {"model": model}

    full_trend = ect_mod.build_trend(factor_prices.index)
    ects_fixed_coef = pd.DataFrame({
        j: ect_mod.apply_fitted_long_run(lr_models[j], factor_prices[j], macro_driver_levels, full_trend)
        for j in cfg.factors
    })

    unconditional_means = train_returns.mean()
    cov = train_returns.cov()
    dynamic_er = compute_dynamic_expected_returns(ects_fixed_coef, pred_models, cfg)

    strategy_returns, strategy_weights = {}, {}
    for strat in STRATEGIES:
        w = compute_weights(dynamic_er, unconditional_means, cov, strat, cfg)
        r = portfolio_returns(w, factor_log_returns, cfg)
        r = _clip_to_window(r, cfg)
        strategy_returns[strat] = r
        strategy_weights[strat] = w

    ects_extended = ects_fixed_coef[ects_fixed_coef.index >= cfg.extended_train_cutoff]
    if cfg.extended_backtest_end is not None:
        ects_extended = ects_extended[ects_extended.index <= cfg.extended_backtest_end]
    return {"returns": strategy_returns, "weights": strategy_weights, "ects": ects_extended, "dynamic_er": dynamic_er}


def expanding_window_backtest(
    factor_log_returns: pd.DataFrame, factor_prices: pd.DataFrame, macro_driver_levels: pd.DataFrame, cfg: Config
) -> dict:
    """Re-estimate long-run + predictive regressions every quarter using only data up to t-1."""
    full_trend = ect_mod.build_trend(factor_prices.index)
    start_idx = int(factor_prices.index.searchsorted(pd.Timestamp(cfg.extended_backtest_start)))
    end_idx = len(factor_prices)
    if cfg.extended_backtest_end is not None:
        end_idx = int(factor_prices.index.searchsorted(pd.Timestamp(cfg.extended_backtest_end), side="right"))

    dynamic_er_rows, dates = [], []
    for t in range(start_idx, end_idx):
        trend_train = full_trend.iloc[:t]
        price_train = factor_prices.iloc[:t]
        macro_train = macro_driver_levels.iloc[:t]
        returns_train = factor_log_returns.iloc[:t]

        row = {}
        for j in cfg.factors:
            X_lr = sm.add_constant(pd.concat([trend_train, macro_train[cfg.macro_drivers]], axis=1))
            lr_model = sm.OLS(price_train[j], X_lr).fit()
            ect_train = lr_model.resid

            Y_train = returns_train[j].iloc[1:]
            X_train = sm.add_constant(ect_train.iloc[:-1])
            pred_model = sm.OLS(Y_train.values, X_train.values).fit()

            row[j] = pred_model.params[0] + pred_model.params[1] * ect_train.iloc[-1]
        dynamic_er_rows.append(row)
        dates.append(factor_prices.index[t])

    dynamic_er = pd.DataFrame(dynamic_er_rows, index=dates)

    strategy_returns = {}
    for strat in STRATEGIES:
        rows = []
        for t_idx in dynamic_er.index:
            train_returns = factor_log_returns.loc[factor_log_returns.index < t_idx]
            unconditional_means = train_returns.mean()
            cov = train_returns.cov()
            w_t = compute_weights(dynamic_er.loc[[t_idx]], unconditional_means, cov, strat, cfg)
            rows.append(w_t.iloc[0])
        w = pd.DataFrame(rows, index=dynamic_er.index)
        r = portfolio_returns(w, factor_log_returns, cfg)
        strategy_returns[strat] = _clip_to_window(r, cfg)

    return {"returns": strategy_returns, "dynamic_er": dynamic_er}


def cumulative_wealth(returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    return start_value * (1.0 + returns).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    wealth = cumulative_wealth(returns)
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1.0
    return drawdown.min()


def annualized_sharpe(returns: pd.Series, cfg: Config) -> float:
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return np.nan
    return returns.mean() / returns.std(ddof=1) * np.sqrt(cfg.quarters_per_year)


def summarize_extended_backtest(strategy_returns: dict, cfg: Config) -> pd.DataFrame:
    rows = {}
    for strat, r in strategy_returns.items():
        rows[strat] = {
            "Ann. Mean (%)": r.mean() * cfg.quarters_per_year * 100,
            "Ann. Vol (%)": r.std(ddof=1) * np.sqrt(cfg.quarters_per_year) * 100,
            "Sharpe": annualized_sharpe(r, cfg),
            "Max Drawdown (%)": max_drawdown(r) * 100,
        }
    return pd.DataFrame(rows).T
