"""
Step 6: Portfolio construction and backtest.

Five mean-variance strategies over the 5 factors, differing only in which
factors get a dynamic (ECT-based) expected return vs. an unconditional
(full-sample mean) expected return:

  FI (Factor Investing): all factors static (unconditional mean)
  MT (Market Timing):    only MKT dynamic, rest static
  AT (Anomaly Timing):   MKT static, SMB/HML/RMW/CMA dynamic
  FT (Factor Timing):    all factors dynamic
  BN (Beta Neutral):     same as AT, but MKT is dropped from the
                          optimization entirely (zero weight by construction,
                          not just a static forecast)

Weights follow the standard quadratic-utility optimum w = (1/gamma) *
Sigma^-1 * mu -- the prompt's w = Sigma^-1 * E[f] omits the 1/gamma scaling,
but gamma is used later for the utility/fee calculation, so we apply it
consistently at the weight stage too (otherwise portfolio variance is
arbitrary and utility comparisons across strategies are not meaningful).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config

STRATEGIES = ["FI", "MT", "FT", "AT", "BN"]

_DYNAMIC_FACTORS = {
    "FI": set(),
    "MT": {"Mkt-RF"},
    "AT": {"SMB", "HML", "RMW", "CMA"},
    "FT": {"Mkt-RF", "SMB", "HML", "RMW", "CMA"},
    "BN": {"SMB", "HML", "RMW", "CMA"},
}


def compute_dynamic_expected_returns(ects: pd.DataFrame, predictive_results: dict, cfg: Config) -> pd.DataFrame:
    """E_t[f_{t+1}] = alpha_hat + delta_hat * ECT_t using given (full-sample or
    training-sample) predictive-regression coefficients."""
    out = {}
    for j in cfg.factors:
        model = predictive_results[j]["model"]
        alpha, delta = model.params[0], model.params[1]
        out[j] = alpha + delta * ects[j]
    return pd.DataFrame(out)


def _expected_return_vector(t_idx, dynamic_er: pd.DataFrame, unconditional_means: pd.Series, dynamic_factors: set, cfg: Config) -> pd.Series:
    mu = unconditional_means.copy()
    for j in dynamic_factors:
        mu[j] = dynamic_er.loc[t_idx, j]
    return mu


def compute_weights(
    dynamic_er: pd.DataFrame, unconditional_means: pd.Series, cov: pd.DataFrame, strategy: str, cfg: Config
) -> pd.DataFrame:
    """Weight time series w_t for one strategy, one column per factor (0 for excluded factors)."""
    dynamic_factors = _DYNAMIC_FACTORS[strategy]
    active_factors = [f for f in cfg.factors if not (strategy == "BN" and f == "Mkt-RF")]

    cov_active = cov.loc[active_factors, active_factors]
    cov_inv = np.linalg.inv(cov_active.values)

    weights = pd.DataFrame(0.0, index=dynamic_er.index, columns=cfg.factors)
    for t_idx in dynamic_er.index:
        mu = _expected_return_vector(t_idx, dynamic_er, unconditional_means, dynamic_factors, cfg)
        mu_active = mu[active_factors].values
        w_active = (1.0 / cfg.risk_aversion) * cov_inv @ mu_active
        weights.loc[t_idx, active_factors] = w_active
    return weights


def portfolio_returns(weights: pd.DataFrame, factor_log_returns: pd.DataFrame, cfg: Config) -> pd.Series:
    """R_p,t = w_t' * f_{t+1}: weights formed at t using time-t info, applied to next quarter's return."""
    w_shifted = weights.shift(1).dropna(how="all")
    common_idx = w_shifted.index.intersection(factor_log_returns.index)
    w_shifted = w_shifted.loc[common_idx]
    f = factor_log_returns.loc[common_idx, cfg.factors]
    returns = (w_shifted[cfg.factors] * f).sum(axis=1)
    return returns


def annualized_sharpe(returns: pd.Series, cfg: Config) -> float:
    if returns.std(ddof=1) == 0 or len(returns) < 2:
        return np.nan
    return returns.mean() / returns.std(ddof=1) * np.sqrt(cfg.quarters_per_year)


def certainty_equivalent_fee(returns: pd.Series, static_returns: pd.Series, cfg: Config) -> float:
    """Annualized fee (%) an investor would pay for the utility gain of timing vs. FI."""
    u_timing = returns.mean() - 0.5 * cfg.risk_aversion * returns.var(ddof=1)
    u_static = static_returns.mean() - 0.5 * cfg.risk_aversion * static_returns.var(ddof=1)
    delta_u = u_timing - u_static
    return delta_u * cfg.quarters_per_year * 100.0


def run_all_strategies(
    factor_log_returns: pd.DataFrame, ects: pd.DataFrame, predictive_results: dict, cfg: Config
) -> dict:
    """Full-sample (in-sample) strategy weights and returns, keyed by strategy name."""
    unconditional_means = factor_log_returns.mean()
    cov = factor_log_returns.cov()
    dynamic_er = compute_dynamic_expected_returns(ects, predictive_results, cfg)

    out = {}
    for strat in STRATEGIES:
        w = compute_weights(dynamic_er, unconditional_means, cov, strat, cfg)
        r = portfolio_returns(w, factor_log_returns, cfg)
        out[strat] = {"weights": w, "returns": r}
    return out


def run_oos_split_strategies(
    factor_log_returns: pd.DataFrame, factor_prices: pd.DataFrame, macro_driver_levels: pd.DataFrame, cfg: Config
) -> dict:
    """OOS Sharpe: estimate everything on the training half, apply forward, unseen, to the test half."""
    import ect as ect_mod
    import predictive as pred_mod

    train = factor_log_returns.index < cfg.is_oos_split_date
    test = ~train

    train_prices = factor_prices.loc[train]
    train_macro = macro_driver_levels.loc[train]
    train_returns = factor_log_returns.loc[train]

    lr_models = {j: ect_mod.estimate_long_run_regression(train_prices[j], train_macro, hac=False) for j in cfg.factors}
    train_ects = pd.DataFrame({j: lr_models[j].resid for j in cfg.factors})
    pred_models = {}
    for j in cfg.factors:
        model, Y, X = pred_mod.run_predictive_regression(train_returns[j], train_ects[j])
        pred_models[j] = {"model": model}

    unconditional_means = train_returns.mean()
    cov = train_returns.cov()

    full_trend = ect_mod.build_trend(factor_prices.index)
    full_ects_from_train_coefs = pd.DataFrame({
        j: ect_mod.apply_fitted_long_run(lr_models[j], factor_prices[j], macro_driver_levels, full_trend)
        for j in cfg.factors
    })
    dynamic_er = compute_dynamic_expected_returns(full_ects_from_train_coefs, pred_models, cfg)

    out = {}
    for strat in STRATEGIES:
        w_full = compute_weights(dynamic_er, unconditional_means, cov, strat, cfg)
        w_test = w_full.loc[test]
        r_test = portfolio_returns(w_test, factor_log_returns, cfg)
        out[strat] = r_test
    return out


def build_table5(is_results: dict, oos_returns: dict, cfg: Config) -> pd.DataFrame:
    static_returns = is_results["FI"]["returns"]
    rows = {"IS Sharpe": {}, "OOS Sharpe": {}, "Fee (%)": {}}
    for strat in STRATEGIES:
        rows["IS Sharpe"][strat] = annualized_sharpe(is_results[strat]["returns"], cfg)
        rows["OOS Sharpe"][strat] = annualized_sharpe(oos_returns[strat], cfg)
        rows["Fee (%)"][strat] = certainty_equivalent_fee(is_results[strat]["returns"], static_returns, cfg)
    return pd.DataFrame(rows).T[STRATEGIES]
