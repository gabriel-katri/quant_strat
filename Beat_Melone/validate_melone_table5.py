"""
ONE-OFF validation, separate from the paper's main results (which stay on
the 1975Q1-1999Q4 in-sample / 2000Q1-2026Q2 OOS window -- this script does
not touch that default anywhere).

Replicates the literal split-sample design behind Melone's Table 5 Sharpe
of 0.999 (Factor Timing), to sanity-check the implementation machinery in
isolation from two other known differences:
  - window: in-sample 1968Q1-1993Q4, OOS 1994Q1-2019Q4 (not our expanding-
    window design)
  - gamma=5 (already the new pipeline default as of this run, see config.py)
  - universe: our 5 factors (MKT, SMB, HML, RMW, CMA) only, vs. Melone's
    ~12 (FF5 + q5) -- NOT matched here, so an exact 0.999 hit isn't
    expected even if the window/gamma machinery is fully correct. This
    isolates window+gamma from the factor-count difference.

Design, mirroring "parameters estimated on the first half, tested on the
second half":
  - long-run regression (ln F_t = a0 + a1*t + beta'M_t + w_t) fit ONCE on
    the IS window only, then applied out-of-sample via `.predict()` on the
    OOS design rows (fixed coefficients, no re-estimation) -- same pattern
    as mtft's apply_fitted_long_run.
  - predictive regression (r_{t+1} = a + delta*ECT_t + v) fit ONCE on the
    IS window's own (ECT_t, r_{t+1}) pairs, then applied with fixed (a,
    delta) to forecast every OOS quarter's return from the prior quarter's
    ECT (which may be the last IS quarter's ECT, for the very first OOS
    forecast -- still no look-ahead).
  - covariance Sigma estimated ONCE on the IS window's factor returns and
    held fixed throughout the OOS test -- consistent with "everything
    estimated on the first half."
  - weights vol-targeted to cfg.target_vol_annual (10%), our project's
    convention throughout; NOTE this means gamma cannot mechanically
    change the resulting Sharpe here (or anywhere in this codebase) --
    gamma only enters the CER formula, never the weight construction. That
    is reported explicitly below rather than left implicit.

Real Pastor-Stambaugh liquidity (LIQ_V) covers 1968-2019, which is exactly
the full span needed here -- no -TEDRATE proxy splice required for this
one-off (unlike the main pipeline, whose OOS window runs through 2026).

Run with: python validate_melone_table5.py
"""

from __future__ import annotations

import sys
import warnings

import pandas as pd
import statsmodels.api as sm

import data as data_mod
import factor_timing as ft_mod
import melone_construct as construct_mod
import melone_ect as ect_mod
from config import Config

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda v: f"{v: .4f}")

IS_START, IS_END = "1968-01-01", "1993-12-31"
OOS_START, OOS_END = "1994-01-01", "2019-12-31"
MELONE_REPORTED_SHARPE = 0.999


def _section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def build_dataset(cfg: Config) -> pd.DataFrame:
    """FF5 + the 4 real macro drivers (no liquidity proxy needed -- OOS ends
    2019Q4, entirely within LIQ_V's own 1968-2019 coverage), 1-quarter lag."""
    ff5_monthly = data_mod.download_ff5_monthly()
    ff5 = data_mod.ff5_monthly_to_quarterly(ff5_monthly[cfg.factor_columns])

    oil = data_mod.build_oil_driver(cfg)
    pot = data_mod.build_potential_output_driver(cfg)
    term = data_mod.build_term_spread_driver(cfg)
    liq = data_mod._fetch_real_liquidity_quarterly().rename("LIQUIDITY_PS")

    macro = pd.concat([oil, pot, term, liq], axis=1)
    macro_lagged = macro.shift(1)

    df = pd.concat([ff5, macro_lagged[cfg.macro_columns]], axis=1)
    df = df[(df.index >= IS_START) & (df.index <= OOS_END)]
    return df.dropna(how="any")


def main() -> None:
    cfg = Config()

    _section("Building 1968-2019 dataset (real Pastor-Stambaugh liquidity, no proxy needed)")
    df = build_dataset(cfg)
    is_mask = (df.index >= IS_START) & (df.index <= IS_END)
    oos_mask = (df.index >= OOS_START) & (df.index <= OOS_END)
    print(f"Full span: {df.index.min().date()} to {df.index.max().date()}, {len(df)} quarters")
    print(f"In-sample [{IS_START} - {IS_END}]: {is_mask.sum()} quarters")
    print(f"Out-of-sample [{OOS_START} - {OOS_END}]: {oos_mask.sum()} quarters")
    print("(Melone's paper: 104 quarters each; small deviations here are the same "
          "'first row lost to the 1-quarter lag' effect seen throughout this project.)")

    factors = cfg.unconstrained_factors
    factor_prices = construct_mod.build_factor_prices(df, factors)
    macro_levels = construct_mod.build_macro_driver_levels(df, cfg.macro_columns)
    global_trend = ect_mod.build_trend(df.index)  # single consistent trend across IS+OOS

    is_idx, oos_idx = df.index[is_mask], df.index[oos_mask]

    _section("Fitting long-run regression ONCE on IS, applying fixed coefficients out-of-sample")
    X_full = sm.add_constant(pd.concat([global_trend, macro_levels], axis=1))
    ect_full_by_factor, is_beta_table = {}, {}
    for f in factors:
        X_is, y_is = X_full.loc[is_idx], factor_prices.loc[is_idx, f]
        lr_model = sm.OLS(y_is.values, X_is.values).fit()
        is_beta_table[f] = lr_model.params

        fitted_full = pd.Series(lr_model.predict(X_full.values), index=df.index)
        ect_full_by_factor[f] = factor_prices[f] - fitted_full  # IS rows = in-sample resid; OOS rows = true OOS residual

    beta_table = pd.DataFrame(is_beta_table, index=["const", "trend"] + list(cfg.macro_columns))
    print("IS-fitted long-run coefficients (held fixed for the entire OOS window):")
    print(beta_table.round(4).to_string())

    _section("Fitting predictive regression ONCE on IS, forecasting OOS returns with fixed (a, delta)")
    predictions = {}
    pred_params = {}
    for f in factors:
        ect = ect_full_by_factor[f]
        y_is = df.loc[is_idx, f].iloc[1:]
        x_is = ect.loc[is_idx].iloc[:-1]
        pred_model = sm.OLS(y_is.values, sm.add_constant(x_is.values)).fit()
        pred_params[f] = pred_model.params

        rows = []
        for d in oos_idx:
            t = df.index.get_loc(d)
            ect_lag = ect.iloc[t - 1]  # prior quarter's ECT -- may be the last IS quarter, still no look-ahead
            forecast = float(pred_model.params[0] + pred_model.params[1] * ect_lag)
            rows.append({"date": d, "actual": float(df.loc[d, f]), "predicted": forecast})
        predictions[f] = pd.DataFrame(rows).set_index("date")

    pred_table = pd.DataFrame(pred_params, index=["a", "delta"])
    print("IS-fitted predictive-regression coefficients (held fixed for the entire OOS window):")
    print(pred_table.round(4).to_string())

    _section("Portfolio: Sigma estimated ONCE on IS, held fixed through OOS; weights vol-targeted to 10%")
    sigma_is = df.loc[is_idx, factors].cov().to_numpy()
    r_hat = pd.concat({f: predictions[f]["predicted"] for f in factors}, axis=1)[factors]
    actual = pd.concat({f: predictions[f]["actual"] for f in factors}, axis=1)[factors]

    import numpy as np
    target_vol_q = cfg.target_vol_annual / (cfg.periods_per_year ** 0.5)
    rows = []
    for d in oos_idx:
        r_hat_d = r_hat.loc[d].to_numpy()
        raw_w = np.linalg.solve(sigma_is, r_hat_d)
        ex_ante_var = float(raw_w @ sigma_is @ raw_w)
        w = raw_w * (target_vol_q / np.sqrt(ex_ante_var)) if ex_ante_var >= 1e-14 else np.zeros_like(raw_w)
        rows.append(w)
    weights = pd.DataFrame(rows, index=oos_idx, columns=factors)
    portfolio_returns = (weights * actual).sum(axis=1)

    sharpe = ft_mod.sharpe_ratio(portfolio_returns, cfg)
    cer = ft_mod.certainty_equivalent_return(portfolio_returns, cfg.cer_gamma, cfg) * 100
    mdd = ft_mod.max_drawdown(portfolio_returns) * 100

    _section("RESULT: split-sample Static (Factor Timing), 5-factor Unconstrained, gamma=5")
    print(f"OOS window:      {oos_idx.min().date()} to {oos_idx.max().date()}  ({len(oos_idx)} quarters)")
    print(f"OOS Sharpe (ann.):        {sharpe:.3f}")
    print(f"OOS CER (%, ann., gamma={cfg.cer_gamma:.0f}):  {cer:.3f}")
    print(f"OOS Max drawdown (%):     {mdd:.3f}")
    print(f"\nMelone's reported Table 5 Sharpe (Factor Timing, ~12-factor FF5+q5 universe): {MELONE_REPORTED_SHARPE}")
    print(f"Ratio (ours / Melone's):  {sharpe / MELONE_REPORTED_SHARPE:.3f}")
    print("\nNote: gamma cannot mechanically move the Sharpe number above under this codebase's "
          "vol-targeting convention (weights are Sigma^-1 . r_hat rescaled to a fixed 10% annualized "
          "vol target; gamma only enters the CER formula, never the weight construction). The gamma=5 "
          "fix changes CER everywhere in this project but leaves every Sharpe/MDD/turnover number "
          "byte-identical to the gamma=3 versions -- confirmed by re-running the main pipeline before this script.")


if __name__ == "__main__":
    sys.exit(main())
