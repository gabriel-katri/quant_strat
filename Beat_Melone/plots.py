"""
Plots for the rolling-Lasso model: coefficient paths and predicted-vs-actual
returns, one figure per factor for each. Uses a fixed categorical color
order (never reassigned) so a given macro signal always has the same color
across every factor's plot.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from config import Config

CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE = CATEGORICAL

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

PREDICTOR_COLORS = {
    "OIL_WTI_LOGRET": BLUE,
    "POT_GDP_GROWTH": AQUA,
    "TERM_SPREAD_10Y3M": VIOLET,
    "LIQUIDITY_PS": RED,
}

FACTOR_COLORS = {
    "MKT_RF": BLUE,
    "SMB": AQUA,
    "HML": YELLOW,
    "RMW": GREEN,
    "CMA": VIOLET,
}

BETA_COLORS = {
    "const": INK_MUTED,
    "trend": ORANGE,
    "OIL_WTI_LOGRET": BLUE,
    "POT_GDP_GROWTH": AQUA,
    "TERM_SPREAD_10Y3M": VIOLET,
    "LIQUIDITY_PS": RED,
}

STRATEGY_COLORS = {
    "Static": BLUE,
    "Kalman (RW)": GREEN,
    "Kalman (MR)": YELLOW,
    "LASSO": VIOLET,
    "LASSO (matched Sigma)": VIOLET,
    "Benchmark (EW)": INK_MUTED,
}

Q_VARIANT_COLORS = {
    "q (MLE)": BLUE,
    "q/10": GREEN,
    "q/100": RED,
}


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "legend.frameon": False,
    })


def _save(fig, name: str, cfg: Config) -> str:
    os.makedirs(cfg.results_dir, exist_ok=True)
    path = os.path.join(cfg.results_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_coefficients_over_time(results: dict, cfg: Config) -> str:
    """One subplot per factor, one colored line per macro signal."""
    _setup_style()
    n = len(cfg.lasso_target_factors)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), sharex=True, constrained_layout=True)

    for ax, factor in zip(axes, cfg.lasso_target_factors):
        coef_df = results[factor]["coefficients"]
        for predictor in cfg.lasso_predictors:
            ax.plot(coef_df.index, coef_df[predictor], color=PREDICTOR_COLORS.get(predictor, INK_MUTED),
                     lw=1.3, label=predictor)
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_title(f"{factor}", loc="left", fontsize=10, color=INK_PRIMARY)

    axes[0].legend(loc="upper left", fontsize=7, ncol=3)
    fig.suptitle("Rolling Lasso Coefficients Over Time", fontsize=13)
    return _save(fig, "lasso_coefficients_over_time", cfg)


def plot_predicted_vs_actual(results: dict, cfg: Config) -> str:
    """One subplot per factor: actual vs predicted return time series."""
    _setup_style()
    n = len(cfg.lasso_target_factors)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True, constrained_layout=True)

    for ax, factor in zip(axes, cfg.lasso_target_factors):
        pred_df = results[factor]["predictions"]
        ax.plot(pred_df.index, pred_df["actual"], color=BLUE, lw=1.2, label="Actual")
        ax.plot(pred_df.index, pred_df["predicted"], color=RED, lw=1.2, ls="--", label="Predicted")
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_title(f"{factor}", loc="left", fontsize=10, color=INK_PRIMARY)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Rolling Lasso: Predicted vs. Actual Returns", fontsize=13)
    return _save(fig, "lasso_predicted_vs_actual", cfg)


def plot_melone_levels(factor_prices: pd.DataFrame, macro_levels: pd.DataFrame, cfg: Config) -> str:
    """Step 1 (Melone): factor log-price levels (top) and macro driver
    levels (bottom), both cumulative sums, sharing a time axis."""
    _setup_style()
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    for factor in factor_prices.columns:
        ax_top.plot(factor_prices.index, factor_prices[factor],
                     color=FACTOR_COLORS.get(factor, INK_MUTED), lw=1.4, label=factor)
    ax_top.axhline(0, color=BASELINE, lw=0.8)
    ax_top.set_ylabel("ln F_t (cum. log return)")
    ax_top.set_title("Factor price levels", loc="left", fontsize=10, color=INK_PRIMARY)
    ax_top.legend(loc="upper left", fontsize=8, ncol=5)

    for signal in macro_levels.columns:
        ax_bottom.plot(macro_levels.index, macro_levels[signal],
                        color=PREDICTOR_COLORS.get(signal, INK_MUTED), lw=1.4, label=signal)
    ax_bottom.axhline(0, color=BASELINE, lw=0.8)
    ax_bottom.set_ylabel("M_t (cum. sum)")
    ax_bottom.set_title("Macro driver levels", loc="left", fontsize=10, color=INK_PRIMARY)
    ax_bottom.legend(loc="upper left", fontsize=7, ncol=3)

    fig.suptitle("Melone Step 1: Factor Prices & Macro Driver Levels", fontsize=13)
    return _save(fig, "melone_levels", cfg)


_PORTFOLIO_COLORS = [BLUE, RED, INK_MUTED]


def plot_cumulative_returns(returns_by_portfolio: dict, cfg: Config, colors: list[str] | None = None,
                             filename: str = "factor_timing_cumulative_returns",
                             title: str = "Factor-Timing Portfolios: Cumulative Returns") -> str:
    """One line per portfolio: cumulative return since the first OOS date."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

    for (name, returns), color in zip(returns_by_portfolio.items(), colors or _PORTFOLIO_COLORS):
        cum = (1 + returns).cumprod() - 1
        ax.plot(cum.index, cum * 100, color=color, lw=1.5, label=name)

    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.set_ylabel("Cumulative return (%)")
    ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=13)
    return _save(fig, filename, cfg)


def plot_drawdown(returns_by_portfolio: dict, cfg: Config, colors: list[str] | None = None,
                   filename: str = "factor_timing_drawdown",
                   title: str = "Factor-Timing Portfolios: Drawdown") -> str:
    """One line per portfolio: drawdown from the running peak of cumulative wealth."""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

    for (name, returns), color in zip(returns_by_portfolio.items(), colors or _PORTFOLIO_COLORS):
        wealth = (1 + returns).cumprod()
        dd = wealth / wealth.cummax() - 1
        ax.plot(dd.index, dd * 100, color=color, lw=1.2, label=name)
        ax.fill_between(dd.index, dd * 100, 0, color=color, alpha=0.08)

    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(title, fontsize=13)
    return _save(fig, filename, cfg)


def plot_kalman_betas(kalman_results: dict, cfg: Config) -> str:
    """Step 2 (Kalman): one subplot per factor, one colored line per state
    (const, trend, and the 4 macro-driver loadings) -- shows how unstable
    the cointegrating coefficients are over time.

    Plots only from `rolling_window_periods` onward (2000Q1+): with only 6
    free states and a design matrix that starts out nearly collinear over a
    handful of points, the diffuse Kalman filter's very first few quarters
    (early in the calibration window) swing to implausible magnitudes
    before settling -- a standard diffuse-initialization/identifiability
    artifact, not a property of the model afterward. It has no bearing on
    the OOS results (which only ever use beta_t/ECT_t from 2000Q1 on); this
    just keeps the burn-in from blowing out the y-axis. Full series,
    burn-in included, are saved to the CSV.
    """
    _setup_style()
    n = len(cfg.unconstrained_factors)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), sharex=True, constrained_layout=True)

    burn_in = cfg.rolling_window_periods
    beta_cols = ["const", "trend"] + list(cfg.macro_columns)
    for ax, factor in zip(axes, cfg.unconstrained_factors):
        beta_t = kalman_results[factor]["beta_t"].iloc[burn_in:]
        for col in beta_cols:
            ax.plot(beta_t.index, beta_t[col], color=BETA_COLORS.get(col, INK_MUTED), lw=1.3, label=col)
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_title(f"{factor}", loc="left", fontsize=10, color=INK_PRIMARY)

    axes[0].legend(loc="upper left", fontsize=7, ncol=3)
    fig.suptitle("Kalman Filter: Time-Varying Cointegrating Coefficients (beta_t), 2000Q1 onward", fontsize=13)
    return _save(fig, "melone_kalman_betas", cfg)


def plot_ect_static_vs_kalman(static_ects: pd.DataFrame, kalman_results: dict, cfg: Config) -> str:
    """One subplot per factor: static (full-sample) ECT vs. Kalman (filtered) ECT.

    Plots only from `rolling_window_periods` onward -- see plot_kalman_betas
    for why the Kalman ECT's early calibration-window values are excluded.
    """
    _setup_style()
    n = len(cfg.unconstrained_factors)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True, constrained_layout=True)

    burn_in = cfg.rolling_window_periods
    for ax, factor in zip(axes, cfg.unconstrained_factors):
        static_ect = static_ects[factor].iloc[burn_in:]
        ax.plot(static_ect.index, static_ect, color=BLUE, lw=1.2, label="Static ECT")
        kalman_ect = kalman_results[factor]["ect"].iloc[burn_in:]
        ax.plot(kalman_ect.index, kalman_ect, color=RED, lw=1.2, ls="--", label="Kalman ECT")
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_title(f"{factor}", loc="left", fontsize=10, color=INK_PRIMARY)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Static vs. Kalman Error-Correction Term (ECT), 2000Q1 onward", fontsize=13)
    return _save(fig, "melone_ect_static_vs_kalman", cfg)


def plot_kalman_fit_vs_actual(factor_prices: pd.DataFrame, kalman_results: dict, cfg: Config) -> str:
    """Diagnostic: ln F_t (actual) vs. the Kalman fitted value (beta_t' M_t), one
    subplot per factor. If fitted tracks actual too closely (ECT ~ 0
    everywhere), the filter is absorbing genuine mean-reversion signal into
    beta_t drift instead of leaving it in the residual."""
    _setup_style()
    n = len(cfg.unconstrained_factors)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True, constrained_layout=True)

    burn_in = cfg.rolling_window_periods
    for ax, factor in zip(axes, cfg.unconstrained_factors):
        actual = factor_prices[factor].iloc[burn_in:]
        fitted = (factor_prices[factor] - kalman_results[factor]["ect"]).iloc[burn_in:]
        ax.plot(actual.index, actual, color=BLUE, lw=1.3, label="ln F_t (actual)")
        ax.plot(fitted.index, fitted, color=RED, lw=1.1, ls="--", label="Kalman fitted (beta_t' M_t)")
        ax.set_title(f"{factor}", loc="left", fontsize=10, color=INK_PRIMARY)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Kalman Diagnostic: Fitted vs. Actual Factor Price, 2000Q1 onward", fontsize=13)
    return _save(fig, "melone_kalman_fit_vs_actual", cfg)


def plot_kalman_ect_q_sensitivity(ect_variants: dict, cfg: Config) -> str:
    """Diagnostic: static-q Kalman ECT vs. ECT with q manually divided by 10
    and by 100, one subplot per factor. `ect_variants` is {label: {factor:
    ect_series}} for labels 'q (MLE)', 'q/10', 'q/100'."""
    _setup_style()
    n = len(cfg.unconstrained_factors)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True, constrained_layout=True)

    burn_in = cfg.rolling_window_periods
    for ax, factor in zip(axes, cfg.unconstrained_factors):
        for label, ect_by_factor in ect_variants.items():
            ect = ect_by_factor[factor].iloc[burn_in:]
            ax.plot(ect.index, ect, color=Q_VARIANT_COLORS.get(label, INK_MUTED), lw=1.2, label=label)
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_title(f"{factor}", loc="left", fontsize=10, color=INK_PRIMARY)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Kalman ECT Sensitivity to q (state noise), 2000Q1 onward", fontsize=13)
    return _save(fig, "melone_kalman_ect_q_sensitivity", cfg)
