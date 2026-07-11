"""
All visualizations for the TSMOM study, saved as PNG files to results/.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import backtest as bt
from config import Config

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


def _savefig(fig, results_dir: str, filename: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    fig.savefig(path)
    plt.close(fig)


def tstat_by_lag(normalized_returns: pd.DataFrame, max_lag: int) -> pd.Series:
    """Pooled regression of x_t on x_{t-lag}, for lag = 1..max_lag.

    Returns the slope t-statistic at each lag (Figure 1 of the paper).
    """
    tstats = {}
    for lag in range(1, max_lag + 1):
        y = normalized_returns
        x_lag = normalized_returns.shift(lag)
        y_flat = y.values.flatten()
        x_flat = x_lag.values.flatten()
        mask = ~np.isnan(y_flat) & ~np.isnan(x_flat)
        if mask.sum() < 30:
            tstats[lag] = np.nan
            continue
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(x_flat[mask], y_flat[mask])
        tstats[lag] = slope / stderr if stderr > 0 else np.nan
    return pd.Series(tstats)


def plot_tstat_by_lag(normalized_returns: pd.DataFrame, max_lag: int, results_dir: str) -> pd.Series:
    tstats = tstat_by_lag(normalized_returns, max_lag)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(tstats.index, tstats.values, color="#3b6ea5")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1.96, color="red", linestyle="--", linewidth=0.8, label="+/- 1.96")
    ax.axhline(-1.96, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag (months)")
    ax.set_ylabel("t-statistic")
    ax.set_title("Figure 1: Pooled t-statistics of lagged vol-scaled returns")
    ax.legend()
    _savefig(fig, results_dir, "01_tstat_by_lag.png")
    return tstats


def plot_cumulative_returns(tsmom_returns: pd.Series, passive_returns: pd.Series, results_dir: str) -> None:
    tsmom_cum = bt.cumulative_returns(tsmom_returns)
    passive_cum = bt.cumulative_returns(passive_returns)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(tsmom_cum.index, tsmom_cum.values, label="TSMOM", color="#2a7f3f", linewidth=1.5)
    ax.plot(passive_cum.index, passive_cum.values, label="Passive Long", color="#a53b3b", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Growth of $1 (log scale)")
    ax.set_title("TSMOM vs Passive Long: Cumulative Returns")
    ax.legend()
    _savefig(fig, results_dir, "02_cumulative_returns.png")


def _compound_to_quarterly(monthly_returns: pd.Series) -> pd.Series:
    periods = pd.PeriodIndex(monthly_returns.index, freq="Q")
    grouped = (1.0 + monthly_returns).groupby(periods).prod() - 1.0
    grouped.index = grouped.index.to_timestamp(how="end").normalize()
    return grouped


def plot_smile(tsmom_returns: pd.Series, benchmark_monthly_excess_returns: pd.Series, results_dir: str) -> None:
    tsmom_q = _compound_to_quarterly(tsmom_returns)
    bench_q = _compound_to_quarterly(benchmark_monthly_excess_returns.dropna())
    data = pd.concat([tsmom_q.rename("tsmom"), bench_q.rename("bench")], axis=1).dropna()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data["bench"] * 100, data["tsmom"] * 100, color="#3b6ea5", alpha=0.7, edgecolor="white")
    if len(data) > 2:
        coeffs = np.polyfit(data["bench"], data["tsmom"], 2)
        xs = np.linspace(data["bench"].min(), data["bench"].max(), 100)
        ys = np.polyval(coeffs, xs)
        ax.plot(xs * 100, ys * 100, color="#a53b3b", linewidth=1.5, label="Quadratic fit")
        ax.legend()
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.axvline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("S&P 500 Quarterly Excess Return (%)")
    ax.set_ylabel("TSMOM Quarterly Return (%)")
    ax.set_title("Figure 4: TSMOM Smile vs S&P 500")
    _savefig(fig, results_dir, "03_tsmom_smile.png")


def plot_sharpe_by_instrument(instrument_returns: pd.DataFrame, cfg: Config, results_dir: str) -> pd.Series:
    sharpes = bt.per_instrument_sharpe(instrument_returns, cfg).sort_values()
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#a53b3b" if v < 0 else "#2a7f3f" for v in sharpes.values]
    ax.barh(sharpes.index, sharpes.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Annualized Sharpe Ratio")
    ax.set_title("TSMOM Sharpe Ratio by Instrument")
    _savefig(fig, results_dir, "04_sharpe_by_instrument.png")
    return sharpes


def plot_drawdown(tsmom_returns: pd.Series, results_dir: str) -> None:
    dd = bt.drawdown_series(tsmom_returns)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(dd.index, dd.values * 100, 0, color="#a53b3b", alpha=0.6)
    ax.plot(dd.index, dd.values * 100, color="#a53b3b", linewidth=1.0)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("TSMOM Portfolio Underwater Plot")
    _savefig(fig, results_dir, "05_drawdown.png")


def plot_rolling_sharpe(tsmom_returns: pd.Series, cfg: Config, results_dir: str, window: int = 36) -> None:
    rs = bt.rolling_sharpe(tsmom_returns, window, cfg.months_per_year)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rs.index, rs.values, color="#3b6ea5", linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{window}-Month Rolling Sharpe Ratio")
    ax.set_title("TSMOM Rolling Sharpe Ratio")
    _savefig(fig, results_dir, "06_rolling_sharpe.png")


def generate_all_plots(
    normalized_returns: pd.DataFrame,
    tsmom_returns: pd.Series,
    passive_returns: pd.Series,
    benchmark_monthly_excess_returns: pd.Series,
    tsmom_instrument_returns: pd.DataFrame,
    cfg: Config,
) -> dict:
    """Generate and save all six plots. Returns any computed series needed
    for the printed summary (t-stats, per-instrument Sharpes).
    """
    tstats = plot_tstat_by_lag(normalized_returns, cfg.ff_lags_for_tstat_plot, cfg.results_dir)
    plot_cumulative_returns(tsmom_returns, passive_returns, cfg.results_dir)
    plot_smile(tsmom_returns, benchmark_monthly_excess_returns, cfg.results_dir)
    sharpes = plot_sharpe_by_instrument(tsmom_instrument_returns, cfg, cfg.results_dir)
    plot_drawdown(tsmom_returns, cfg.results_dir)
    plot_rolling_sharpe(tsmom_returns, cfg, cfg.results_dir)

    return {"tstats_by_lag": tstats, "sharpe_by_instrument": sharpes}
