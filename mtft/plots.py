"""
All plotting for the MTFT pipeline.

Uses a fixed categorical color order (never reassigned when the series
subset changes) drawn from a validated colorblind-safe palette, a single
blue/red diverging pair for actual-vs-model comparisons, and a light,
recessive chart chrome (muted gridlines/axes, no dual axes, no rainbow).
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

from config import Config

# ---------------------------------------------------------------------------
# Palette (validated categorical order; see dataviz skill reference palette)
# ---------------------------------------------------------------------------
CATEGORICAL = ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834"]
BLUE, AQUA, YELLOW, GREEN, VIOLET, RED, MAGENTA, ORANGE = CATEGORICAL

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

FACTOR_COLORS = {"Mkt-RF": BLUE, "SMB": AQUA, "HML": YELLOW, "RMW": GREEN, "CMA": VIOLET}
MACRO_COLORS = {"oil": BLUE, "pot_output": AQUA, "term_spread": YELLOW, "liquidity": GREEN}
STRATEGY_COLORS = {"FI": BLUE, "MT": AQUA, "FT": YELLOW, "AT": GREEN, "BN": VIOLET}

NBER_RECESSIONS = [
    ("1969-12-01", "1970-11-01"), ("1973-11-01", "1975-03-01"), ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"), ("1990-07-01", "1991-03-01"), ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"), ("2020-02-01", "2020-04-01"),
]


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
        "font.size": 10,
        "legend.frameon": False,
    })


def _shade_recessions(ax, index):
    lo, hi = index.min(), index.max()
    for start, end in NBER_RECESSIONS:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if e >= lo and s <= hi:
            ax.axvspan(max(s, lo), min(e, hi), color=INK_MUTED, alpha=0.10, lw=0)


def _save(fig, name: str, cfg: Config) -> str:
    os.makedirs(cfg.results_dir, exist_ok=True)
    path = os.path.join(cfg.results_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_figure2(macro_driver_levels: pd.DataFrame, factor_prices: pd.DataFrame, cfg: Config) -> str:
    """Panel A uses small multiples (one axis per driver) since the term spread's
    cumulative sum lives on a completely different scale (0-300+) from the
    cumulative log-return drivers (near 0) -- overlaying them on one axis would
    make oil and potential output invisible."""
    _setup_style()
    n_macro = len(macro_driver_levels.columns)
    fig = plt.figure(figsize=(10, 10.5))
    gs = fig.add_gridspec(2, n_macro, height_ratios=[1, 1.6], hspace=0.35, wspace=0.35)

    for i, col in enumerate(macro_driver_levels.columns):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(macro_driver_levels.index, macro_driver_levels[col], color=MACRO_COLORS.get(col, INK_MUTED), lw=1.4)
        _shade_recessions(ax, macro_driver_levels.index)
        ax.set_title(col, loc="left", fontsize=9, color=INK_SECONDARY)
        ax.tick_params(labelsize=7)
    fig.text(0.06, 0.94, "Panel A: Macro Drivers (cumulative, each on its own scale)", fontsize=11, color=INK_PRIMARY)

    ax = fig.add_subplot(gs[1, :])
    for col in factor_prices.columns:
        label = cfg.factor_labels.get(col, col)
        ax.plot(factor_prices.index, factor_prices[col], color=FACTOR_COLORS.get(col, INK_MUTED), lw=1.6, label=label)
    _shade_recessions(ax, factor_prices.index)
    ax.set_title("Panel B: Factor Prices (cumulative log returns)", loc="left", color=INK_PRIMARY, fontsize=11)
    ax.legend(loc="upper left", fontsize=8, ncol=5)

    fig.suptitle("Figure 2: Macro Drivers and Factor Prices", fontsize=13, y=0.99)
    return _save(fig, "figure2_macro_and_prices", cfg)


def plot_figure1a(factor_price: pd.Series, fitted: pd.Series, factor_label: str, cfg: Config) -> str:
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(factor_price.index, factor_price.values, color=BLUE, lw=1.8, label=f"{factor_label} actual price")
    ax.plot(fitted.index, fitted.values, color=RED, lw=1.4, ls="--", label=f"{factor_label} macro-implied equilibrium")
    _shade_recessions(ax, factor_price.index)
    ax.set_title(f"Figure 1a: {factor_label} Price vs. Macro-Implied Equilibrium", loc="left", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, "figure1a_hml_price_vs_equilibrium", cfg)


def plot_figure1b(predictive_result: dict, factor_label: str, cfg: Config) -> str:
    """predictive_result: {'model': OLS results, 'Y': pd.Series (indexed)}."""
    actual = predictive_result["Y"]
    fitted = pd.Series(predictive_result["model"].fittedvalues, index=actual.index)

    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(actual.index, actual.values, color=BLUE, lw=1.4, label=f"{factor_label} actual return")
    ax.plot(fitted.index, fitted.values, color=RED, lw=1.4, ls="--", label=f"{factor_label} fitted return (ECT model)")
    ax.axhline(0, color=BASELINE, lw=0.8)
    ax.set_title(f"Figure 1b: {factor_label} Actual vs. Fitted Returns", loc="left", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, "figure1b_hml_actual_vs_fitted_returns", cfg)


def plot_crisis_distributions(results: dict, years: list[int], period_label: str, cfg: Config) -> str:
    """Figure 5 style: KDE of predicted annual return, CER vs FECM, one subplot per year."""
    _setup_style()
    fig, axes = plt.subplots(1, len(years), figsize=(6 * len(years), 4.5), sharey=False)
    if len(years) == 1:
        axes = [axes]

    for ax, year in zip(axes, years):
        r = results[year]
        for dist, color, name in [(r["cer_dist"], INK_MUTED, "CER"), (r["fecm_dist"], BLUE, "Macro-FECM")]:
            kde = gaussian_kde(dist)
            grid = np.linspace(dist.min(), dist.max(), 400)
            ax.plot(grid, kde(grid), color=color, lw=1.8, label=name)
            ax.fill_between(grid, kde(grid), color=color, alpha=0.12)
        ax.axvline(r["realized"], color=RED, lw=1.8, ls="--", label="Realized")
        ax.set_title(f"{year}", loc="left", fontsize=11)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle(f"Figure 5: Predicted Value-Leg Return Distributions -- {period_label}", fontsize=13, y=1.02)
    return _save(fig, f"figure5_predicted_dist_{period_label.lower().replace(' ', '_')}", cfg)


def plot_cumulative_wealth_2020(strategy_returns: dict, strategies: list[str], cfg: Config, name: str) -> str:
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for strat in strategies:
        r = strategy_returns[strat]
        wealth = (1.0 + r).cumprod()
        ax.plot(wealth.index, wealth.values, color=STRATEGY_COLORS.get(strat, INK_MUTED), lw=1.8, label=strat)
    ax.axhline(1.0, color=BASELINE, lw=0.8)
    ax.set_title("Cumulative Wealth, 2020Q1 - Latest ($1 start)", loc="left", fontsize=12)
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, name, cfg)


def plot_ect_timeseries(ects: pd.DataFrame, cfg: Config, zoom_start: str | None = None) -> str:
    _setup_style()
    n_panels = 2 if zoom_start else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4.5 * n_panels))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for col in ects.columns:
        ax.plot(ects.index, ects[col], color=FACTOR_COLORS.get(col, INK_MUTED), lw=1.3, label=col)
    ax.axhline(0, color=BASELINE, lw=0.8)
    _shade_recessions(ax, ects.index)
    ax.set_title("ECT Time Series (full sample)", loc="left", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, ncol=5)

    if zoom_start:
        ax = axes[1]
        zoomed = ects[ects.index >= zoom_start]
        for col in zoomed.columns:
            ax.plot(zoomed.index, zoomed[col], color=FACTOR_COLORS.get(col, INK_MUTED), lw=1.6, marker="o", ms=3, label=col)
        ax.axhline(0, color=BASELINE, lw=0.8)
        ax.set_title(f"ECT Time Series (zoom: {zoom_start} onward)", loc="left", fontsize=11)
        ax.legend(loc="upper left", fontsize=8, ncol=5)

    return _save(fig, "ect_timeseries", cfg)


def plot_predicted_vs_realized(predictive_results: dict, cfg: Config) -> str:
    _setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    all_vals = []
    for j in cfg.factors:
        model = predictive_results[j]["model"]
        fitted = model.fittedvalues
        actual = predictive_results[j]["Y"].values
        label = cfg.factor_labels[j]
        ax.scatter(fitted, actual, color=FACTOR_COLORS.get(j, INK_MUTED), s=14, alpha=0.6, label=label)
        all_vals.extend(list(fitted) + list(actual))

    lo, hi = min(all_vals), max(all_vals)
    ax.plot([lo, hi], [lo, hi], color=BASELINE, lw=1.0, ls="--")
    ax.set_xlabel("Predicted quarterly return")
    ax.set_ylabel("Realized quarterly return")
    ax.set_title("Predicted vs. Realized Returns (all factors pooled)", loc="left", fontsize=12)
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, "predicted_vs_realized_scatter", cfg)
