"""
Central configuration for the TSMOM strategy.

All tunable parameters live here so the rest of the codebase never hard-codes
a universe member, a lookback window, or a target volatility.
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Universe: futures proxied by liquid ETFs available on yfinance
# ---------------------------------------------------------------------------
UNIVERSE = {
    # Equity indexes
    "SPY": "Equity - US",
    "EWJ": "Equity - Japan",
    "EWG": "Equity - Germany",
    "EWU": "Equity - UK",
    "EWA": "Equity - Australia",
    "EWQ": "Equity - France",
    "EWP": "Equity - Spain",
    "EWN": "Equity - Netherlands",
    "EWI": "Equity - Italy",
    # Bonds
    "TLT": "Bond - US Long",
    "IEF": "Bond - US 10y",
    "SHY": "Bond - US 2y",
    "BWX": "Bond - Intl",
    # Commodities
    "GLD": "Commodity - Gold",
    "SLV": "Commodity - Silver",
    "USO": "Commodity - Oil",
    "UNG": "Commodity - Nat Gas",
    "DBA": "Commodity - Agriculture",
    "DBB": "Commodity - Base Metals",
    # Currencies
    "FXA": "Currency - AUD",
    "FXE": "Currency - EUR",
    "FXB": "Currency - GBP",
    "FXY": "Currency - JPY",
    "FXC": "Currency - CAD",
    "FXF": "Currency - CHF",
}

TICKERS = list(UNIVERSE.keys())

# Risk-free proxy: 3-month T-bill discount rate (^IRX, annualized, in percent)
RISK_FREE_TICKER = "^IRX"

# Benchmark for the smile plot / factor regressions
BENCHMARK_TICKER = "SPY"

# ---------------------------------------------------------------------------
# Data parameters
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 261
MONTHS_PER_YEAR = 12

# yfinance download start; each ticker will actually start at its own
# inception date since yfinance simply returns what's available.
DATA_START = "1990-01-01"
DATA_END = None  # None -> up to today

# ---------------------------------------------------------------------------
# Ex-ante volatility (Eq. 1): EWMA of squared daily returns
# ---------------------------------------------------------------------------
VOL_COM = 60.0  # center of mass in days
VOL_ANNUALIZATION_FACTOR = TRADING_DAYS_PER_YEAR

# ---------------------------------------------------------------------------
# TSMOM signal & position sizing (Eq. 5)
# ---------------------------------------------------------------------------
LOOKBACK_MONTHS = 12
VOL_TARGET = 0.40  # 40% annualized volatility target per position

# Cap on position_size = VOL_TARGET / sigma, i.e. max leverage per
# instrument. ETF proxies for very low-duration instruments (e.g. a 2y
# T-bill ETF during a near-zero-rate regime) can have near-zero realized
# vol, which would otherwise imply extreme, unrealistic leverage under
# pure vol-target sizing -- a real futures desk would impose a similar
# guardrail. Set to None to disable.
MAX_POSITION_SIZE = 10.0

# ---------------------------------------------------------------------------
# Cross-sectional momentum (XSMOM) parameters
# ---------------------------------------------------------------------------
XSMOM_LOOKBACK_MONTHS = 12

# Minimum number of instruments with a valid signal required for a month to
# be included in the "diversified" portfolio average. Early history (mid-
# 1990s) has only 1-2 ETFs live, which turns the "diversified" portfolio
# into an undiversified, highly-levered single-name bet -- not a fair
# reflection of the strategy the paper describes. Set to 1 to disable.
MIN_INSTRUMENTS_FOR_PORTFOLIO = 5

# ---------------------------------------------------------------------------
# Transaction costs (bps per unit notional turnover) - default 0, structured
# so a cost model can be plugged in later without touching the backtest core.
# ---------------------------------------------------------------------------
TRANSACTION_COST_BPS = 0.0

# ---------------------------------------------------------------------------
# Factor regression
# ---------------------------------------------------------------------------
FF_LAGS_FOR_TSTAT_PLOT = 60  # months, for Figure 1 (t-stat by lag)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"


@dataclass
class Config:
    """Bundles all parameters for convenient passing / overriding in tests."""

    universe: dict = field(default_factory=lambda: dict(UNIVERSE))
    tickers: list = field(default_factory=lambda: list(TICKERS))
    risk_free_ticker: str = RISK_FREE_TICKER
    benchmark_ticker: str = BENCHMARK_TICKER
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    months_per_year: int = MONTHS_PER_YEAR
    data_start: str = DATA_START
    data_end: str | None = DATA_END
    vol_com: float = VOL_COM
    vol_annualization_factor: int = VOL_ANNUALIZATION_FACTOR
    lookback_months: int = LOOKBACK_MONTHS
    vol_target: float = VOL_TARGET
    max_position_size: float | None = MAX_POSITION_SIZE
    xsmom_lookback_months: int = XSMOM_LOOKBACK_MONTHS
    min_instruments_for_portfolio: int = MIN_INSTRUMENTS_FOR_PORTFOLIO
    transaction_cost_bps: float = TRANSACTION_COST_BPS
    ff_lags_for_tstat_plot: int = FF_LAGS_FOR_TSTAT_PLOT
    results_dir: str = RESULTS_DIR
