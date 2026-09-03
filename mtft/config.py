"""
Central configuration for the Macro Trends and Factor Timing (MTFT) strategy.

Replicates Favero, Melone & Tamoni (2022), "Macro Trends and Factor Timing".
All tunable parameters live here so the rest of the codebase never hard-codes
a URL, a date range, or a risk-aversion coefficient.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
SIZE_BM_6_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "6_Portfolios_2x3_CSV.zip"
)
FRED_CSV_TEMPLATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

FRED_SERIES = {
    "oil": "WTISPLC",       # WTI crude, monthly spot price
    "pot_output": "GDPPOT", # Real potential GDP, quarterly
    "gs10": "GS10",         # 10Y Treasury, monthly
    "tb3ms": "TB3MS",       # 3M T-bill, monthly
    "vix": "VIXCLS",        # VIX, daily -> used as liquidity proxy fallback
    "baa": "BAA",           # Moody's Baa corporate yield (robustness)
    "aaa": "AAA",           # Moody's Aaa corporate yield (robustness)
}

PASTOR_STAMBAUGH_URL = (
    "https://research.chicagobooth.edu/-/media/research/famamiller/data/liq_data_1962_2019.txt"
)

# Gold series was discontinued on FRED (LBMA licensing). Fall back to a
# Stooq daily spot-gold series for the Step 9 robustness check.
GOLD_STOOQ_URL = "https://stooq.com/q/d/l/?s=xauusd&i=m"

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
FACTOR_LABELS = {"Mkt-RF": "MKT", "SMB": "SMB", "HML": "HML", "RMW": "RMW", "CMA": "CMA"}

# Data-availability note (Step 0, Macro Driver 4): the paper-era Wharton URL
# for the Pastor-Stambaugh liquidity file is dead, but the same traded
# liquidity factor (LIQ_V) is mirrored on the Fama-Miller Center's data page
# (column 4 of PASTOR_STAMBAUGH_URL) back to 1968 -- it just stops in Dec
# 2019, so it can't cover the extended 2020-2026 sample. The core pipeline
# (Steps 1-8) still runs on 3 macro drivers over the full 1968-2026 sample
# for consistency; the now-real (not VIX-proxy) 4-driver liquidity panel is
# used for the Step 9 robustness check and is naturally bounded at 2019-12
# by its source rather than by VIX's 1990 start.
MACRO_DRIVERS = ["oil", "pot_output", "term_spread"]
MACRO_DRIVERS_EXT = ["oil", "pot_output", "term_spread", "liquidity"]

SAMPLE_START = "1968-01-01"
SAMPLE_END = None  # None -> latest available

# ---------------------------------------------------------------------------
# Step 2: Johansen cointegration
# ---------------------------------------------------------------------------
JOHANSEN_MAXLAGS = 8
JOHANSEN_DET_ORDER = 1  # linear trend in the cointegrating relation

# ---------------------------------------------------------------------------
# Step 5: Out-of-sample evaluation
# ---------------------------------------------------------------------------
OOS_START_DATES = ["1980-01-01", "1990-01-01", "2000-01-01"]

# ---------------------------------------------------------------------------
# Step 6: Portfolio construction
# ---------------------------------------------------------------------------
RISK_AVERSION = 5.0
QUARTERS_PER_YEAR = 4
IS_OOS_SPLIT_DATE = "1996-01-01"  # midpoint-ish split for Table 5 OOS Sharpe

# ---------------------------------------------------------------------------
# Step 7: Extended backtest
# ---------------------------------------------------------------------------
EXTENDED_BACKTEST_START = "2020-01-01"
EXTENDED_BACKTEST_END = "2026-03-31"  # cap at 2026Q1
EXTENDED_TRAIN_CUTOFF = "2019-10-01"  # last training quarter (2019Q4 = period ending 2019-12-31)

# ---------------------------------------------------------------------------
# Step 8: Risk management bootstrap
# ---------------------------------------------------------------------------
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 42
VAR_QUANTILE = 0.10

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"


@dataclass
class Config:
    """Bundles all parameters for convenient passing / overriding in tests."""

    factors: list = field(default_factory=lambda: list(FACTORS))
    factor_labels: dict = field(default_factory=lambda: dict(FACTOR_LABELS))
    macro_drivers: list = field(default_factory=lambda: list(MACRO_DRIVERS))
    macro_drivers_ext: list = field(default_factory=lambda: list(MACRO_DRIVERS_EXT))
    sample_start: str = SAMPLE_START
    sample_end: str | None = SAMPLE_END
    johansen_maxlags: int = JOHANSEN_MAXLAGS
    johansen_det_order: int = JOHANSEN_DET_ORDER
    oos_start_dates: list = field(default_factory=lambda: list(OOS_START_DATES))
    risk_aversion: float = RISK_AVERSION
    quarters_per_year: int = QUARTERS_PER_YEAR
    is_oos_split_date: str = IS_OOS_SPLIT_DATE
    extended_backtest_start: str = EXTENDED_BACKTEST_START
    extended_backtest_end: str | None = EXTENDED_BACKTEST_END
    extended_train_cutoff: str = EXTENDED_TRAIN_CUTOFF
    bootstrap_n: int = BOOTSTRAP_N
    bootstrap_seed: int = BOOTSTRAP_SEED
    var_quantile: float = VAR_QUANTILE
    results_dir: str = RESULTS_DIR
