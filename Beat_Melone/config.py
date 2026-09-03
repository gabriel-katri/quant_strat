"""
Central configuration for the rolling-Lasso dataset build.

Frequency: quarterly. FF5 factor returns are compounded from monthly to
quarterly; GDPPOT is natively quarterly (no interpolation needed at this
frequency); oil is summed from monthly log returns; term spread and the
liquidity driver take the end-of-quarter level.

Sample start: 1975Q1 (not 1990Q1, and not the earliest-technically-available
1968Q1). Diagnosed via diagnose_sample_start.py: Johansen trace tests at
three candidate starts (1968Q1, 1975Q1, 1990Q1), sample end fixed at 2019Q4
(the real liquidity driver's cutoff) so only the start varies. 1968Q1 gives
only 4/5 factors rejecting r=0 at 95% -- CMA outright fails (trace stat
below its 95% CV), consistent with thin/survivorship-biased Compustat
coverage pre-1975 (CMA, HML) and administratively-set, non-market WTI oil
prices pre-mid-1970s (all 5 factors, via the shared oil driver). 1975Q1
gives 5/5 factors rejecting r=0, with larger trace-stat margins than even
the 1990Q1 default for HML, RMW, and CMA -- adopted as the new default.
"""

from dataclasses import dataclass, field

SAMPLE_START = "1975-01-01"
SAMPLE_END = None  # None -> latest available

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
FRED_CSV_TEMPLATE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

FACTOR_COLUMNS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF"]

# Pastor-Stambaugh traded liquidity factor (LIQ_V). The paper-era Wharton
# host is 404; this Chicago Booth Fama-Miller Center mirror is confirmed
# live and covers 1962-2019 (column 4 of the file). See build_liquidity_driver()
# in data.py for how this is spliced with a -TEDRATE proxy for 2020Q1 on,
# since LIQ_V itself stops at 2019Q4.
PASTOR_STAMBAUGH_URL = "https://research.chicagobooth.edu/-/media/research/famamiller/data/liq_data_1962_2019.txt"
LIQUIDITY_REAL_CUTOFF = "2019-12-31"  # last quarter with real LIQ_V; -TEDRATE proxy used after this

# FRED series IDs for Melone (2022)'s own macro drivers.
FRED_SERIES = {
    "wti": "WTISPLC",       # WTI crude spot price, monthly -> summed to quarterly log return
    "gdppot": "GDPPOT",     # Real potential GDP, quarterly (native frequency, no interpolation)
    "gs10": "GS10",         # 10Y Treasury, monthly -> end-of-quarter level
    "tb3ms": "TB3MS",       # 3M T-bill, monthly -> end-of-quarter level
    "tedrate": "TEDRATE",   # TED spread, liquidity proxy fallback for 2020Q1 on (discontinued 2022-01-21)
}

MACRO_COLUMNS = ["OIL_WTI_LOGRET", "POT_GDP_GROWTH", "TERM_SPREAD_10Y3M", "LIQUIDITY_PS"]

# ---------------------------------------------------------------------------
# Rolling-Lasso model (predicts each factor's quarterly return from the
# already-lagged macro drivers)
# ---------------------------------------------------------------------------
LASSO_TARGET_FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]  # RF is a rate, not a return to predict
LASSO_PREDICTORS = list(MACRO_COLUMNS)
ROLLING_WINDOW_QUARTERS = 100  # 25 years: 1975Q1-1999Q4 in-sample/initialization window
CV_SPLITS = 5
LASSO_MAX_ITER = 20_000
LASSO_RANDOM_STATE = 42

RESULTS_DIR = "results"
OUTPUT_CSV = "rolling_lasso_dataset.csv"

# ---------------------------------------------------------------------------
# Factor-timing portfolios (mean-variance weights from the rolling-Lasso
# predicted returns, covariance estimated on the same rolling window)
# ---------------------------------------------------------------------------
BETA_NEUTRAL_FACTORS = ["SMB", "HML", "RMW", "CMA"]  # MKT_RF excluded
UNCONSTRAINED_FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA"]
TARGET_VOL_ANNUAL = 0.10  # 10% annualized vol target
CER_GAMMA = 5.0  # risk aversion for the Certainty Equivalent Return (Melone (2022) uses gamma=5, not 3)
PERIODS_PER_YEAR = 4  # quarterly

# ---------------------------------------------------------------------------
# Melone static cointegration (Johansen test) + Kalman time-varying-beta
# cointegration
# ---------------------------------------------------------------------------
JOHANSEN_MAXLAGS = 4  # VAR lag search cap (AIC-selected), short quarterly sample
JOHANSEN_DET_ORDER = 1  # linear trend in the cointegrating relation
OOS_SUBPERIODS = {"2000-2019": ("2000-01-01", "2019-12-31"), "2020-2026": ("2020-01-01", None)}

# ---------------------------------------------------------------------------
# Factor-specific macro drivers: the original 6 ad hoc FRED signals (used by
# the rolling-Lasso model before the switch to Melone's own 4 drivers), each
# factor paired with its own 3-driver subset instead of one shared 4-driver
# panel. FRED series IDs reused from the original rolling-Lasso build.
# ---------------------------------------------------------------------------
FRED_SERIES_FACTOR_SPECIFIC = {
    "cpi": "CPIAUCSL",
    "unrate": "UNRATE",
    "vix": "VIXCLS",
    "indpro": "INDPRO",  # PMI proxy -- see main README note on ISM PMI's 2016 FRED discontinuation
    "term_spread": "T10Y2Y",
    "m2": "M2SL",
}

FACTOR_SPECIFIC_MACRO_COLUMNS = [
    "CPI_YoY", "UNRATE", "VIX", "PMI_proxy_INDPRO_YoY", "TERM_SPREAD_10Y2Y", "REAL_M2_YoY",
]

FACTOR_SPECIFIC_DRIVERS = {
    "MKT_RF": ["VIX", "PMI_proxy_INDPRO_YoY", "CPI_YoY"],
    "SMB": ["VIX", "TERM_SPREAD_10Y2Y", "UNRATE"],
    "HML": ["CPI_YoY", "TERM_SPREAD_10Y2Y", "PMI_proxy_INDPRO_YoY"],
    "RMW": ["VIX", "REAL_M2_YoY", "UNRATE"],
    "CMA": ["CPI_YoY", "TERM_SPREAD_10Y2Y", "REAL_M2_YoY"],
}

FACTOR_SPECIFIC_OUTPUT_CSV = "factor_specific_dataset.csv"


@dataclass
class Config:
    sample_start: str = SAMPLE_START
    sample_end: str | None = SAMPLE_END
    factor_columns: list = field(default_factory=lambda: list(FACTOR_COLUMNS))
    macro_columns: list = field(default_factory=lambda: list(MACRO_COLUMNS))
    fred_series: dict = field(default_factory=lambda: dict(FRED_SERIES))
    results_dir: str = RESULTS_DIR
    output_csv: str = OUTPUT_CSV
    lasso_target_factors: list = field(default_factory=lambda: list(LASSO_TARGET_FACTORS))
    lasso_predictors: list = field(default_factory=lambda: list(LASSO_PREDICTORS))
    rolling_window_periods: int = ROLLING_WINDOW_QUARTERS
    cv_splits: int = CV_SPLITS
    lasso_max_iter: int = LASSO_MAX_ITER
    lasso_random_state: int = LASSO_RANDOM_STATE
    beta_neutral_factors: list = field(default_factory=lambda: list(BETA_NEUTRAL_FACTORS))
    unconstrained_factors: list = field(default_factory=lambda: list(UNCONSTRAINED_FACTORS))
    target_vol_annual: float = TARGET_VOL_ANNUAL
    cer_gamma: float = CER_GAMMA
    periods_per_year: int = PERIODS_PER_YEAR
    johansen_maxlags: int = JOHANSEN_MAXLAGS
    johansen_det_order: int = JOHANSEN_DET_ORDER
    oos_subperiods: dict = field(default_factory=lambda: dict(OOS_SUBPERIODS))
    fred_series_factor_specific: dict = field(default_factory=lambda: dict(FRED_SERIES_FACTOR_SPECIFIC))
    factor_specific_macro_columns: list = field(default_factory=lambda: list(FACTOR_SPECIFIC_MACRO_COLUMNS))
    factor_specific_drivers: dict = field(default_factory=lambda: {k: list(v) for k, v in FACTOR_SPECIFIC_DRIVERS.items()})
    factor_specific_output_csv: str = FACTOR_SPECIFIC_OUTPUT_CSV
    liquidity_real_cutoff: str = LIQUIDITY_REAL_CUTOFF
