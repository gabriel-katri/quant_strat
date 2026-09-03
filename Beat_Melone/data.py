"""
Download and assemble the quarterly factor + macro-driver dataset.

Fama-French 5 factors (Kenneth French's data library) are downloaded
monthly and compounded to quarterly; Melone (2022)'s own 4 macro drivers
(FRED, plus the real Pastor-Stambaugh traded liquidity factor spliced with
a -TED spread proxy for its post-2019 gap) are each aggregated to end-of-
quarter (oil: summed monthly log returns; GDPPOT: native quarterly
frequency; term spread and liquidity: end-of-quarter level/return). The
macro columns are then lagged one quarter relative to the factor returns so
the assembled DataFrame is regression-ready: row t already pairs r_t with
x_{t-1} (i.e. "predict r_{t+1} with x_t" reads directly off row t+1).

Every live download goes through `_cached`, which writes a local pickle on
success and falls back to the last cached copy if the live fetch fails.
"""

from __future__ import annotations

import io
import pickle
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import Config, FF5_URL, FRED_CSV_TEMPLATE, PASTOR_STAMBAUGH_URL

_HTTP_TIMEOUT = 30
_MONTH_END = "ME"
_QUARTER_END = "QE"

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _cached(key: str, fetch_fn):
    """Try a live fetch; on failure, fall back to the last cached copy, if any."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{key}.pkl"
    try:
        result = fetch_fn()
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return result
    except Exception as e:
        if not path.exists():
            raise RuntimeError(f"Live download failed for '{key}' and no cached copy exists: {e!r}") from e
        age_days = (datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)).days
        print(f"[WARN] Live download failed for '{key}' ({e!r}); using cached copy from {age_days}d ago.")
        with open(path, "rb") as f:
            return pickle.load(f)


def _download_fred_series(series_id: str) -> pd.Series:
    def _fetch() -> pd.Series:
        url = FRED_CSV_TEMPLATE.format(series_id=series_id)
        df = pd.read_csv(url)
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        s = df.set_index("date")["value"].dropna()
        s.name = series_id
        return s

    return _cached(f"fred_{series_id}", _fetch)


def _parse_ff5_monthly_block(raw_csv_text: str, value_cols: list[str]) -> pd.DataFrame:
    """Parse Ken French's FF5 CSV, keeping only the leading monthly block.

    The file has 4 header lines before the column row, then the monthly
    block, then a blank line before the annual block starts.
    """
    lines = raw_csv_text.splitlines()
    body = lines[4:]
    end = next((i for i, line in enumerate(body) if line.strip() == ""), len(body))
    block_text = "\n".join(body[:end])

    df = pd.read_csv(io.StringIO(block_text))
    df.columns = ["date"] + list(df.columns[1:])
    df["date_num"] = pd.to_numeric(df["date"], errors="coerce")
    df = df[df["date_num"] > 190001].copy()
    df["date_num"] = df["date_num"].astype(int)

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = pd.to_datetime(df["date_num"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.set_index("date")[value_cols]


def download_ff5_monthly() -> pd.DataFrame:
    """Monthly FF5 factors + RF, decimal units, renamed to MKT_RF/SMB/HML/RMW/CMA/RF."""

    def _fetch() -> pd.DataFrame:
        resp = requests.get(FF5_URL, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(csv_name).decode("utf-8", errors="replace")
        df = _parse_ff5_monthly_block(raw, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
        return df / 100.0

    df = _cached("ff5_monthly", _fetch)
    return df.rename(columns={"Mkt-RF": "MKT_RF"})


def ff5_monthly_to_quarterly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compound monthly simple returns to quarterly simple returns."""
    return monthly.resample(_QUARTER_END).apply(lambda x: (1.0 + x).prod() - 1.0)


def build_oil_driver(cfg: Config) -> pd.Series:
    """Macro driver 1: WTI crude, quarterly sum of monthly log returns."""
    price = _download_fred_series(cfg.fred_series["wti"]).resample(_MONTH_END).last()
    log_ret = np.log(price / price.shift(1)).dropna()
    quarterly = log_ret.resample(_QUARTER_END).sum()
    return quarterly.rename("OIL_WTI_LOGRET")


def build_potential_output_driver(cfg: Config) -> pd.Series:
    """Macro driver 2: real potential GDP (GDPPOT), log q/q growth.

    GDPPOT is natively quarterly on FRED, so no interpolation is needed at
    this frequency (unlike the monthly build, which had to interpolate).
    """
    level = _download_fred_series(cfg.fred_series["gdppot"]).resample(_QUARTER_END).last()
    growth = np.log(level / level.shift(1)).dropna()
    return growth.rename("POT_GDP_GROWTH")


def build_term_spread_driver(cfg: Config) -> pd.Series:
    """Macro driver 3: GS10 - TB3MS, end-of-quarter level (percentage points)."""
    gs10 = _download_fred_series(cfg.fred_series["gs10"]).resample(_QUARTER_END).last()
    tb3ms = _download_fred_series(cfg.fred_series["tb3ms"]).resample(_QUARTER_END).last()
    spread = (gs10 - tb3ms).dropna()
    return spread.rename("TERM_SPREAD_10Y3M")


def _fetch_real_liquidity_quarterly() -> pd.Series:
    """LIQ_V (Pastor & Stambaugh 2003's traded liquidity factor: the 10-1
    portfolio return from a sort on historical liquidity betas), quarterly
    sum of the monthly series. Source: Chicago Booth's Fama-Miller Center
    mirror (the paper-era Wharton URL is 404, confirmed). -99 is the file's
    own missing-value sentinel for months before the traded factor's
    construction (pre-1968); the file itself only runs through Dec 2019.
    """

    def _fetch() -> pd.Series:
        resp = requests.get(PASTOR_STAMBAUGH_URL, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        lines = [ln for ln in resp.text.splitlines() if ln.strip() and not ln.startswith("%")]
        rows = [re.split(r"\s+", ln.strip()) for ln in lines]
        raw = pd.DataFrame(rows, columns=["month", "agg_liq", "innov_liq", "liq_v"])
        raw["liq_v"] = pd.to_numeric(raw["liq_v"], errors="coerce")
        raw["month"] = pd.to_datetime(raw["month"], format="%Y%m") + pd.offsets.MonthEnd(0)
        s = raw.set_index("month")["liq_v"]
        return s[s != -99].dropna()

    monthly = _cached("liq_v_monthly", _fetch)
    return monthly.resample(_QUARTER_END).sum()


def build_liquidity_driver(cfg: Config) -> tuple[pd.Series, str]:
    """Macro driver 4: real Pastor-Stambaugh traded liquidity (LIQ_V) through
    `cfg.liquidity_real_cutoff` (2019Q4), then -TED spread from 2020Q1 on.

    LIQ_V itself only covers 1968-2019 (see _fetch_real_liquidity_quarterly);
    TEDRATE was also discontinued by the Fed/ICE on 2022-01-21, so its tail
    (past that date) is flat-carried at its last observed quarterly value.

    This splice is a real measurement-definition break, not just a swap of
    data vendor: LIQ_V is a traded portfolio RETURN (roughly -0.05 to +0.05
    per quarter), while -TEDRATE is a negated interest-rate LEVEL (roughly
    -3 to 0 percentage points) -- the two are not on comparable scales, and
    the spliced series has a level discontinuity exactly at the 2019Q4/2020Q1
    boundary. This is a deliberate, documented limitation (not smoothed over
    or flat-carried across the boundary): the 2020-2026 tail of every
    backtest here sits on a materially different liquidity measurement than
    the 1975-2019 in-sample/early-OOS period, in exactly the window where
    liquidity conditions moved sharply through COVID, QE, and the 2022
    tightening cycle. Flag this when interpreting any 2020-2026 result that
    leans on the liquidity driver.
    """
    cutoff = pd.Timestamp(cfg.liquidity_real_cutoff)
    liq_real = _fetch_real_liquidity_quarterly()

    ted = _download_fred_series(cfg.fred_series["tedrate"]).resample(_QUARTER_END).last()
    last_ted_obs = ted.index.max()
    full_index = pd.date_range(ted.index.min(), pd.Timestamp.today().normalize(), freq=_QUARTER_END)
    ted_filled = ted.reindex(full_index).ffill()
    n_ted_filled = int((ted_filled.index > last_ted_obs).sum())
    neg_ted = -1.0 * ted_filled

    spliced = pd.concat([liq_real[liq_real.index <= cutoff], neg_ted[neg_ted.index > cutoff]])
    spliced = spliced[~spliced.index.duplicated(keep="first")].sort_index().rename("LIQUIDITY_PS")

    print(f"[WARN] Liquidity driver spliced at {cutoff.date()}: real Pastor-Stambaugh LIQ_V "
          f"({liq_real.index.min().date()} to {liq_real.index.max().date()}), then -TEDRATE proxy "
          f"from {(cutoff + pd.offsets.QuarterBegin(startingMonth=1)).date()} on (flat-carried past "
          f"{last_ted_obs.date()}, {n_ted_filled} quarters). Units differ across the splice (traded "
          f"return vs. negated rate level) -- documented limitation, see build_liquidity_driver().")

    source = f"Real Pastor-Stambaugh LIQ_V through {cutoff.date()}, -TEDRATE proxy from 2020Q1 on"
    return spliced, source


def build_macro_signals(cfg: Config) -> tuple[pd.DataFrame, str]:
    """Build the 4 Melone (2022) macro drivers, each on an end-of-quarter index."""
    oil = build_oil_driver(cfg)
    pot = build_potential_output_driver(cfg)
    term = build_term_spread_driver(cfg)
    liq, liq_source = build_liquidity_driver(cfg)

    macro = pd.concat([oil, pot, term, liq], axis=1)
    return macro, liq_source


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Assemble the final quarterly dataset: factor returns + one-quarter-lagged macro drivers."""
    ff5_monthly = download_ff5_monthly()
    ff5 = ff5_monthly_to_quarterly(ff5_monthly[cfg.factor_columns])
    macro, _liq_source = build_macro_signals(cfg)

    macro_lagged = macro.shift(1)  # row t now holds x_{t-1}, ready to predict r_t off row t

    df = pd.concat([ff5, macro_lagged[cfg.macro_columns]], axis=1)
    df = df[df.index >= cfg.sample_start]
    if cfg.sample_end is not None:
        df = df[df.index <= cfg.sample_end]
    df = df.dropna(how="any")
    return df
