"""
Step 0: Data acquisition.

Downloads the Fama-French 5 factors (Kenneth French's data library) and the
four macro drivers (FRED, with a Pastor-Stambaugh traded liquidity factor ->
VIX fallback waterfall for liquidity), converts everything to quarterly
log/simple returns, and aligns all series onto a common end-of-quarter index.

Every live download goes through `_cached`, which writes a local pickle on
success and falls back to the last cached copy if the live fetch fails (no
network, a source is down, etc.) -- so the pipeline can still run, on
slightly stale data, without a live connection.
"""

from __future__ import annotations

import io
import pickle
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import Config, FF5_URL, FRED_CSV_TEMPLATE, FRED_SERIES, PASTOR_STAMBAUGH_URL, SIZE_BM_6_URL

_HTTP_TIMEOUT = 30
_QUARTER_END = "QE"

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def _cached(key: str, fetch_fn):
    """Try a live fetch; on failure, fall back to the last cached copy, if any.

    A successful live fetch always overwrites the cache, so offline runs are
    only ever as stale as the last time this ran with a network connection.
    """
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
    """Download a single FRED series as observation_date -> value."""

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


def _parse_french_monthly_block(raw_csv_text: str, value_cols: list[str], skiprows: int) -> pd.DataFrame:
    """Parse a Ken French library CSV, keeping only the leading monthly block.

    French's files stack a monthly block followed by other blocks (annual
    data, equal-weighted returns, ...) in the same CSV, sometimes with a
    different column count that trips up the C parser. The blocks are always
    separated by a blank line, so we slice out only the first block (up to
    the first blank line after the header) before handing it to pandas.
    """
    lines = raw_csv_text.splitlines()
    body = lines[skiprows:]
    end = next((i for i, line in enumerate(body) if line.strip() == ""), len(body))
    block_text = "\n".join(body[:end])

    df = pd.read_csv(io.StringIO(block_text))
    df.columns = ["date"] + list(df.columns[1:])
    df["date_num"] = pd.to_numeric(df["date"], errors="coerce")
    df = df[df["date_num"] > 190001].copy()
    df["date_num"] = df["date_num"].astype(int)

    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([-99.99, -999, -999.0], np.nan)

    df["date"] = pd.to_datetime(df["date_num"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.set_index("date")[value_cols]


def download_ff5_monthly(cfg: Config) -> pd.DataFrame:
    """Download monthly FF5 factors + RF, in decimal (not percent) units."""

    def _fetch() -> pd.DataFrame:
        resp = requests.get(FF5_URL, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(csv_name).decode("utf-8", errors="replace")
        cols = cfg.factors + ["RF"]
        df = _parse_french_monthly_block(raw, cols, skiprows=4)
        return df / 100.0

    df = _cached("ff5_monthly", _fetch)
    return df[df.index >= cfg.sample_start]


def download_size_bm_6_monthly() -> pd.DataFrame:
    """Download the 6 value-weight size/B-M portfolios (for Step 8's 'High' leg)."""

    def _fetch() -> pd.DataFrame:
        resp = requests.get(SIZE_BM_6_URL, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw = zf.read(csv_name).decode("utf-8", errors="replace")
        cols = ["SMALL LoBM", "ME1 BM2", "SMALL HiBM", "BIG LoBM", "ME2 BM2", "BIG HiBM"]
        df = _parse_french_monthly_block(raw, cols, skiprows=15)
        return df / 100.0

    return _cached("size_bm_6_monthly", _fetch)


def ff5_monthly_to_quarterly_log_returns(monthly: pd.DataFrame) -> pd.DataFrame:
    """Compound monthly simple returns to quarterly, then convert to log returns."""
    quarterly_simple = monthly.resample(_QUARTER_END).apply(lambda x: (1.0 + x).prod() - 1.0)
    quarterly_log = np.log(1.0 + quarterly_simple)
    return quarterly_log


def build_oil_driver() -> pd.Series:
    """Macro driver 1: WTI crude oil, quarterly sum of monthly log returns."""
    price = _download_fred_series(FRED_SERIES["oil"])
    log_ret = np.log(price / price.shift(1)).dropna()
    return log_ret.resample(_QUARTER_END).sum().rename("oil")


def build_potential_output_driver() -> pd.Series:
    """Macro driver 2: real potential GDP, log q/q growth (already quarterly)."""
    level = _download_fred_series(FRED_SERIES["pot_output"])
    level = level.resample(_QUARTER_END).last()
    growth = np.log(level / level.shift(1)).dropna()
    return growth.rename("pot_output")


def build_term_spread_driver() -> pd.Series:
    """Macro driver 3: GS10 - TB3MS, quarterly end-of-quarter level (a flow variable)."""
    gs10 = _download_fred_series(FRED_SERIES["gs10"]).resample(_QUARTER_END).last()
    tb3ms = _download_fred_series(FRED_SERIES["tb3ms"]).resample(_QUARTER_END).last()
    spread = (gs10 - tb3ms).dropna()
    return spread.rename("term_spread")


def _parse_pastor_stambaugh_text(raw_text: str) -> pd.Series:
    """Parse the Fama-Miller mirror of the PS liquidity file.

    Whitespace-delimited, '%'-prefixed header/comment lines, columns are
    [month(YYYYMM), agg liq level, non-traded liq innovation, traded liq
    (LIQ_V)]. LIQ_V is -99 before the traded factor can be computed (pre-
    1968); those rows are dropped.
    """
    rows = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            continue
        rows.append((parts[0], parts[3]))

    df = pd.DataFrame(rows, columns=["month", "liq_v"])
    df["liq_v"] = pd.to_numeric(df["liq_v"], errors="coerce").replace(-99.0, np.nan)
    df["date"] = pd.to_datetime(df["month"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.set_index("date")["liq_v"].dropna()


def _download_pastor_stambaugh_liquidity() -> pd.Series:
    def _fetch() -> pd.Series:
        resp = requests.get(PASTOR_STAMBAUGH_URL, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        return _parse_pastor_stambaugh_text(resp.text)

    return _cached("pastor_stambaugh_liquidity", _fetch)


def build_liquidity_driver(cfg: Config) -> tuple[pd.Series, str]:
    """Macro driver 4: Pastor-Stambaugh traded liquidity factor (LIQ_V), else -VIX proxy.

    Returns (series, source_label). Falls back to Option B (end-of-quarter
    VIX, inverted) only if the Fama-Miller mirror is unreachable and no
    cached copy exists.
    """
    try:
        ps = _download_pastor_stambaugh_liquidity()
    except Exception:
        ps = None

    if ps is not None and len(ps) > 0:
        return ps.resample(_QUARTER_END).sum().rename("liquidity"), "Pastor-Stambaugh"

    vix = _download_fred_series(FRED_SERIES["vix"])
    quarterly = vix.resample(_QUARTER_END).last()
    liquidity_proxy = (-1.0 * quarterly).rename("liquidity")
    return liquidity_proxy, "VIX (inverted, quarterly end-of-quarter)"


def build_macro_drivers(cfg: Config) -> tuple[pd.DataFrame, str]:
    """Build the 4-driver panel (oil, potential output, term spread, liquidity)."""
    oil = build_oil_driver()
    pot = build_potential_output_driver()
    term = build_term_spread_driver()
    liq, liq_source = build_liquidity_driver(cfg)

    macro = pd.concat([oil, pot, term, liq], axis=1)
    macro.columns = ["oil", "pot_output", "term_spread", "liquidity"]
    return macro, liq_source


def _align(factor_log_returns: pd.DataFrame, macro: pd.DataFrame, cols: list[str], cfg: Config) -> pd.DataFrame:
    aligned = pd.concat([factor_log_returns, macro[cols]], axis=1)
    aligned = aligned[aligned.index >= cfg.sample_start]
    if cfg.sample_end is not None:
        aligned = aligned[aligned.index <= cfg.sample_end]
    return aligned.dropna(how="any")


def load_data(cfg: Config) -> dict:
    """End-to-end Step 0 load.

    Returns two aligned panels:
      - core: 3 macro drivers (oil, potential output, term spread), full
        1968-2026 sample -- this is the primary panel used in Steps 1-8.
      - extended: 4 macro drivers (+ VIX-proxy liquidity), limited to
        1990-2026 by VIX availability -- used only for the Step 9
        robustness check, since folding it into the primary panel would
        cut the usable sample by more than half.
    """
    ff5_monthly = download_ff5_monthly(cfg)
    factor_log_returns = ff5_monthly_to_quarterly_log_returns(ff5_monthly[cfg.factors])

    macro_raw, liquidity_source = build_macro_drivers(cfg)

    core = _align(factor_log_returns, macro_raw, cfg.macro_drivers, cfg)
    extended = _align(factor_log_returns, macro_raw, cfg.macro_drivers_ext, cfg)

    size_bm_monthly = download_size_bm_6_monthly()

    return {
        "factor_log_returns": core[cfg.factors],
        "macro_series": core[cfg.macro_drivers],
        "factor_log_returns_ext": extended[cfg.factors],
        "macro_series_ext": extended[cfg.macro_drivers_ext],
        "liquidity_source": liquidity_source,
        "size_bm_monthly": size_bm_monthly,
        "ff5_monthly": ff5_monthly,
    }
