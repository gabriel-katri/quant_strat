# TSMOM: Time Series Momentum

A from-scratch implementation of the Time Series Momentum (TSMOM) strategy
from Moskowitz, Ooi & Pedersen (2012), *"Time Series Momentum,"* Journal of
Financial Economics. Data proxied via liquid ETFs on yfinance (no futures
data vendor required).

## What it does

1. Downloads daily prices for a 25-instrument universe (equities, bonds,
   commodities, currencies) and a risk-free proxy (`^IRX`), computes daily
   excess returns.
2. Estimates ex-ante annualized volatility per instrument via an EWMA of
   squared daily returns (center of mass = 60 days), lagged one day to
   avoid look-ahead bias (Eq. 1 of the paper).
3. Builds the TSMOM signal at each month-end: `sign(trailing 12m excess
   return)`, sized to a 40% annualized volatility target (Eq. 5), and
   compounds an equal-weighted diversified portfolio across all available
   instruments.
4. Runs the same construction with a signal fixed at `+1` (a "passive
   long," volatility-scaled but never short) as a benchmark.
5. Regresses TSMOM monthly returns on the Fama-French/Carhart four factors
   (MKT, SMB, HML, UMD) and separately on `MKT + MKT^2` to test for a
   convex, straddle-like payoff (Eq. 4).
6. Builds a cross-sectional momentum (XSMOM) strategy on the same universe
   and decomposes TSMOM's predictability into auto-covariance,
   cross-serial, and mean components (Eq. 6-7).
7. Saves six diagnostic plots and a full performance summary to `results/`.

## Project structure

```
tsmom/
├── config.py          # universe, parameters (lookback, vol target, etc.)
├── data.py             # yfinance download & cleaning, excess returns
├── volatility.py       # EWMA ex-ante volatility (Eq. 1)
├── signals.py           # TSMOM / XSMOM signal generation & position sizing
├── backtest.py           # portfolio construction & performance metrics
├── factors.py            # Fama-French/Carhart factor regressions (Eq. 4)
├── decomposition.py       # TSMOM vs XSMOM decomposition (Eq. 6-7)
├── plots.py                # all visualizations
├── main.py                  # orchestrates the full pipeline
├── requirements.txt
└── results/                  # generated plots, CSVs (created on run)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

This downloads data, runs the full backtest and analysis, prints a summary
of all performance metrics to stdout, and writes six PNG plots plus a few
CSV summaries to `results/`.

## Tweaking parameters

Everything lives in `config.py`:

- `UNIVERSE` — the instrument dictionary (ticker -> description). Add or
  remove tickers freely; the rest of the pipeline handles a ragged panel
  (instruments with different inception dates) automatically.
- `LOOKBACK_MONTHS` — signal lookback window (default 12, per the paper).
- `VOL_TARGET` — annualized volatility target per position (default 40%).
- `VOL_COM` — EWMA center of mass for the volatility estimator (default 60
  trading days).
- `TRANSACTION_COST_BPS` — set > 0 to charge a simple turnover-based cost;
  the cost model is applied in `backtest.apply_transaction_costs` and is a
  no-op at the default of 0.

## Notes & caveats

- **ETF proxies, not futures.** yfinance does not provide continuous
  futures history, so ETFs stand in for the paper's futures universe. This
  introduces basis effects, dividend/expense-ratio drag, and shorter
  history than the original 1985-2009 futures sample (most ETFs used here
  only go back to the mid-2000s at best).
- **Fama-French/momentum factors** are downloaded live from Ken French's
  data library via `pandas_datareader`; this requires an internet
  connection at run time and will print a warning (not fail the whole run)
  if unavailable.
- **The Eq. 6-7 decomposition** is implemented as an empirical
  auto-covariance / cross-serial / mean-component estimate on vol-
  normalized returns, in the spirit of the paper's decomposition; see the
  docstring in `decomposition.py` for the exact construction and its
  approximations.
- No transaction costs are charged by default, per the assignment spec,
  but the hook is in place in `backtest.py`.
