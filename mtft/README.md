# MTFT: Macro Trends and Factor Timing

A from-scratch implementation of the factor timing strategy from Favero,
Melone & Tamoni (2022), *"Macro Trends and Factor Timing."* Tests whether
the Fama-French 5 factors are cointegrated with a set of macro drivers,
uses the resulting error-correction term (ECT) to forecast next-quarter
factor returns, and backtests mean-variance factor-timing portfolios built
on those forecasts.

## What it does

1. Downloads FF5 monthly factors (Ken French's data library), compounds to
   quarterly log returns, and builds 3 macro drivers from FRED: WTI oil
   (log returns), real potential GDP (log growth), and the 10Y-3M term
   spread. A 4th driver (VIX, inverted, as a liquidity proxy — the paper's
   Pastor-Stambaugh liquidity series is no longer hosted at its original
   URL) is built separately since it only goes back to 1990 and would
   truncate the primary 1968-2026 sample by more than half.
2. Constructs factor "prices" (cumulative log returns) and macro driver
   levels (cumulative sums), then runs the Johansen cointegration test on
   each factor against the macro drivers (Table 1).
3. Estimates the long-run regression (factor price ~ trend + macro driver
   levels) and takes the residual as the ECT (Table A.1, Figure 1a).
4. Regresses next-quarter factor returns on the current ECT (Table 3 Panel
   A, Figure 1b) and evaluates true out-of-sample forecasts with an
   expanding window that re-estimates both regressions every quarter
   (Table 3 Panel B: Campbell-Thompson R², Clark-West test).
5. Backtests 5 mean-variance strategies (FI, MT, FT, AT, and a beta-neutral
   BN variant) that differ only in which factors get a dynamic ECT-based
   forecast vs. a static unconditional mean (Table 5).
6. Runs an extended 2020-2026 backtest, both with coefficients frozen at
   2019Q4 and with a fully expanding window.
7. Bootstraps predicted return distributions for the Value portfolio's long
   leg around the 2008-09 GFC and 2020-21 COVID crash, comparing a
   constant-expected-return model to the macro-ECM (Figure 5, 10% VaR).
8. Checks robustness to alternative macro drivers: gold instead of oil,
   the Baa-Aaa corporate spread instead of the term spread, and the
   VIX-liquidity 4-driver panel.

## Project structure

```
mtft/
├── config.py           # sample dates, factor/macro lists, strategy params
├── data.py              # FF5 + FRED downloads, quarterly alignment
├── construct.py           # factor prices & macro driver levels (Step 1)
├── cointegration.py         # Johansen test (Step 2)
├── ect.py                     # long-run regression & ECT (Step 3)
├── predictive.py                # in-sample predictive regression (Step 4)
├── oos.py                         # expanding-window OOS evaluation (Step 5)
├── portfolio.py                     # FI/MT/FT/AT/BN strategies (Step 6)
├── backtest_2020.py                   # extended 2020-2026 backtest (Step 7)
├── risk.py                              # crisis-period bootstrap VaR (Step 8)
├── robustness.py                          # alternative macro drivers (Step 9)
├── plots.py                                 # all visualizations
├── main.py                                    # orchestrates the full pipeline
├── requirements.txt
└── results/                                     # generated plots, CSVs (on run)
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

Takes about 15 seconds (all data is downloaded fresh each run) and prints
every table to stdout, saving 8 PNG plots and ~10 CSV tables to `results/`.

## Notes & caveats

- **Liquidity driver:** the Pastor-Stambaugh traded liquidity file is no
  longer hosted at the paper-era URL (confirmed 404). The VIX-inverted
  fallback only starts in 1990, so the primary pipeline (Steps 1-8) uses 3
  macro drivers over the full 1968-2026 sample; the VIX-augmented 4-driver
  spec is reported separately as a Step 9 robustness variant on its
  shorter subsample.
- **Gold robustness driver:** FRED's LBMA gold series was discontinued and
  Stooq's CSV export is now gated behind a JS proof-of-work challenge, so
  the gold-for-oil robustness check uses the GLD ETF (2004+) as a proxy.
- **OOS evaluation (Step 5) and the extended 2020-2026 expanding-window
  backtest (Step 7)** are "fully real-time": both the long-run regression
  and the predictive regression are re-estimated every quarter using only
  data available through t-1, avoiding look-ahead bias in the ECT itself.
- **Portfolio weights** use the standard quadratic-utility optimum
  `w = (1/gamma) * Sigma^-1 * mu` — the prompt's `w = Sigma^-1 * E[f]`
  omits the `1/gamma` scaling, but gamma is also used for the utility/fee
  calculation, so it's applied consistently at the weight stage too.
- Requires an internet connection at run time (Ken French's data library,
  FRED, and Yahoo Finance for the gold proxy).
