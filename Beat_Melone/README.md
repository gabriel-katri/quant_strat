# Rolling LASSO

Two stages: (1) assemble a quarterly dataset (1975Q1 - latest available) of
Fama-French 5 factor returns and Melone (2022)'s 4 macro drivers, and (2)
fit a rolling-window Lasso, one model per factor, to predict each factor's
quarterly return from the lagged macro drivers.

Sample start (1975Q1) and the initial/in-sample window (100 quarters,
1975Q1-1999Q4) were set via `diagnose_sample_start.py`: Johansen trace tests
at three candidate starts (1968Q1, 1975Q1, 1990Q1), sample end fixed at
2019Q4 so only the start varies. 1968Q1 (the earliest quarter with full
coverage on all 4 drivers + all 5 FF factors) gives only 4/5 factors
rejecting r=0 at 95% — CMA fails outright — consistent with thin/
survivorship-biased Compustat coverage pre-1975 (CMA, HML) and
administratively-set, non-market WTI oil prices pre-mid-1970s (all 5
factors, via the shared oil driver). 1975Q1 gives 5/5 factors rejecting
r=0, with larger trace-stat margins than even 1990Q1 for HML, RMW, and CMA.

## Columns

- **Factor returns** (Kenneth French's data library): `MKT_RF`, `SMB`,
  `HML`, `RMW`, `CMA`, `RF` — decimal units, monthly simple returns
  compounded to quarterly.
- **Macro drivers** — Melone (Favero, Melone & Tamoni 2022)'s own set,
  from FRED, each lagged 1 quarter vs. the factor returns so the dataset
  is regression-ready (row `t` pairs `r_t` with `x_{t-1}`; equivalently,
  "predict `r_{t+1}` with `x_t`" reads off row `t+1`):
  - `OIL_WTI_LOGRET` — WTI crude (`WTISPLC`), quarterly sum of monthly log
    returns
  - `POT_GDP_GROWTH` — real potential GDP (`GDPPOT`), log q/q growth.
    GDPPOT is natively quarterly on FRED, so no interpolation is needed at
    this frequency.
  - `TERM_SPREAD_10Y3M` — `GS10 - TB3MS` (10Y minus 3M Treasury spread),
    end-of-quarter level
  - `LIQUIDITY_PS` — **real** Pastor-Stambaugh traded liquidity factor
    (LIQ_V), sourced from Chicago Booth's Fama-Miller Center mirror (the
    paper-era Wharton URL is 404). LIQ_V only covers 1968-2019, so the
    series is **spliced**: real LIQ_V through 2019Q4, then a `-TEDRATE`
    proxy from 2020Q1 on (TEDRATE itself flat-carried past its 2022-01-21
    discontinuation). This is a genuine measurement-definition break, not
    just a data-vendor swap — LIQ_V is a traded portfolio *return*
    (~-0.05 to +0.05/quarter), `-TEDRATE` is a negated rate *level*
    (~-3 to 0 pp) — documented in `build_liquidity_driver()` in `data.py`
    and flagged via `[WARN]` at run time. Treat any 2020-2026 result that
    leans on the liquidity driver with that caveat in mind.

## Project structure

```
Rolling_LASSO/
├── config.py           # sample dates, FRED series IDs, model params
├── data.py              # downloads, cache fallback, dataset assembly
├── main.py                # Stage 1: prints head/describe/NaN/corr, saves CSV
├── lasso.py                 # Stage 2: rolling-window LassoCV per factor
├── plots.py                   # Stage 2 plots (coefficients, predicted vs actual)
├── train_lasso.py                # Stage 2 entry point
├── requirements.txt
└── results/                         # CSVs + PNGs (generated on run)
```

## Setup & run

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py          # Stage 1: build results/rolling_lasso_dataset.csv
python train_lasso.py   # Stage 2: rolling Lasso, reads the CSV from Stage 1
```

## Stage 2: rolling-window Lasso

For each factor (`MKT_RF`, `SMB`, `HML`, `RMW`, `CMA`), at every quarter
`t`: trains on the trailing 100 quarters (25 years, 1975Q1-1999Q4 for the
first prediction) `[t-100, t-1]`, standardizes the 4 macro predictors
using **only the training window's** mean/std (same transform applied to
the test point — no look-ahead), picks lambda via `LassoCV` with a
5-split `TimeSeriesSplit` inside the window, and predicts the return at
`t`. The macro lag is already baked into the dataset from Stage 1, so no
further shifting is needed here. OOS predictions run 2000Q1-2026Q2 (106
quarters) regardless of the sample start, since the window length grew by
exactly as much as the sample start moved earlier.

Outputs (`results/`):
- `lasso_predictions.csv` — actual vs. predicted return per factor/date
- `lasso_coefficients.csv` — selected Lasso coefficients per factor/date
- `lasso_oos_r2.csv` — out-of-sample R² per factor (vs. the OOS-period
  sample mean of actual — the standard R² convention, not
  Campbell-Thompson)
- `lasso_coefficients_over_time.png` — one panel per factor, colored by
  macro driver (same driver = same color across all 5 panels)
- `lasso_predicted_vs_actual.png` — actual vs. predicted return time
  series, one panel per factor

OOS R² is still negative for 4 of 5 factors (-0.1% to -5.9%; HML is
closest to flat), but the longer, cleaner 1975-1999 training window and
the real liquidity driver noticeably improved hit rates (45-71%, MKT_RF
now the strongest at 71%) and the resulting portfolio's risk-adjusted
performance (see `run_melone.py`'s comparison table) vs. the earlier
1990Q1/40-quarter setup.

## Notes

- Downloads go through a local pickle cache (`cache/`, gitignored) that's
  written on every successful live fetch and used as a fallback if a
  source is unreachable, so the script can still run offline once it has
  succeeded at least once with a network connection.
- **On the liquidity driver:** see the `LIQUIDITY_PS` note above — it's
  real Pastor-Stambaugh data through 2019Q4, then a `-TEDRATE` proxy
  after, with a documented scale discontinuity exactly at that boundary.
- **On the Kalman filter (`melone_kalman.py`):** two variants are
  estimated and reported side by side — unconstrained random walk (RW)
  and mean-reverting toward the static cointegrating vector (MR, with an
  MLE-estimated persistence `phi` per factor). Neither is selected as a
  "winner"; see `run_melone.py`'s comparison table for both, full period
  and by sub-period (2000-2019, 2020-2026).
