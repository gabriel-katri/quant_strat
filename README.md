Projects
Beat_Melone

Adaptive Factor Timing with Time-Varying Cointegration

A critique and extension of Favero, Melone & Tamoni (2022). The original paper uses static cointegration-based signals to time Fama-French factors. This project shows that the static approach breaks down post-2020 due to macro regime shifts, and proposes a Kalman filter-based adaptive alternative that restores out-of-sample performance.

Key results:

Static cointegration model: OOS Sharpe ≈ 0.04 (collapses post-2020)
Adaptive Kalman filter model: OOS Sharpe ≈ 0.69
2×2 attribution isolates the method (Kalman vs. static) as the performance driver
mtft

Macro Trend Factor Timing

Macro-driven factor timing using economic indicators (CPI, unemployment, VIX, industrial production, term spread, real M2) mapped to Fama-French factors with ex ante economic justification.

tsmom

Time-Series Momentum

Implementation of time-series momentum strategies following Moskowitz, Ooi & Pedersen (2012).

Stack
Python (NumPy, pandas, SciPy, matplotlib, seaborn)
Data: Yahoo Finance, FRED, Kenneth French Data Library
