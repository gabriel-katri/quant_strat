"""
Ex-ante volatility estimation (Moskowitz, Ooi & Pedersen 2012, Eq. 1).

    sigma_t^2 = 261 * sum_{i=0}^inf (1-delta) delta^i * r_{t-1-i}^2

i.e. an exponentially weighted moving average of squared daily (excess)
returns, with the weighting parametrized by a center of mass (com) of 60
days, annualized by the trading-day count. Daily mean returns are assumed
~0 and are not subtracted, as is standard practice for this estimator.

Critically, sigma_t is computed using only returns available *before* t,
so it can be used to scale the position that is held over the return
realized at t without look-ahead bias.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config


def ewma_ex_ante_vol(excess_returns: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute annualized ex-ante volatility, lagged by one day (Eq. 1).

    Parameters
    ----------
    excess_returns : DataFrame of daily excess returns, index=date, columns=tickers
    cfg : Config

    Returns
    -------
    DataFrame of the same shape, where sigma.loc[t, ticker] is the
    volatility estimate to be used *for* time t (i.e. it is built entirely
    from returns strictly before t).
    """
    sq_returns = excess_returns**2

    min_periods = int(cfg.vol_com)
    ewma_var = sq_returns.ewm(com=cfg.vol_com, adjust=False, min_periods=min_periods).mean()

    annualized_var = ewma_var * cfg.vol_annualization_factor
    sigma = np.sqrt(annualized_var)

    # Use sigma computed through t-1 for time-t scaling (shift forward by one day)
    sigma_lagged = sigma.shift(1)

    return sigma_lagged
