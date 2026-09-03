"""
Rolling-window Lasso: predicts each factor's quarterly return from the
already-lagged macro drivers (OIL_WTI_LOGRET, POT_GDP_GROWTH,
TERM_SPREAD_10Y3M, LIQUIDITY_PS).

At each step t, trains on the trailing `rolling_window_periods` (e.g. 100
quarters) observations [t-window, t-1], picks lambda via LassoCV with a
TimeSeriesSplit (no shuffling, so validation folds are always later in
time than their training fold -- no look-ahead within the window), and
predicts the return at t. Predictors are standardized using only the
training window's mean/std (also applied to the test point), so there is
no look-ahead in the standardization either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit

from config import Config


def _standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0, ddof=0)
    sigma[sigma == 0] = 1.0  # guard against a degenerate (constant) predictor in-window
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def rolling_lasso_for_factor(df: pd.DataFrame, factor: str, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling-window Lasso for a single factor.

    Returns (predictions_df, coefficients_df), both indexed by the
    prediction date.
    """
    predictors = cfg.lasso_predictors
    window = cfg.rolling_window_periods
    dates = df.index

    pred_rows, coef_rows = [], []
    for t in range(window, len(df)):
        train = df.iloc[t - window : t]
        test = df.iloc[t]

        X_train = train[predictors].to_numpy(dtype=float)
        y_train = train[factor].to_numpy(dtype=float)
        X_test = test[predictors].to_numpy(dtype=float).reshape(1, -1)
        y_test = float(test[factor])

        X_train_std, X_test_std = _standardize(X_train, X_test)

        cv = TimeSeriesSplit(n_splits=cfg.cv_splits)
        model = LassoCV(cv=cv, max_iter=cfg.lasso_max_iter, random_state=cfg.lasso_random_state)
        model.fit(X_train_std, y_train)
        y_pred = float(model.predict(X_test_std)[0])

        date = dates[t]
        pred_rows.append({"date": date, "actual": y_test, "predicted": y_pred, "alpha": model.alpha_})
        coef_rows.append({"date": date, **dict(zip(predictors, model.coef_))})

    pred_df = pd.DataFrame(pred_rows).set_index("date")
    coef_df = pd.DataFrame(coef_rows).set_index("date")
    return pred_df, coef_df


def run_all_factors(df: pd.DataFrame, cfg: Config) -> dict:
    """Returns {factor: {'predictions': df, 'coefficients': df}}."""
    out = {}
    for factor in cfg.lasso_target_factors:
        pred_df, coef_df = rolling_lasso_for_factor(df, factor, cfg)
        out[factor] = {"predictions": pred_df, "coefficients": coef_df}
    return out


def oos_r2(actual: pd.Series, predicted: pd.Series) -> float:
    """Standard out-of-sample R^2 vs. the OOS-period sample mean of actual."""
    sse_model = np.sum((actual - predicted) ** 2)
    sse_mean = np.sum((actual - actual.mean()) ** 2)
    return 1.0 - sse_model / sse_mean


def build_r2_table(results: dict, cfg: Config) -> pd.DataFrame:
    rows = []
    for factor in cfg.lasso_target_factors:
        pred_df = results[factor]["predictions"]
        rows.append({
            "Factor": factor,
            "N obs": len(pred_df),
            "OOS R2 (%)": oos_r2(pred_df["actual"], pred_df["predicted"]) * 100,
        })
    return pd.DataFrame(rows).set_index("Factor")


def hit_rate(actual: pd.Series, predicted: pd.Series) -> float:
    """Share of periods where sign(predicted) == sign(actual)."""
    return float((np.sign(predicted) == np.sign(actual)).mean())


def build_hit_rate_table(results: dict, cfg: Config) -> pd.DataFrame:
    rows = []
    for factor in cfg.lasso_target_factors:
        pred_df = results[factor]["predictions"]
        rows.append({
            "Factor": factor,
            "N obs": len(pred_df),
            "Hit rate (%)": hit_rate(pred_df["actual"], pred_df["predicted"]) * 100,
        })
    return pd.DataFrame(rows).set_index("Factor")
