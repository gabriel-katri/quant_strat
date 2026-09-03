"""
Step 2 (Kalman): time-varying-beta cointegration via a linear-Gaussian
state-space model, in two variants -- both are estimated and reported; the
prompt this was built against is explicit that neither is a "winner" to be
selected on OOS performance.

  Observation (both variants): ln F_t = x_t' beta_t + eps_t, eps_t ~ N(0, r)

  (a) Unconstrained (random walk):
        beta_{t+1} = beta_t + eta_t,                     eta_t ~ N(0, q * I_k)

  (b) Mean-reverting:
        beta_{t+1} = phi*beta_t + (1-phi)*beta_static + eta_t,  eta_t ~ N(0, q * I_k)
      where beta_static is Melone's own fixed cointegrating vector from the
      static model (melone_ect.py, full-sample OLS) and phi in (0,1) is a
      persistence parameter, estimated (not hand-set).

x_t = [1, trend_t, M_t (4 macro driver levels)], so beta_t has 6 states:
intercept, trend, and one loading per macro driver -- the same design as
the static long-run regression, but letting the coefficients evolve over
time instead of fitting one set for the whole sample.

For both variants, a single scalar q (shared innovation variance across all
6 states) and a single scalar observation variance r are estimated by
MLE -- this keeps the hyperparameter search low-dimensional and numerically
stable given the sample size, vs. estimating a separate variance per state.
Variant (b) additionally estimates a single scalar phi (shared persistence
across all 6 states), via a logit-transformed free parameter so phi is
mechanically constrained to (0,1) -- true mean reversion, never a random
walk (phi=1) or instant snap-back (phi=0) by construction.

Q, R (and phi for variant b) are estimated on the initial
`rolling_window_periods` quarters (1975Q1-1999Q4) and then FIXED; the full
sample is then filtered once with those fixed hyperparameters. Because the
Kalman filter is already a single causal forward pass (beta_t depends only
on y_1..t, x_1..t), this filtering step introduces no look-ahead on its
own -- only the downstream predictive regression (delta) still needs
real-time expanding-window re-estimation (see run_oos_for_factor). Note
variant (b)'s beta_static anchor IS a full-sample quantity (by the prompt's
own definition -- "Melone's own fixed cointegrating vector from the static
model"), so unlike beta_tilde_t's period-to-period dynamics, the anchor
itself is not point-in-time real-time; treat variant (b)'s OOS numbers with
that caveat in mind.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.statespace.mlemodel import MLEModel

import melone_ect as ect_mod
from config import Config


class TVPRegression(MLEModel):
    """Time-varying-parameter regression, scalar (obs_var, state_var) state space."""

    def __init__(self, endog: np.ndarray, exog: np.ndarray):
        k_states = exog.shape[1]
        super().__init__(endog, k_states=k_states, k_posdef=k_states, initialization="diffuse")
        self["design"] = exog.T[np.newaxis, :, :]
        self["transition"] = np.eye(k_states)
        self["selection"] = np.eye(k_states)
        self._k_states = k_states

    @property
    def param_names(self):
        return ["log_r", "log_q"]

    @property
    def start_params(self):
        v = float(np.var(self.endog))
        return np.array([np.log(v * 0.5 + 1e-8), np.log(v * 0.01 + 1e-10)])

    def transform_params(self, unconstrained):
        return unconstrained

    def untransform_params(self, constrained):
        return constrained

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self["obs_cov", 0, 0] = np.exp(params[0])
        self["state_cov"] = np.exp(params[1]) * np.eye(self._k_states)


def build_design(index: pd.Index, macro_levels: pd.DataFrame, trend: pd.Series) -> pd.DataFrame:
    """x_t = [1, trend_t, M_t...], one row per date."""
    const = pd.Series(1.0, index=index, name="const")
    return pd.concat([const, trend, macro_levels], axis=1)


def _scale_factors(X_train: pd.DataFrame) -> pd.Series:
    """Per-column std on the calibration window (1.0 for ~constant columns, e.g. 'const').

    The design columns span wildly different scales (const ~O(1), trend
    ~O(1-40), POT_GDP_GROWTH ~O(0.005), TERM_SPREAD ~O(1-3)); feeding that
    directly into a diffuse-initialized Kalman filter with a single shared
    state-noise variance q makes the small-scale columns numerically
    unstable (a q sized for O(1) columns is enormous relative to a
    POT_GDP_GROWTH-scale coefficient). Dividing each column by its std
    before filtering, then dividing the recovered beta_t by the same
    scale, is an exact reparametrization (x_j*beta_j is unchanged) that
    just makes all 6 states live on comparable numeric scales internally.
    """
    std = X_train.std(ddof=0)
    return std.where(std > 1e-8, 1.0)


def calibrate_q_r(factor_price: pd.Series, macro_levels: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, pd.Series]:
    """MLE-estimate (log_r, log_q) on the initial `rolling_window_periods` quarters.

    Returns (params, scale) -- `scale` (from this same window) is reused to
    rescale the design matrix everywhere downstream, so calibration and
    filtering stay on the same internal units.
    """
    n_train = cfg.rolling_window_periods
    price_train = factor_price.iloc[:n_train]
    macro_train = macro_levels.iloc[:n_train]
    trend_train = ect_mod.build_trend(price_train.index)
    X_train = build_design(price_train.index, macro_train, trend_train)
    scale = _scale_factors(X_train)
    X_train_scaled = X_train / scale

    mod = TVPRegression(price_train.values, X_train_scaled.values)
    try:
        res = mod.fit(disp=False, maxiter=500)
        if not np.all(np.isfinite(res.params)):
            raise ValueError("non-finite MLE params")
        return res.params, scale
    except Exception as e:
        v = float(np.var(price_train.values))
        fallback = np.array([np.log(v * 0.5 + 1e-8), np.log(v * 0.001 + 1e-10)])
        print(f"[WARN] Kalman MLE calibration failed ({e!r}); using fallback (r,q) prior.")
        return fallback, scale


def filter_full_sample(factor_price: pd.Series, macro_levels: pd.DataFrame, params: np.ndarray, scale: pd.Series,
                        cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Filter the FULL sample with (q, r) fixed at `params`. Returns (beta_t df, kalman ECT_t)."""
    trend = ect_mod.build_trend(factor_price.index)
    X = build_design(factor_price.index, macro_levels, trend)
    X_scaled = X / scale

    mod = TVPRegression(factor_price.values, X_scaled.values)
    res = mod.filter(params)

    beta_cols = ["const", "trend"] + list(macro_levels.columns)
    beta_t_scaled = pd.DataFrame(res.filtered_state.T, index=factor_price.index, columns=beta_cols)
    beta_t = beta_t_scaled / scale.values  # undo the rescaling: back to original (x_j, beta_j) units

    fitted = (X.values * beta_t.values).sum(axis=1)
    kalman_ect = pd.Series(factor_price.values - fitted, index=factor_price.index)
    return beta_t, kalman_ect


def filter_with_q_divisor(factor_price: pd.Series, macro_levels: pd.DataFrame, params: np.ndarray,
                           scale: pd.Series, q_divisor: float, cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Diagnostic: re-filter with q manually divided by `q_divisor` (r unchanged),
    instead of the MLE-estimated q. log(q/d) = log(q) - log(d)."""
    scaled_params = params.copy()
    scaled_params[1] = params[1] - np.log(q_divisor)
    return filter_full_sample(factor_price, macro_levels, scaled_params, scale, cfg)


def run_kalman_for_factor(factor_price: pd.Series, macro_levels: pd.DataFrame, cfg: Config) -> dict:
    params, scale = calibrate_q_r(factor_price, macro_levels, cfg)
    beta_t, kalman_ect = filter_full_sample(factor_price, macro_levels, params, scale, cfg)
    return {"params": params, "scale": scale, "r": float(np.exp(params[0])), "q": float(np.exp(params[1])),
            "beta_t": beta_t, "ect": kalman_ect}


def run_all_kalman(factor_prices: pd.DataFrame, macro_levels: pd.DataFrame, cfg: Config) -> dict:
    """Returns {factor: {'params', 'scale', 'r', 'q', 'beta_t', 'ect'}}."""
    return {f: run_kalman_for_factor(factor_prices[f], macro_levels, cfg) for f in cfg.unconstrained_factors}


def run_oos_for_factor(factor_return: pd.Series, kalman_ect: pd.Series, cfg: Config) -> pd.DataFrame:
    """Expanding-window predictive regression of r_{t+1} on the Kalman ECT_t.

    Only (a, delta) are re-estimated at each step -- the ECT series itself
    is already causal (see module docstring), so it does not need
    re-filtering inside this loop.
    """
    min_train = cfg.rolling_window_periods
    dates = factor_return.index

    rows = []
    for t in range(min_train, len(dates)):
        ret_train = factor_return.iloc[1:t]
        x_train = kalman_ect.iloc[: t - 1]
        X = sm.add_constant(x_train.values)
        pred_model = sm.OLS(ret_train.values, X).fit()

        forecast = float(pred_model.params[0] + pred_model.params[1] * kalman_ect.iloc[t - 1])
        rows.append({"date": dates[t], "actual": float(factor_return.iloc[t]), "predicted": forecast})

    return pd.DataFrame(rows).set_index("date")


def run_all_oos(dataset: pd.DataFrame, kalman_results: dict, cfg: Config) -> dict:
    return {f: run_oos_for_factor(dataset[f], kalman_results[f]["ect"], cfg) for f in cfg.unconstrained_factors}


# --------------------------------------------------------------------------------
# Variant (b): mean-reverting toward the static cointegrating vector
# --------------------------------------------------------------------------------

class TVPRegressionMeanReverting(MLEModel):
    """Time-varying-parameter regression with a mean-reverting (AR(1)) state,
    reparametrized around the static anchor so the underlying filter is a
    standard zero-mean state space:

        y_tilde_t = x_t' beta_tilde_t + eps_t,          eps_t ~ N(0, r)
        beta_tilde_{t+1} = phi * beta_tilde_t + eta_t,   eta_t ~ N(0, q * I_k)

    where y_tilde_t = y_t - x_t'beta_static and beta_tilde_t = beta_t -
    beta_static are both deviations from the fixed static anchor. Recovering
    beta_t = beta_tilde_t (filtered) + beta_static happens outside this
    class (see filter_full_sample_mean_reverting). phi is estimated via a
    logit-transformed free parameter, mechanically constraining it to (0,1).
    """

    def __init__(self, endog_tilde: np.ndarray, exog: np.ndarray):
        k_states = exog.shape[1]
        super().__init__(endog_tilde, k_states=k_states, k_posdef=k_states, initialization="diffuse")
        self["design"] = exog.T[np.newaxis, :, :]
        self["selection"] = np.eye(k_states)
        self._k_states = k_states

    @property
    def param_names(self):
        return ["log_r", "log_q", "phi_logit"]

    @property
    def start_params(self):
        v = float(np.var(self.endog))
        return np.array([np.log(v * 0.5 + 1e-8), np.log(v * 0.01 + 1e-10), 0.0])  # phi_logit=0 -> phi=0.5

    def transform_params(self, unconstrained):
        return unconstrained

    def untransform_params(self, constrained):
        return constrained

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        phi = 1.0 / (1.0 + np.exp(-params[2]))
        self["obs_cov", 0, 0] = np.exp(params[0])
        self["state_cov"] = np.exp(params[1]) * np.eye(self._k_states)
        self["transition"] = phi * np.eye(self._k_states)


def calibrate_mean_reverting(factor_price: pd.Series, macro_levels: pd.DataFrame, beta_static: pd.Series,
                              cfg: Config) -> tuple[np.ndarray, pd.Series]:
    """MLE-estimate (log_r, log_q, phi_logit) on the initial `rolling_window_periods`
    quarters. Returns (params, scale)."""
    n_train = cfg.rolling_window_periods
    price_train = factor_price.iloc[:n_train]
    macro_train = macro_levels.iloc[:n_train]
    trend_train = ect_mod.build_trend(price_train.index)
    X_train = build_design(price_train.index, macro_train, trend_train)
    beta_cols = X_train.columns
    y_tilde_train = price_train - X_train.values @ beta_static.reindex(beta_cols).values

    scale = _scale_factors(X_train)
    X_train_scaled = X_train / scale

    mod = TVPRegressionMeanReverting(y_tilde_train.values, X_train_scaled.values)
    try:
        res = mod.fit(disp=False, maxiter=500)
        if not np.all(np.isfinite(res.params)):
            raise ValueError("non-finite MLE params")
        return res.params, scale
    except Exception as e:
        v = float(np.var(y_tilde_train.values))
        fallback = np.array([np.log(v * 0.5 + 1e-8), np.log(v * 0.001 + 1e-10), 0.0])
        print(f"[WARN] Kalman (mean-reverting) MLE calibration failed ({e!r}); using fallback (r,q,phi) prior.")
        return fallback, scale


def filter_full_sample_mean_reverting(factor_price: pd.Series, macro_levels: pd.DataFrame, beta_static: pd.Series,
                                       params: np.ndarray, scale: pd.Series,
                                       cfg: Config) -> tuple[pd.DataFrame, pd.Series]:
    """Filter the FULL sample with (q, r, phi) fixed at `params`. Returns (beta_t df, ECT_t)."""
    trend = ect_mod.build_trend(factor_price.index)
    X = build_design(factor_price.index, macro_levels, trend)
    beta_cols = X.columns
    beta_static_vec = beta_static.reindex(beta_cols).values
    y_tilde = factor_price - X.values @ beta_static_vec
    X_scaled = X / scale

    mod = TVPRegressionMeanReverting(y_tilde.values, X_scaled.values)
    res = mod.filter(params)

    beta_tilde_scaled = pd.DataFrame(res.filtered_state.T, index=factor_price.index, columns=beta_cols)
    beta_tilde = beta_tilde_scaled / scale.values
    beta_t = beta_tilde + beta_static_vec  # broadcasts beta_static back across every row

    fitted = (X.values * beta_t.values).sum(axis=1)
    ect = pd.Series(factor_price.values - fitted, index=factor_price.index)
    return beta_t, ect


def run_kalman_mean_reverting_for_factor(factor_price: pd.Series, macro_levels: pd.DataFrame,
                                          beta_static: pd.Series, cfg: Config) -> dict:
    params, scale = calibrate_mean_reverting(factor_price, macro_levels, beta_static, cfg)
    beta_t, ect = filter_full_sample_mean_reverting(factor_price, macro_levels, beta_static, params, scale, cfg)
    phi = float(1.0 / (1.0 + np.exp(-params[2])))
    return {"params": params, "scale": scale, "r": float(np.exp(params[0])), "q": float(np.exp(params[1])),
            "phi": phi, "beta_t": beta_t, "ect": ect}


def run_all_kalman_mean_reverting(factor_prices: pd.DataFrame, macro_levels: pd.DataFrame,
                                   beta_static_by_factor: dict, cfg: Config) -> dict:
    """Returns {factor: {'params', 'scale', 'r', 'q', 'phi', 'beta_t', 'ect'}}.

    `beta_static_by_factor` = {factor: pd.Series} of full-sample static
    coefficients (index = ['const', 'trend'] + macro column names), e.g.
    from melone_ect.build_beta_table(...) sliced back into per-factor Series.
    """
    return {
        f: run_kalman_mean_reverting_for_factor(factor_prices[f], macro_levels, beta_static_by_factor[f], cfg)
        for f in cfg.unconstrained_factors
    }
