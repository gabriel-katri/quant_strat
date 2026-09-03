"""
Diagnostics for the Step 2 Kalman filter (melone_kalman.py):

1. Estimated (Q, R) per factor and the Q/R ratio -- a large Q/R means the
   filter treats the state as almost free to move every period (tracks
   closely, small residual); a tiny Q/R means it behaves close to the
   static full-sample regression.
2. ln F_t (actual) vs. the Kalman fitted value (beta_t' M_t), one subplot
   per factor -- makes it visible if the filter is tracking too closely
   (fitted ~ actual everywhere, ECT ~ 0, no mean-reversion signal left to
   exploit).
3. ECT re-computed with q manually divided by 10 and by 100 (r unchanged),
   to see how sensitive the ECT is to the MLE's q estimate.

Requires results/rolling_lasso_dataset.csv -- run main.py first if it
doesn't exist.

Run with: python diagnose_kalman.py
"""

from __future__ import annotations

import os
import sys
import warnings

import pandas as pd

import melone_construct as construct_mod
import melone_kalman as kalman_mod
import plots as plots_mod
from config import Config

warnings.filterwarnings("ignore")
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v: .6f}")


def _section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def main() -> None:
    cfg = Config()
    dataset_path = os.path.join(cfg.results_dir, cfg.output_csv)
    if not os.path.exists(dataset_path):
        sys.exit(f"{dataset_path} not found -- run `python main.py` first.")
    os.makedirs(cfg.results_dir, exist_ok=True)

    df = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
    factor_prices, macro_levels = construct_mod.build_levels(df, cfg)

    _section(f"Calibrating Kalman filter (Q, R via MLE on the initial {cfg.rolling_window_periods}Q window)")
    kalman_results = kalman_mod.run_all_kalman(factor_prices, macro_levels, cfg)

    _section("Q (state noise), R (observation noise), and Q/R ratio")
    qr_table = pd.DataFrame({
        f: {"R (obs var)": kalman_results[f]["r"], "Q (state var)": kalman_results[f]["q"],
            "Q/R ratio": kalman_results[f]["q"] / kalman_results[f]["r"]}
        for f in cfg.unconstrained_factors
    }).T
    print(qr_table.to_string())
    qr_path = os.path.join(cfg.results_dir, "melone_kalman_qr_diagnostic.csv")
    qr_table.to_csv(qr_path)
    print(f"\nSaved {qr_path}")

    _section("Plot: ln F_t (actual) vs. Kalman fitted value (beta_t' M_t)")
    fit_path = plots_mod.plot_kalman_fit_vs_actual(factor_prices, kalman_results, cfg)
    print(f"Saved {fit_path}")

    _section("Re-filtering with q manually divided by 10 and by 100 (r unchanged)")
    ect_variants = {"q (MLE)": {}, "q/10": {}, "q/100": {}}
    for f in cfg.unconstrained_factors:
        params, scale = kalman_results[f]["params"], kalman_results[f]["scale"]
        ect_variants["q (MLE)"][f] = kalman_results[f]["ect"]
        for divisor, label in [(10.0, "q/10"), (100.0, "q/100")]:
            _beta_t, ect = kalman_mod.filter_with_q_divisor(
                factor_prices[f], macro_levels, params, scale, divisor, cfg)
            ect_variants[label][f] = ect

    q_sensitivity_table = pd.DataFrame({
        f: {label: kalman_results[f]["q"] / divisor for label, divisor in
            [("q (MLE)", 1.0), ("q/10", 10.0), ("q/100", 100.0)]}
        for f in cfg.unconstrained_factors
    }).T
    print(q_sensitivity_table.to_string())

    sensitivity_path = plots_mod.plot_kalman_ect_q_sensitivity(ect_variants, cfg)
    print(f"Saved {sensitivity_path}")

    ect_variants_wide = pd.concat(
        {label: pd.concat(ect_by_factor, axis=1) for label, ect_by_factor in ect_variants.items()}, axis=1)
    ect_variants_path = os.path.join(cfg.results_dir, "melone_kalman_ect_q_sensitivity.csv")
    ect_variants_wide.to_csv(ect_variants_path)
    print(f"Saved {ect_variants_path}")


if __name__ == "__main__":
    sys.exit(main())
