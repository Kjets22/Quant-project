"""
ab_compare.py — Phase 3 A/B verdict (training-free; safe to run anytime).

Compares a variant baseline CSV against the Phase-A control baseline on the
dev-set walk-forward metrics, and applies the keep/discard rule:

  KEEP if, out-of-sample, the variant improves mean reward OR mean Sharpe,
  OR reduces the fold-to-fold Sharpe std (steadier) — without collapsing return.
  Otherwise DISCARD.

Both CSVs are produced by control.py / run_full_control.py and have identical
columns. This script does no training and touches no running job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

RUNS_DIR = Path(__file__).with_name("runs")


def _agg(df: pd.DataFrame) -> dict:
    return {
        "folds": len(df),
        "mean_capture": df["oos_capture"].mean(),
        "mean_sharpe": df["oos_sharpe"].mean(),
        "std_sharpe": df["oos_sharpe"].std(ddof=0),
        "mean_maxdd": df["oos_max_dd"].mean(),
        "pct_beat_bh": df["beats_buy_hold"].mean(),
        "mean_raw_pnl": df["oos_raw_pnl"].mean(),
    }


def compare(control_csv: Path, variant_csv: Path, label: str = "variant") -> None:
    ctrl = pd.read_csv(control_csv)
    var = pd.read_csv(variant_csv)
    c, v = _agg(ctrl), _agg(var)

    print("=" * 72)
    print(f"A/B: {label}  vs  CONTROL")
    print(f"  control = {control_csv}")
    print(f"  variant = {variant_csv}")
    print("=" * 72)
    rows = [
        ("mean OOS capture", c["mean_capture"], v["mean_capture"], "higher"),
        ("mean OOS Sharpe", c["mean_sharpe"], v["mean_sharpe"], "higher"),
        ("OOS Sharpe std", c["std_sharpe"], v["std_sharpe"], "lower"),
        ("mean OOS max DD", c["mean_maxdd"], v["mean_maxdd"], "lower"),
        ("% folds beat B&H", c["pct_beat_bh"], v["pct_beat_bh"], "higher"),
        ("mean OOS raw P&L", c["mean_raw_pnl"], v["mean_raw_pnl"], "higher"),
    ]
    print(f"{'metric':>18} | {'control':>10} {'variant':>10} {'delta':>10}  want")
    print("-" * 64)
    for name, cv, vv, want in rows:
        print(f"{name:>18} | {cv:>10.3f} {vv:>10.3f} {vv-cv:>+10.3f}  {want}")

    # Keep/discard rule.
    improved_reward = v["mean_capture"] > c["mean_capture"]
    improved_sharpe = v["mean_sharpe"] > c["mean_sharpe"]
    steadier = v["std_sharpe"] < c["std_sharpe"]
    collapsed = v["mean_raw_pnl"] < 0 and v["mean_raw_pnl"] < c["mean_raw_pnl"] - abs(c["mean_raw_pnl"])
    keep = (improved_reward or improved_sharpe or steadier) and not collapsed

    print("-" * 64)
    print("VERDICT:")
    print(f"  reward improved? {improved_reward}  sharpe improved? {improved_sharpe}"
          f"  steadier? {steadier}  return collapsed? {collapsed}")
    if keep:
        print(f"  [KEEP] '{label}' beats the control out-of-sample -> keep the flag.")
    else:
        print(f"  [DISCARD] '{label}' does not beat the control -> revert the flag.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="A/B compare a variant vs the control baseline")
    p.add_argument("--control", type=str, default=str(RUNS_DIR / "control_baseline.csv"))
    p.add_argument("--variant", type=str, required=True)
    p.add_argument("--label", type=str, default="variant")
    args = p.parse_args()
    compare(Path(args.control), Path(args.variant), args.label)
