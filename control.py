"""
control.py — Phase A: the honest, properly-powered baseline.

Runs the existing walk-forward across every ticker on the DEV SET ONLY (the
lockbox is never touched here), collecting per-fold out-of-sample metrics. Every
fold is logged as a trial. The result in runs/control_baseline.csv is the number
every later phase (B–F) must beat on out-of-sample data.

The bar that matters is BUY-AND-HOLD (raw_pnl vs buy_hold_pnl), not random.
"""

from __future__ import annotations

import sys

try:  # pragma: no cover - force UTF-8 so the console never crashes on glyphs
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Respect a parallel-launch thread budget so N concurrent control processes do
# not oversubscribe the CPU (set by run_full_control.py).
_ct_threads = os.environ.get("CT_TORCH_THREADS")
if _ct_threads:
    try:
        import torch

        torch.set_num_threads(max(1, int(_ct_threads)))
    except Exception:  # noqa: BLE001
        pass

from basket import BASKET_TICKERS, ticker_cfg
from lockbox import Lockbox, load_dev_ticker
from trials import log_trial
from validate import walk_forward

RUNS_DIR = Path(__file__).with_name("runs")
CONTROL_CSV = RUNS_DIR / "control_baseline.csv"


def run_control(tickers: list[str], k_folds: int, timesteps: int,
                out_csv: Path = CONTROL_CSV, phase: str = "A",
                device: str = "cpu", use_position_features: bool = False) -> pd.DataFrame:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lockbox = Lockbox.load_or_build(tickers)
    print(f"lockbox cutoff = {lockbox.cutoff}  (dev = everything before this)")
    print(f"device = {device}  (benchmark: CPU single-env is ~3.7x faster than GPU here)")
    print(f"use_position_features = {use_position_features}")

    rows: list[dict] = []
    for t in tickers:
        dev = load_dev_ticker(t, lockbox)
        cfg = ticker_cfg(t)
        cfg.train.total_timesteps = timesteps
        cfg.train.device = device
        cfg.env.use_position_features = use_position_features
        print(f"\n##### {t}: dev rows={len(dev)}  folds={k_folds}  "
              f"timesteps={timesteps} #####")
        folds = walk_forward(dev, cfg, k=k_folds)
        for fr in folds:
            beats_bh = fr.test.raw_pnl > fr.test.buy_hold_pnl
            beats_rnd = fr.test.capture_reward > fr.random.capture_reward
            row = {
                "ticker": t,
                "fold": fr.fold,
                "oos_capture": fr.test.capture_reward,
                "oos_raw_pnl": fr.test.raw_pnl,
                "buy_hold_pnl": fr.test.buy_hold_pnl,
                "random_capture": fr.random.capture_reward,
                "oos_sharpe": fr.test.sharpe,
                "train_sharpe": fr.train.sharpe,
                "oos_max_dd": fr.test.max_drawdown,
                "n_trades": fr.test.n_trades,
                "pct_in_market": fr.test.pct_in_market,
                "beats_buy_hold": int(beats_bh),
                "beats_random": int(beats_rnd),
            }
            rows.append(row)
            log_trial(
                phase=phase,
                params={"model": "control_ppo", "ticker": t, "fold": fr.fold,
                        "timesteps": timesteps, "k_folds": k_folds,
                        "use_position_features": use_position_features},
                metrics={"oos_sharpe": round(fr.test.sharpe, 4),
                         "oos_capture": round(fr.test.capture_reward, 4),
                         "oos_raw_pnl": round(fr.test.raw_pnl, 4),
                         "buy_hold_pnl": round(fr.test.buy_hold_pnl, 4),
                         "beats_buy_hold": int(beats_bh)},
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    _summary(df, out_csv)
    return df


def _summary(df: pd.DataFrame, out_csv: Path) -> None:
    print("\n" + "=" * 78)
    print(f"CONTROL BASELINE  ->  {out_csv}")
    print("=" * 78)
    if df.empty:
        print("no folds produced (data too short?)")
        return

    # Per-ticker.
    print(f"{'ticker':>6} | {'folds':>5} {'oos_cap':>8} {'oos_shrp':>8} "
          f"{'oos_MDD':>8} {'%>B&H':>6} {'%>rnd':>6}")
    print("-" * 60)
    for t, g in df.groupby("ticker"):
        print(f"{t:>6} | {len(g):>5} {g['oos_capture'].mean():>8.2f} "
              f"{g['oos_sharpe'].mean():>8.3f} {g['oos_max_dd'].mean():>8.2f} "
              f"{g['beats_buy_hold'].mean():>6.0%} {g['beats_random'].mean():>6.0%}")

    print("-" * 60)
    print("POOLED (out-of-sample, all tickers/folds):")
    print(f"  folds total       : {len(df)}")
    print(f"  capture reward    : mean {df['oos_capture'].mean():.3f}  "
          f"std {df['oos_capture'].std(ddof=0):.3f}")
    print(f"  OOS Sharpe        : mean {df['oos_sharpe'].mean():.3f}  "
          f"std {df['oos_sharpe'].std(ddof=0):.3f}")
    print(f"  OOS max drawdown  : mean {df['oos_max_dd'].mean():.3f}  "
          f"std {df['oos_max_dd'].std(ddof=0):.3f}")
    print(f"  % folds beat B&H  : {df['beats_buy_hold'].mean():.0%}   <-- THE BAR")
    print(f"  % folds beat rnd  : {df['beats_random'].mean():.0%}   (table-stakes)")
    print(f"  vs random capture : {df['oos_capture'].mean():.2f} "
          f"vs {df['random_capture'].mean():.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the Phase-A control baseline")
    p.add_argument("--tickers", type=str, default=None,
                   help="comma list; default = full basket")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--out", type=str, default=str(CONTROL_CSV))
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "auto"])
    p.add_argument("--position-features", action="store_true",
                   help="enable the Enhancement-1 position-awareness block")
    p.add_argument("--phase", type=str, default="A")
    args = p.parse_args()

    tickers = ([s.strip().upper() for s in args.tickers.split(",")]
               if args.tickers else BASKET_TICKERS)
    run_control(tickers, k_folds=args.folds, timesteps=args.timesteps,
                out_csv=Path(args.out), device=args.device, phase=args.phase,
                use_position_features=args.position_features)
