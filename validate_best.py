"""
validate_best.py — robustly validate the current best QQQ config on ALL folds.

The search uses only 2 folds at reduced steps (noisy). This runs the study's best
hyperparameters over all K folds at a higher step count, in parallel (one process
per fold), and reports per-fold Sharpe + buy-and-hold beat rate. Dev set only;
the lockbox stays sealed.

  python validate_best.py                 # launcher: all folds in parallel + report
  python validate_best.py --fold N        # one fold worker (internal)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "runs" / "phase3" / "best_val"
OUT.mkdir(parents=True, exist_ok=True)
K = 6
VAL_TIMESTEPS = int(os.environ.get("VAL_TIMESTEPS", "250000"))


def best_cfg():
    from basket import ticker_cfg
    from tune_qqq import make_study
    p = make_study().best_trial.params
    cfg = ticker_cfg("QQQ")
    cfg.train.device = "cpu"
    cfg.train.total_timesteps = VAL_TIMESTEPS
    cfg.reward.reward_mode = "money"
    cfg.train.learning_rate = p["lr"]
    cfg.train.ent_coef = p["ent_coef"]
    cfg.train.n_steps = p["n_steps"]
    cfg.train.gamma = p["gamma"]
    cfg.train.n_epochs = p["n_epochs"]
    cfg.train.net_arch = p["net_arch"]
    cfg.reward.flat_bonus = p["flat_bonus"]
    cfg.env.use_position_features = p["posfeat"]
    cfg.env.use_regime_features = p["regime"]
    cfg.env.use_alphatrend_features = p["alpha"]
    return cfg, p


def run_fold(fold: int) -> None:
    from agent import evaluate, train
    from lockbox import Lockbox, load_dev_ticker
    from validate import fold_bounds
    out = OUT / f"fold_{fold}.csv"
    if out.exists():
        print(f"skip fold {fold}")
        return
    cfg, _ = best_cfg()
    dev = load_dev_ticker("QQQ", Lockbox.load_or_build())
    n, window = len(dev), cfg.env.window
    tr_end, te_end = fold_bounds(n, K, window, fold)
    df_tr = dev.iloc[:tr_end].reset_index(drop=True)
    df_te = dev.iloc[tr_end:te_end].reset_index(drop=True)
    model = train(df_tr, cfg)
    res = evaluate(model, df_te, cfg, f"f{fold}")
    pd.DataFrame([{
        "fold": fold, "oos_sharpe": res.sharpe, "oos_raw_pnl": res.raw_pnl,
        "buy_hold_pnl": res.buy_hold_pnl, "oos_max_dd": res.max_drawdown,
        "n_trades": res.n_trades, "pct_in_market": res.pct_in_market,
        "beats_buy_hold": int(res.raw_pnl > res.buy_hold_pnl),
    }]).to_csv(out, index=False)
    print(f"DONE fold {fold}")


def launch() -> None:
    cfg, p = best_cfg()
    print("Validating BEST config on all", K, "folds @", VAL_TIMESTEPS, "steps")
    print("params:", p)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "3"; env["MKL_NUM_THREADS"] = "3"
    env["CT_TORCH_THREADS"] = "3"; env["PYTHONWARNINGS"] = "ignore"
    procs = []
    for f in range(K):
        procs.append(subprocess.Popen(
            [sys.executable, "-u", "validate_best.py", "--fold", str(f)],
            cwd=str(HERE), env=env))
    for p_ in procs:
        p_.wait()

    files = sorted(OUT.glob("fold_*.csv"))
    if not files:
        print("no fold results.")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print("\n=== BEST CONFIG — all-fold validation (QQQ, OOS) ===")
    for _, r in d.sort_values("fold").iterrows():
        print(f"  fold {int(r.fold)}: Sharpe={r.oos_sharpe:6.3f}  beatBH={int(r.beats_buy_hold)}"
              f"  P&L={r.oos_raw_pnl:8.2f} vs B&H={r.buy_hold_pnl:8.2f}"
              f"  trades={int(r.n_trades):5d}  inMkt={r.pct_in_market:.0%}")
    print(f"\n  POOLED: mean Sharpe={d.oos_sharpe.mean():.3f}  "
          f"beat-B&H={d.beats_buy_hold.mean():.0%} ({int(d.beats_buy_hold.sum())}/{len(d)})  "
          f"mean P&L={d.oos_raw_pnl.mean():.2f} vs B&H={d.buy_hold_pnl.mean():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    a = ap.parse_args()
    if a.fold is not None:
        run_fold(a.fold)
    else:
        launch()
