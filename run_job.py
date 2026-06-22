"""
run_job.py — run ONE (tag, ticker, fold) walk-forward job and write one CSV row.

Folds are independent trainings, so splitting work to the (ticker, fold) level
lets many jobs run in parallel and fill idle CPU cores. Each job is resumable:
if its result file already exists, it exits immediately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from basket import ticker_cfg
from lockbox import Lockbox, load_dev_ticker
from trials import log_trial
from validate import run_one_fold

JOBS_DIR = Path(__file__).with_name("runs") / "phase3" / "jobs"


def job_path(tag: str, ticker: str, fold: int) -> Path:
    return JOBS_DIR / f"{tag}__{ticker}__f{fold}.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--fold", type=int, required=True, help="1-based fold number")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--position-features", action="store_true")
    ap.add_argument("--reward-json", default=None)
    ap.add_argument("--phase", default="X")
    a = ap.parse_args()

    out = job_path(a.tag, a.ticker, a.fold)
    if out.exists():
        print(f"skip {out.name} (already done)")
        return
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    lb = Lockbox.load_or_build()
    dev = load_dev_ticker(a.ticker, lb)
    cfg = ticker_cfg(a.ticker)
    cfg.train.total_timesteps = a.timesteps
    cfg.train.device = "cpu"
    cfg.env.use_position_features = a.position_features
    overrides = json.loads(a.reward_json) if a.reward_json else {}
    for k, v in overrides.items():
        if not hasattr(cfg.reward, k):
            raise ValueError(f"unknown RewardConfig field: {k}")
        setattr(cfg.reward, k, v)

    fr = run_one_fold(dev, cfg, a.k, a.fold - 1)   # convert to 0-based index
    if fr is None:
        print(f"{a.tag}/{a.ticker}/f{a.fold}: test block too short; no row.")
        return

    beats_bh = fr.test.raw_pnl > fr.test.buy_hold_pnl
    beats_rnd = fr.test.capture_reward > fr.random.capture_reward
    row = {
        "ticker": a.ticker, "fold": fr.fold,
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
    pd.DataFrame([row]).to_csv(out, index=False)
    log_trial(
        phase=a.phase,
        params={"model": "job", "tag": a.tag, "ticker": a.ticker, "fold": fr.fold,
                "timesteps": a.timesteps, "use_position_features": a.position_features,
                **{f"rw_{k}": v for k, v in overrides.items()}},
        metrics={"oos_sharpe": round(fr.test.sharpe, 4),
                 "oos_capture": round(fr.test.capture_reward, 4),
                 "oos_raw_pnl": round(fr.test.raw_pnl, 4),
                 "buy_hold_pnl": round(fr.test.buy_hold_pnl, 4),
                 "beats_buy_hold": int(beats_bh)},
    )
    print(f"DONE {out.name}")


if __name__ == "__main__":
    main()
