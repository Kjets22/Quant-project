"""
Swing-trade test: same hourly all-weather + cross-asset setup, but A/B the
TURNOVER PENALTY so the agent holds positions for a swing (hours-to-days) instead
of scalping. scalp = light penalty (0.4, current); swing = heavy penalty (2.5).
Reports average hold length so we can see it actually swing-trades. Dev set only.

  python run_swing.py            # launcher (parallel) + report
  python run_swing.py --variant swing --ticker QQQ --fold 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "runs" / "phase3" / "swing"
OUT.mkdir(parents=True, exist_ok=True)
TICKERS = ["QQQ", "SPY"]
K = 6
TIMESTEPS = int(os.environ.get("SWING_TIMESTEPS", "150000"))
VARIANTS = {"scalp": 0.4, "swing": 2.5}     # turnover_penalty_w


def cfg_for(ticker: str, variant: str):
    from basket import ticker_cfg
    cfg = ticker_cfg(ticker)
    cfg.train.device = "cpu"
    cfg.train.total_timesteps = TIMESTEPS
    cfg.train.ent_coef = 0.01
    cfg.train.net_arch = "128,128"
    cfg.reward.reward_mode = "money"
    cfg.reward.flat_bonus = 0.01
    cfg.reward.turnover_penalty_w = VARIANTS[variant]    # the swing knob
    cfg.reward.allow_short = True
    cfg.env.window = 96
    cfg.env.use_regime_features = True
    cfg.env.use_cross_features = True
    cfg.env.short_only_in_down = True
    cfg.env.force_long_in_up = True
    return cfg


def run_one(variant, ticker, fold):
    from agent import evaluate, train
    from run_cross import aligned_hourly
    from validate import fold_bounds
    out = OUT / f"{variant}_{ticker}_f{fold}.csv"
    if out.exists():
        print(f"skip {variant}/{ticker}/f{fold}")
        return
    cfg = cfg_for(ticker, variant)
    dev = aligned_hourly(ticker)
    n, window = len(dev), cfg.env.window
    tr_end, te_end = fold_bounds(n, K, window, fold)
    df_tr = dev.iloc[:tr_end].reset_index(drop=True)
    df_te = dev.iloc[tr_end:te_end].reset_index(drop=True)
    model = train(df_tr, cfg)
    res = evaluate(model, df_te, cfg, f"{variant}-{ticker}-f{fold}")
    hold = (res.n_steps * res.pct_in_market) / max(res.n_trades, 1)  # ~bars per trade
    pd.DataFrame([{
        "variant": variant, "ticker": ticker, "fold": fold,
        "sharpe": res.sharpe, "pnl": res.raw_pnl, "buy_hold_pnl": res.buy_hold_pnl,
        "trades": res.n_trades, "hold_bars": hold,
        "beats_bh": int(res.raw_pnl > res.buy_hold_pnl),
    }]).to_csv(out, index=False)
    print(f"DONE {variant}/{ticker}/f{fold}: P&L={res.raw_pnl:.2f} (Sh {res.sharpe:.2f}) "
          f"trades={res.n_trades} hold~{hold:.0f}h vs B&H={res.buy_hold_pnl:.2f}")


def launch():
    print(f"Swing A/B (turnover {VARIANTS}) hourly on {TICKERS} @ {TIMESTEPS}")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"; env["MKL_NUM_THREADS"] = "2"
    env["CT_TORCH_THREADS"] = "2"; env["PYTHONWARNINGS"] = "ignore"
    jobs = [(v, t, f) for v in VARIANTS for t in TICKERS for f in range(K)]
    import time
    running = []
    while jobs or running:
        while jobs and len(running) < 14:
            v, t, f = jobs.pop(0)
            running.append(subprocess.Popen(
                [sys.executable, "-u", "run_swing.py", "--variant", v,
                 "--ticker", t, "--fold", str(f)], cwd=str(HERE), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        running = [p for p in running if p.poll() is None]
        time.sleep(3)
    report()


def report():
    files = sorted(OUT.glob("*_f*.csv"))
    if not files:
        print("no results.")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print("\n=== SWING vs SCALP (hourly all-weather+cross, OOS) ===")
    for (v, tk), s in d.groupby(["variant", "ticker"]):
        print(f"  {v:5} {tk}: Sharpe {s.sharpe.mean():+.2f}  P&L {s.pnl.mean():8.2f}  "
              f"B&H {s.buy_hold_pnl.mean():8.2f}  beat-B&H {s.beats_bh.mean():.0%}  "
              f"trades {s.trades.mean():.0f}  hold ~{s.hold_bars.mean():.0f}h")
    print()
    for v, s in d.groupby("variant"):
        print(f"  POOLED {v:5}: beat-B&H {s.beats_bh.mean():.0%}  Sharpe {s.sharpe.mean():+.2f}  "
              f"P&L {s.pnl.mean():.2f} vs B&H {s.buy_hold_pnl.mean():.2f}  "
              f"avg hold ~{s.hold_bars.mean():.0f}h")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None)
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.variant and a.ticker and a.fold is not None:
        run_one(a.variant, a.ticker, a.fold)
    else:
        launch()
