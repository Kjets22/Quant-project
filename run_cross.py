"""
run_cross.py — QQQ <-> SPY with CROSS-ASSET features, hourly, regime-gated.

QQQ and SPY are tightly coupled and lead/lag each other, so each agent gets the
partner's return/momentum/trend/spread as extra inputs (causal). We test BOTH
regime policies head-to-head, dev set only (lockbox sealed):
  * all-weather : force long in up-trend + gated short in down-trend
  * bear-hunter : gated short in down-trend, NO forced long

Per (variant, ticker, fold): train hourly long/short/flat PPO with money reward +
turnover penalty + regime + cross features; evaluate OOS vs buy-and-hold.

  python run_cross.py            # launcher (all jobs parallel) + report
  python run_cross.py --variant allweather --ticker QQQ --fold 3   # one worker
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
OUT = HERE / "runs" / "phase3" / "cross"
OUT.mkdir(parents=True, exist_ok=True)
TICKERS = ["QQQ", "SPY"]
PARTNER = {"QQQ": "SPY", "SPY": "QQQ"}
K = 6
TF = "60min"
TIMESTEPS = int(os.environ.get("CROSS_TIMESTEPS", "150000"))
# A/B: all-weather + cross-asset, WITHOUT vs WITH support/resistance features.
VARIANTS = {"base": False, "sr": True}            # use_sr_features


def hourly(dev: pd.DataFrame) -> pd.DataFrame:
    g = dev.set_index("timestamp").resample(TF).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def aligned_hourly(ticker: str):
    """Hourly bars for `ticker` with a causal partner_close column from PARTNER."""
    from environment import align_partner
    from lockbox import Lockbox, load_dev_ticker
    lb = Lockbox.load_or_build()
    self_h = hourly(load_dev_ticker(ticker, lb))
    partner_h = hourly(load_dev_ticker(PARTNER[ticker], lb))
    return align_partner(self_h, partner_h)


def cfg_for(ticker: str, variant: str):
    from basket import ticker_cfg
    cfg = ticker_cfg(ticker)
    cfg.train.device = "cpu"
    cfg.train.total_timesteps = TIMESTEPS
    cfg.train.ent_coef = 0.01
    cfg.train.net_arch = "128,128"
    cfg.reward.reward_mode = "money"
    cfg.reward.flat_bonus = 0.01
    cfg.reward.turnover_penalty_w = 0.4
    cfg.reward.allow_short = True
    cfg.env.window = 96
    cfg.env.use_regime_features = True
    cfg.env.use_cross_features = True
    cfg.env.short_only_in_down = True
    cfg.env.force_long_in_up = True               # keep the proven all-weather gate
    cfg.env.use_sr_features = VARIANTS[variant]    # the A/B knob: support/resistance
    return cfg


def run_one(variant: str, ticker: str, fold: int) -> None:
    from agent import evaluate, train
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
    pd.DataFrame([{
        "variant": variant, "ticker": ticker, "fold": fold,
        "sharpe": res.sharpe, "pnl": res.raw_pnl, "buy_hold_pnl": res.buy_hold_pnl,
        "max_dd": res.max_drawdown, "trades": res.n_trades,
        "in_mkt": res.pct_in_market, "beats_bh": int(res.raw_pnl > res.buy_hold_pnl),
    }]).to_csv(out, index=False)
    print(f"DONE {variant}/{ticker}/f{fold}: P&L={res.raw_pnl:.2f} "
          f"(Sh {res.sharpe:.2f}) vs B&H={res.buy_hold_pnl:.2f}")


def launch(max_parallel: int = 14) -> None:
    import time
    print(f"Cross-asset QQQ<->SPY, hourly, {list(VARIANTS)} @ {TIMESTEPS} steps")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"; env["MKL_NUM_THREADS"] = "2"
    env["CT_TORCH_THREADS"] = "2"; env["PYTHONWARNINGS"] = "ignore"
    jobs = [(v, t, f) for v in VARIANTS for t in TICKERS for f in range(K)]
    running, done = [], 0
    while jobs or running:
        while jobs and len(running) < max_parallel:
            v, t, f = jobs.pop(0)
            running.append(subprocess.Popen(
                [sys.executable, "-u", "run_cross.py", "--variant", v,
                 "--ticker", t, "--fold", str(f)], cwd=str(HERE), env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        running = [p for p in running if p.poll() is None]
        time.sleep(3)
    report()


def report() -> None:
    files = sorted(OUT.glob("*_f*.csv"))
    if not files:
        print("no results.")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print("\n=== CROSS-ASSET QQQ<->SPY (hourly, OOS) — all-weather vs bear-hunter ===")
    for (v, tk), s in d.groupby(["variant", "ticker"]):
        print(f"  {v:11} {tk}: Sharpe {s.sharpe.mean():+.2f}  P&L {s.pnl.mean():8.2f}  "
              f"B&H {s.buy_hold_pnl.mean():8.2f}  beat-B&H {s.beats_bh.mean():.0%}  "
              f"trades {s.trades.mean():.0f}")
    print()
    for v, s in d.groupby("variant"):
        print(f"  POOLED {v:11}: beat-B&H {s.beats_bh.mean():.0%} of folds  "
              f"mean P&L {s.pnl.mean():.2f} vs B&H {s.buy_hold_pnl.mean():.2f}  "
              f"mean Sharpe {s.sharpe.mean():+.2f}")


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
