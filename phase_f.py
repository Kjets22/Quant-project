"""
phase_f.py — THE LOCKBOX TEST (opened exactly once).

Freezes the all-weather regime-conditioned config, trains on ALL dev data, and
evaluates ONCE on the sealed lockbox (most recent 12 months, never used in any
tuning). Reports true out-of-sample performance per ticker vs buy-and-hold.

Frozen config: hourly bars, window=96, money reward, turnover_penalty=0.4,
regime features, force-long in confirmed up-trends, gated-short in confirmed
down-trends. (This is the config selected after dev-set development; the lockbox
was never consulted during that process.)

  python phase_f.py                 # launcher: all tickers in parallel + report
  python phase_f.py --ticker QQQ    # one ticker worker
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
OUT = HERE / "runs" / "phase3" / "lockbox"
OUT.mkdir(parents=True, exist_ok=True)
TICKERS = [t.strip().upper() for t in os.environ.get(
    "TICKERS", "SPY,QQQ,AAPL,MSFT,NVDA,JPM,XLE,TLT").split(",") if t.strip()]
TF = "60min"
TIMESTEPS = int(os.environ.get("F_TIMESTEPS", "150000"))


def hourly(df: pd.DataFrame) -> pd.DataFrame:
    g = df.set_index("timestamp").resample(TF).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def frozen_cfg(ticker: str):
    from basket import ticker_cfg
    cfg = ticker_cfg(ticker)
    cfg.train.device = "cpu"
    cfg.train.total_timesteps = TIMESTEPS
    cfg.train.ent_coef = 0.01
    cfg.train.net_arch = "128,128"
    cfg.env.window = 96
    cfg.env.use_regime_features = True
    cfg.env.short_only_in_down = True       # gated short
    cfg.env.force_long_in_up = True         # forced long in confirmed up-trend
    cfg.reward.reward_mode = "money"
    cfg.reward.flat_bonus = 0.01
    cfg.reward.turnover_penalty_w = 0.4
    cfg.reward.allow_short = True
    return cfg


def run_ticker(ticker: str) -> None:
    from agent import evaluate, train
    from data import load_from_cache
    from basket import ticker_cfg
    from lockbox import Lockbox
    out = OUT / f"result_{ticker}.csv"
    if out.exists():
        print(f"skip {ticker}")
        return
    full = load_from_cache(ticker_cfg(ticker))
    lb = Lockbox.load()
    dev_full = lb.dev(full)                              # dev (free)
    lock = lb.open_lockbox({ticker: full}, phase="F")[ticker]   # LOCKBOX (Phase F only)

    cfg = frozen_cfg(ticker)
    df_tr = hourly(dev_full)
    df_te = hourly(lock)
    model = train(df_tr, cfg)
    res = evaluate(model, df_te, cfg, f"{ticker}-LOCKBOX")
    pd.DataFrame([{
        "ticker": ticker,
        "oos_sharpe": res.sharpe, "oos_pnl": res.raw_pnl,
        "buy_hold_pnl": res.buy_hold_pnl, "oos_max_dd": res.max_drawdown,
        "n_trades": res.n_trades, "pct_in_market": res.pct_in_market,
        "beats_buy_hold": int(res.raw_pnl > res.buy_hold_pnl),
    }]).to_csv(out, index=False)
    print(f"DONE {ticker}: Sharpe={res.sharpe:.2f}  P&L={res.raw_pnl:.2f} "
          f"vs B&H={res.buy_hold_pnl:.2f}  beatBH={int(res.raw_pnl > res.buy_hold_pnl)}")


def launch() -> None:
    print("=" * 60)
    print("PHASE F — LOCKBOX (opened once). Frozen all-weather config.")
    print(f"tickers: {TICKERS}   timesteps: {TIMESTEPS}")
    print("=" * 60)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"; env["MKL_NUM_THREADS"] = "2"
    env["CT_TORCH_THREADS"] = "2"; env["PYTHONWARNINGS"] = "ignore"
    procs = [subprocess.Popen(
        [sys.executable, "-u", "phase_f.py", "--ticker", t], cwd=str(HERE), env=env)
        for t in TICKERS]
    for p in procs:
        p.wait()
    report()


def report() -> None:
    files = sorted(OUT.glob("result_*.csv"))
    if not files:
        print("no lockbox results.")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print("\n" + "=" * 70)
    print("LOCKBOX RESULT (true out-of-sample, most recent 12 months)")
    print("=" * 70)
    print(f"  {'ticker':>6} {'Sharpe':>8} {'agent P&L':>10} {'B&H P&L':>10} "
          f"{'beatBH':>7} {'maxDD':>8} {'trades':>7} {'inMkt':>6}")
    for _, r in d.sort_values("buy_hold_pnl").iterrows():
        print(f"  {r.ticker:>6} {r.oos_sharpe:>8.2f} {r.oos_pnl:>10.2f} "
              f"{r.buy_hold_pnl:>10.2f} {int(r.beats_buy_hold):>7} "
              f"{r.oos_max_dd:>8.2f} {int(r.n_trades):>7} {r.pct_in_market:>6.0%}")
    print("-" * 70)
    fell = d[d.buy_hold_pnl < 0]
    rose = d[d.buy_hold_pnl >= 0]
    print(f"  ALL tickers : beat B&H {d.beats_buy_hold.mean():.0%} "
          f"({int(d.beats_buy_hold.sum())}/{len(d)})  mean Sharpe {d.oos_sharpe.mean():+.2f}")
    if len(fell):
        print(f"  FELL (B&H<0): beat B&H {fell.beats_buy_hold.mean():.0%} "
              f"({int(fell.beats_buy_hold.sum())}/{len(fell)})  mean Sharpe {fell.oos_sharpe.mean():+.2f}")
    if len(rose):
        print(f"  ROSE (B&H>=0): beat B&H {rose.beats_buy_hold.mean():.0%} "
              f"({int(rose.beats_buy_hold.sum())}/{len(rose)})  mean Sharpe {rose.oos_sharpe.mean():+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.ticker:
        run_ticker(a.ticker)
    else:
        launch()
