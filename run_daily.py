"""
Experiment A — DAILY swing trades (longer holds, less noise).

Resample to daily bars, force LOW turnover (heavy turnover penalty) so the agent
takes few, multi-week positions instead of day-trading. All-weather regime gating
(long in up-trend, gated short in down-trend, flat in chop). Compares the RL agent
to a simple daily SMA50 long/flat rule and to buy-and-hold. Dev set only.

  python run_daily.py            # launcher (parallel) + report
  python run_daily.py --ticker QQQ --fold 2
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
# DAILY_SR=1 -> add causal support/resistance features, into a separate folder so
# it's a clean A/B against the base daily run.
_SR = os.environ.get("DAILY_SR", "0") == "1"
OUT = HERE / "runs" / "phase3" / ("daily_sr" if _SR else "daily")
OUT.mkdir(parents=True, exist_ok=True)
TICKERS = ["QQQ", "SPY", "TLT"]
K = 4
TIMESTEPS = int(os.environ.get("DAILY_TIMESTEPS", "60000"))


def daily(dev: pd.DataFrame) -> pd.DataFrame:
    g = dev.set_index("timestamp").resample("1D").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def cfg_for(ticker: str):
    from basket import ticker_cfg
    cfg = ticker_cfg(ticker)
    cfg.train.device = "cpu"
    cfg.train.total_timesteps = TIMESTEPS
    cfg.train.ent_coef = 0.01
    cfg.train.net_arch = "128,128"
    cfg.reward.reward_mode = "money"
    cfg.reward.money_scale = 0.1            # daily $ moves are large -> scale down
    cfg.reward.flat_bonus = 0.02
    cfg.reward.turnover_penalty_w = 1.5     # HEAVY -> few, long-held trades
    cfg.reward.allow_short = True
    cfg.env.window = 40                     # 40 days of context
    cfg.env.use_regime_features = True
    cfg.env.short_only_in_down = True
    cfg.env.force_long_in_up = True
    cfg.env.use_sr_features = _SR          # support/resistance A/B knob (daily)
    cfg.env.sr_lookback = 60               # ~3 months of daily S/R levels
    return cfg


def sma_rule(close, n=50):
    s = pd.Series(close)
    sma = s.rolling(n).mean().shift(1).to_numpy()
    sig = np.nan_to_num((close > sma).astype(float))[:-1]
    rets = np.diff(close)
    price = close[:-1]
    gross = float((sig * rets).sum())
    cost = float((np.abs(np.diff(np.concatenate([[0.0], sig]))) * 0.0002 * price).sum())
    pl = sig * rets
    sh = float(pl.mean() / pl.std() * np.sqrt(len(pl))) if pl.std() > 1e-9 else 0.0
    return gross - cost, sh


def run_one(ticker, fold):
    from agent import evaluate, train
    from lockbox import Lockbox, load_dev_ticker
    from validate import fold_bounds
    out = OUT / f"{ticker}_f{fold}.csv"
    if out.exists():
        print(f"skip {ticker} f{fold}")
        return
    cfg = cfg_for(ticker)
    dev = daily(load_dev_ticker(ticker, Lockbox.load_or_build()))
    n, window = len(dev), cfg.env.window
    tr_end, te_end = fold_bounds(n, K, window, fold)
    df_tr = dev.iloc[:tr_end].reset_index(drop=True)
    df_te = dev.iloc[tr_end:te_end].reset_index(drop=True)
    model = train(df_tr, cfg)
    res = evaluate(model, df_te, cfg, f"{ticker}-f{fold}")
    rule_pnl, rule_sh = sma_rule(df_te["close"].to_numpy(float))
    pd.DataFrame([{
        "ticker": ticker, "fold": fold, "rl_sharpe": res.sharpe, "rl_pnl": res.raw_pnl,
        "buy_hold_pnl": res.buy_hold_pnl, "rule_pnl": rule_pnl, "rule_sharpe": rule_sh,
        "rl_trades": res.n_trades, "rl_beats_bh": int(res.raw_pnl > res.buy_hold_pnl),
        "rule_beats_bh": int(rule_pnl > res.buy_hold_pnl),
    }]).to_csv(out, index=False)
    print(f"DONE {ticker} f{fold}: RL={res.raw_pnl:.2f} (Sh {res.sharpe:.2f}) "
          f"trades={res.n_trades} | rule={rule_pnl:.2f} | B&H={res.buy_hold_pnl:.2f}")


def launch():
    print(f"DAILY swing on {TICKERS} @ {TIMESTEPS} steps, heavy turnover penalty")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"; env["MKL_NUM_THREADS"] = "2"
    env["CT_TORCH_THREADS"] = "2"; env["PYTHONWARNINGS"] = "ignore"
    procs = [subprocess.Popen(
        [sys.executable, "-u", "run_daily.py", "--ticker", t, "--fold", str(f)],
        cwd=str(HERE), env=env) for t in TICKERS for f in range(K)]
    for p in procs:
        p.wait()
    report()


def report():
    files = sorted(OUT.glob("*_f*.csv"))
    if not files:
        print("no results.")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print("\n=== DAILY SWING — RL vs SMA50 rule vs B&H (OOS) ===")
    for tk, s in d.groupby("ticker"):
        print(f"  {tk}: RL Sharpe {s.rl_sharpe.mean():+.2f} P&L {s.rl_pnl.mean():8.2f} "
              f"trades {s.rl_trades.mean():.0f} beat-B&H {s.rl_beats_bh.mean():.0%} | "
              f"rule P&L {s.rule_pnl.mean():8.2f} beat {s.rule_beats_bh.mean():.0%} | "
              f"B&H {s.buy_hold_pnl.mean():8.2f}")
    print(f"\n  POOLED: RL beats B&H {d.rl_beats_bh.mean():.0%}; "
          f"rule beats B&H {d.rule_beats_bh.mean():.0%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.ticker and a.fold is not None:
        run_one(a.ticker, a.fold)
    else:
        launch()
