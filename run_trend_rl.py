"""
run_trend_rl.py — the diagnosis-driven model: HOURLY bars, LONG-or-FLAT only,
regime-aware, money reward. Tests whether the RL agent can learn the trend-
participation edge (be long in uptrends, flat in downtrends) that beats B&H.

Per (ticker, fold): resample dev 5-min -> 1h, train a long/flat regime-aware PPO
agent, evaluate OOS, and ALSO score a simple SMA50 long/flat rule + B&H on the
same slice for a clean three-way comparison. Dev set only; lockbox sealed.

  python run_trend_rl.py            # launcher: all (ticker,fold) in parallel + report
  python run_trend_rl.py --ticker QQQ --fold 3   # one worker
"""

from __future__ import annotations

import argparse
import copy
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
OUT = HERE / "runs" / "phase3" / "trend_rl"
OUT.mkdir(parents=True, exist_ok=True)
TICKERS = ["QQQ", "TLT"]
K = 6
TF = "60min"
TIMESTEPS = int(os.environ.get("TREND_TIMESTEPS", "150000"))


def hourly(dev: pd.DataFrame) -> pd.DataFrame:
    g = dev.set_index("timestamp").resample(TF).agg(
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
    cfg.reward.flat_bonus = 0.01
    cfg.reward.turnover_penalty_w = float(os.environ.get("TURNOVER", "0.4"))  # anti-churn
    cfg.reward.allow_short = False          # LONG / FLAT only (no shorting)
    cfg.env.use_regime_features = True      # trend-awareness
    cfg.env.window = 96
    return cfg


def sma_rule(close: np.ndarray, n: int = 50):
    """Simple long/flat rule: long if close > SMA(n), else flat. Net of cost."""
    s = pd.Series(close)
    sma = s.rolling(n).mean().shift(1).to_numpy()
    sig = np.nan_to_num((close > sma).astype(float))[:-1]
    rets = np.diff(close)
    price = close[:-1]
    gross = float((sig * rets).sum())
    traded = np.abs(np.diff(np.concatenate([[0.0], sig])))
    cost = float((traded * 0.0002 * price).sum())
    pl = sig * rets
    sh = float(pl.mean() / pl.std() * np.sqrt(len(pl))) if pl.std() > 1e-9 else 0.0
    return gross - cost, sh


def run_one(ticker: str, fold: int) -> None:
    from agent import evaluate, train
    from lockbox import Lockbox, load_dev_ticker
    from validate import fold_bounds
    out = OUT / f"{ticker}_f{fold}.csv"
    if out.exists():
        print(f"skip {ticker} f{fold}")
        return
    cfg = cfg_for(ticker)
    dev = hourly(load_dev_ticker(ticker, Lockbox.load_or_build()))
    n, window = len(dev), cfg.env.window
    tr_end, te_end = fold_bounds(n, K, window, fold)
    df_tr = dev.iloc[:tr_end].reset_index(drop=True)
    df_te = dev.iloc[tr_end:te_end].reset_index(drop=True)
    model = train(df_tr, cfg)
    res = evaluate(model, df_te, cfg, f"{ticker}-f{fold}")
    rule_pnl, rule_sh = sma_rule(df_te["close"].to_numpy(float))
    pd.DataFrame([{
        "ticker": ticker, "fold": fold,
        "rl_sharpe": res.sharpe, "rl_pnl": res.raw_pnl,
        "buy_hold_pnl": res.buy_hold_pnl,
        "rule_pnl": rule_pnl, "rule_sharpe": rule_sh,
        "rl_trades": res.n_trades, "rl_in_mkt": res.pct_in_market,
        "rl_beats_bh": int(res.raw_pnl > res.buy_hold_pnl),
        "rule_beats_bh": int(rule_pnl > res.buy_hold_pnl),
    }]).to_csv(out, index=False)
    print(f"DONE {ticker} f{fold}: RL P&L={res.raw_pnl:.2f} (Sh {res.sharpe:.2f}) | "
          f"rule={rule_pnl:.2f} | B&H={res.buy_hold_pnl:.2f}")


def launch() -> None:
    print(f"Trend RL (hourly, long/flat, regime-aware) on {TICKERS} @ {TIMESTEPS} steps")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"; env["MKL_NUM_THREADS"] = "2"
    env["CT_TORCH_THREADS"] = "2"; env["PYTHONWARNINGS"] = "ignore"
    procs = []
    for t in TICKERS:
        for f in range(K):
            procs.append(subprocess.Popen(
                [sys.executable, "-u", "run_trend_rl.py", "--ticker", t, "--fold", str(f)],
                cwd=str(HERE), env=env))
    for p in procs:
        p.wait()
    report()


def report() -> None:
    files = sorted(OUT.glob("*_f*.csv"))
    if not files:
        print("no results.")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print("\n=== HOURLY LONG/FLAT REGIME RL  vs  trend-rule  vs  B&H (OOS) ===")
    for tk, s in d.groupby("ticker"):
        print(f"\n  {tk} ({len(s)} folds):")
        print(f"    RL agent   : mean Sharpe {s.rl_sharpe.mean():+.2f}  "
              f"P&L {s.rl_pnl.mean():8.2f}  beat-B&H {s.rl_beats_bh.mean():.0%}  "
              f"trades {s.rl_trades.mean():.0f}  inMkt {s.rl_in_mkt.mean():.0%}")
        print(f"    trend rule : mean Sharpe {s.rule_sharpe.mean():+.2f}  "
              f"P&L {s.rule_pnl.mean():8.2f}  beat-B&H {s.rule_beats_bh.mean():.0%}")
        print(f"    buy & hold : P&L {s.buy_hold_pnl.mean():8.2f}")
    print(f"\n  POOLED: RL beats B&H {d.rl_beats_bh.mean():.0%} of folds; "
          f"rule beats B&H {d.rule_beats_bh.mean():.0%} of folds")


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
