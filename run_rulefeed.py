"""
Feed the SIMPLE RULE's signal to the bot and see if it can match/beat the rule.

A/B (QQQ + SPY, hourly, all-weather + cross):
  norule = the agent as-is
  rule   = the agent ALSO gets the simple rule's own inputs (above SMA20/50/200 + dist)
We score BOTH against the actual SMA50 long/flat rule and B&H. The questions:
  (a) does giving the bot the rule's info beat the bot without it?
  (b) can the bot (even with the rule handed to it) match or beat the simple rule,
      or does the RL add noise and underperform the one-line rule?
Dev set only.

  python run_rulefeed.py            # launcher + report
  python run_rulefeed.py --variant rule --ticker QQQ --fold 3
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
OUT = HERE / "runs" / "phase3" / "rulefeed"
OUT.mkdir(parents=True, exist_ok=True)
TICKERS = ["QQQ", "SPY"]
K = 6
TIMESTEPS = int(os.environ.get("RULE_TIMESTEPS", "150000"))
VARIANTS = {"norule": False, "rule": True}     # use_rule_features


def cfg_for(ticker, variant):
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
    cfg.env.force_long_in_up = True
    cfg.env.use_rule_features = VARIANTS[variant]    # hand the bot the rule's signal
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
    rule_pnl, rule_sh = sma_rule(df_te["close"].to_numpy(float))
    pd.DataFrame([{
        "variant": variant, "ticker": ticker, "fold": fold,
        "bot_sharpe": res.sharpe, "bot_pnl": res.raw_pnl, "buy_hold_pnl": res.buy_hold_pnl,
        "rule_pnl": rule_pnl, "rule_sharpe": rule_sh, "trades": res.n_trades,
        "bot_beats_rule": int(res.raw_pnl > rule_pnl),
        "bot_beats_bh": int(res.raw_pnl > res.buy_hold_pnl),
    }]).to_csv(out, index=False)
    print(f"DONE {variant}/{ticker}/f{fold}: bot={res.raw_pnl:.1f} "
          f"rule={rule_pnl:.1f} B&H={res.buy_hold_pnl:.1f}")


def launch():
    print(f"Rule-feed A/B (norule vs rule) QQQ/SPY @ {TIMESTEPS}")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"; env["MKL_NUM_THREADS"] = "2"
    env["CT_TORCH_THREADS"] = "2"; env["PYTHONWARNINGS"] = "ignore"
    import time
    jobs = [(v, t, f) for v in VARIANTS for t in TICKERS for f in range(K)]
    running = []
    while jobs or running:
        while jobs and len(running) < 14:
            v, t, f = jobs.pop(0)
            running.append(subprocess.Popen(
                [sys.executable, "-u", "run_rulefeed.py", "--variant", v,
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
    print("\n=== RULE-FEED — can the bot match/beat the simple rule given its signal? ===")
    for (v, tk), s in d.groupby(["variant", "ticker"]):
        print(f"  {v:6} {tk}: bot P&L {s.bot_pnl.mean():7.2f} (Sh {s.bot_sharpe.mean():+.2f})  "
              f"rule P&L {s.rule_pnl.mean():7.2f} (Sh {s.rule_sharpe.mean():+.2f})  "
              f"B&H {s.buy_hold_pnl.mean():6.2f}  bot>rule {s.bot_beats_rule.mean():.0%}")
    print()
    for v, s in d.groupby("variant"):
        print(f"  POOLED {v:6}: bot beats rule {s.bot_beats_rule.mean():.0%} of folds  "
              f"bot P&L {s.bot_pnl.mean():.2f} vs rule {s.rule_pnl.mean():.2f} vs B&H {s.buy_hold_pnl.mean():.2f}")
    print("  READ: if 'rule' variant >> 'norule', the rule signal helped the bot.")
    print("        if bot < rule even with the signal, the RL adds noise -> use the rule.")


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
