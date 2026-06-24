"""
Error-regime experiment (QQQ + SPY only).

1. Tweaked hyperparameters (bigger 256x256 net, higher LR, more entropy, more steps).
2. Per-regime P&L breakdown so we SEE which regime the model loses in (up/down/chop).
3. A/B: base vs 'chopguard' = force FLAT in confirmed chop (a dedicated policy for
   the regime the trend agent usually gets wrong). Hourly all-weather + cross-asset.
Dev set only.

  python run_errorregime.py          # launcher + report
  python run_errorregime.py --variant chopguard --ticker QQQ --fold 3
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
OUT = HERE / "runs" / "phase3" / "errorregime"
OUT.mkdir(parents=True, exist_ok=True)
TICKERS = ["QQQ", "SPY"]
K = 6
TIMESTEPS = int(os.environ.get("ER_TIMESTEPS", "200000"))
VARIANTS = {"base": False, "chopguard": True}     # flat_in_chop


def cfg_for(ticker: str, variant: str):
    from basket import ticker_cfg
    cfg = ticker_cfg(ticker)
    cfg.train.device = "cpu"
    cfg.train.total_timesteps = TIMESTEPS
    cfg.train.learning_rate = 5e-4          # tweaked
    cfg.train.ent_coef = 0.02               # tweaked: more exploration
    cfg.train.net_arch = "256,256"          # tweaked: bigger net
    cfg.train.n_steps = 4096                # tweaked: longer rollouts
    cfg.reward.reward_mode = "money"
    cfg.reward.flat_bonus = 0.01
    cfg.reward.turnover_penalty_w = 0.6
    cfg.reward.allow_short = True
    cfg.env.window = 96
    cfg.env.use_regime_features = True
    cfg.env.use_cross_features = True
    cfg.env.short_only_in_down = True
    cfg.env.force_long_in_up = True
    cfg.env.flat_in_chop = VARIANTS[variant]    # the error-regime A/B knob
    return cfg


def eval_with_regime(model, df_te, cfg):
    from agent import _max_drawdown, _sharpe
    from environment import CaptureTradingEnv
    from regime import regime_id
    env = CaptureTradingEnv(df_te, cfg)
    reg = regime_id(env.close_px, min_run=12)
    obs, _ = env.reset(seed=0)
    prev_eq = 0.0
    prev_pos = 0
    per = {0: 0.0, 1: 0.0, 2: 0.0}      # P&L by regime (up/down/chop)
    step_pnls, equity = [], []
    trades = steps = 0
    done = False
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        bar = env.t                      # regime of the bar about to be traded
        obs, _r, term, trunc, info = env.step(int(a))
        d = info["equity"] - prev_eq
        prev_eq = info["equity"]
        per[int(reg[bar])] += d
        step_pnls.append(d)
        equity.append(info["equity"])
        if info["position"] != prev_pos:
            trades += 1
        prev_pos = info["position"]
        steps += 1
        done = term or trunc
    close = df_te["close"].to_numpy()
    return {
        "sharpe": _sharpe(np.asarray(step_pnls)), "pnl": prev_eq,
        "buy_hold_pnl": float(close[-1] - close[env.window + 1]),
        "max_dd": _max_drawdown(np.asarray(equity)), "trades": trades,
        "pnl_up": per[0], "pnl_down": per[1], "pnl_chop": per[2],
    }


def run_one(variant, ticker, fold):
    from agent import train
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
    m = eval_with_regime(model, df_te, cfg)
    m.update(variant=variant, ticker=ticker, fold=fold,
             beats_bh=int(m["pnl"] > m["buy_hold_pnl"]))
    pd.DataFrame([m]).to_csv(out, index=False)
    print(f"DONE {variant}/{ticker}/f{fold}: P&L={m['pnl']:.2f} (up {m['pnl_up']:.1f} "
          f"down {m['pnl_down']:.1f} chop {m['pnl_chop']:.1f}) vs B&H={m['buy_hold_pnl']:.2f}")


def launch():
    print(f"Error-regime (tweaked hp) QQQ/SPY, base vs chopguard @ {TIMESTEPS}")
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
                [sys.executable, "-u", "run_errorregime.py", "--variant", v,
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
    print("\n=== ERROR-REGIME (tweaked hp) — where does P&L come from? ===")
    for (v, tk), s in d.groupby(["variant", "ticker"]):
        print(f"  {v:9} {tk}: P&L {s.pnl.mean():7.2f} (up {s.pnl_up.mean():6.1f} "
              f"down {s.pnl_down.mean():6.1f} chop {s.pnl_chop.mean():6.1f}) "
              f"B&H {s.buy_hold_pnl.mean():6.2f} Sh {s.sharpe.mean():+.2f} "
              f"beat {s.beats_bh.mean():.0%} trades {s.trades.mean():.0f}")
    print()
    for v, s in d.groupby("variant"):
        print(f"  POOLED {v:9}: beat-B&H {s.beats_bh.mean():.0%}  Sharpe {s.sharpe.mean():+.2f}  "
              f"P&L {s.pnl.mean():.2f} vs B&H {s.buy_hold_pnl.mean():.2f}  "
              f"(chop P&L {s.pnl_chop.mean():+.1f})")


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
