"""
regime_moe.py — regime mixture-of-experts on QQQ (the user's idea).

Per fold: train N_REGIMES PPO experts, each with its money reward MASKED to one
causal regime (up / down / chop) so it specializes. At inference a router applies
expert[regime[t]] at each bar. We then measure the routed strategy vs buy-and-hold
out-of-sample. Dev set only; lockbox sealed. Causal regime id (past-only).

  python regime_moe.py              # launcher: all folds in parallel + report
  python regime_moe.py --fold N     # one fold worker (trains 3 experts + routes)
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
OUT = HERE / "runs" / "phase3" / "moe"
OUT.mkdir(parents=True, exist_ok=True)
K = 6
TIMESTEPS = int(os.environ.get("MOE_TIMESTEPS", "150000"))


def base_cfg():
    from basket import ticker_cfg
    cfg = ticker_cfg("QQQ")
    cfg.train.device = "cpu"
    cfg.train.total_timesteps = TIMESTEPS
    cfg.train.learning_rate = 3e-4
    cfg.train.n_steps = 2048
    cfg.train.gamma = 0.99
    cfg.train.ent_coef = 0.01
    cfg.train.net_arch = "128,128"
    cfg.reward.reward_mode = "money"
    cfg.reward.flat_bonus = 0.005
    cfg.env.regime_min_run = 24       # sticky regime (~2h) to avoid whipsaw switching
    return cfg


def route_eval(experts, df_te, cfg_base) -> dict:
    from agent import _max_drawdown, _sharpe
    from environment import CaptureTradingEnv
    from regime import regime_id
    cfg = copy.deepcopy(cfg_base)
    cfg.reward.active_regime = -1                      # real reward/equity at eval
    env = CaptureTradingEnv(df_te, cfg)
    reg = regime_id(env.close_px, min_run=int(cfg.env.regime_min_run))
    obs, _ = env.reset(seed=0)
    step_pnls, equity_curve = [], []
    prev_eq = 0.0
    prev_pos = 0
    in_mkt = trades = steps = 0
    done = False
    while not done:
        r = int(reg[env.t])
        action, _ = experts[r].predict(obs, deterministic=True)
        obs, _rew, term, trunc, info = env.step(int(action))
        step_pnls.append(info["equity"] - prev_eq)
        prev_eq = info["equity"]
        equity_curve.append(info["equity"])
        if info["position"] != 0:
            in_mkt += 1
        if info["position"] != prev_pos:
            trades += 1
        prev_pos = info["position"]
        steps += 1
        done = term or trunc
    close = df_te["close"].to_numpy()
    bh = float(close[-1] - close[env.window + 1])
    return {
        "oos_sharpe": _sharpe(np.asarray(step_pnls)),
        "oos_raw_pnl": prev_eq,
        "buy_hold_pnl": bh,
        "oos_max_dd": _max_drawdown(np.asarray(equity_curve)),
        "n_trades": trades,
        "pct_in_market": in_mkt / max(steps, 1),
        "beats_buy_hold": int(prev_eq > bh),
    }


def run_fold(fold: int) -> None:
    from agent import train
    from lockbox import Lockbox, load_dev_ticker
    from regime import N_REGIMES
    from validate import fold_bounds
    out = OUT / f"moe_fold_{fold}.csv"
    if out.exists():
        print(f"skip fold {fold}")
        return
    cfg_base = base_cfg()
    dev = load_dev_ticker("QQQ", Lockbox.load_or_build())
    n, window = len(dev), cfg_base.env.window
    tr_end, te_end = fold_bounds(n, K, window, fold)
    df_tr = dev.iloc[:tr_end].reset_index(drop=True)
    df_te = dev.iloc[tr_end:te_end].reset_index(drop=True)

    experts = []
    for r in range(N_REGIMES):
        cfg = copy.deepcopy(cfg_base)
        cfg.reward.active_regime = r            # mask reward to regime r
        cfg.train.seed = 42 + r
        print(f"fold {fold}: training expert for regime {r} ...", flush=True)
        experts.append(train(df_tr, cfg))

    m = route_eval(experts, df_te, cfg_base)
    m["fold"] = fold
    pd.DataFrame([m]).to_csv(out, index=False)
    print(f"DONE fold {fold}: routed Sharpe={m['oos_sharpe']:.3f} "
          f"P&L={m['oos_raw_pnl']:.2f} vs B&H={m['buy_hold_pnl']:.2f} "
          f"beatBH={m['beats_buy_hold']}")


def launch() -> None:
    print(f"Regime MoE on QQQ: {K} folds x 3 experts @ {TIMESTEPS} steps each")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "3"; env["MKL_NUM_THREADS"] = "3"
    env["CT_TORCH_THREADS"] = "3"; env["PYTHONWARNINGS"] = "ignore"
    procs = [subprocess.Popen(
        [sys.executable, "-u", "regime_moe.py", "--fold", str(f)],
        cwd=str(HERE), env=env) for f in range(K)]
    for p in procs:
        p.wait()
    report()


def report() -> None:
    files = sorted(OUT.glob("moe_fold_*.csv"))
    if not files:
        print("no MoE results yet.")
        return
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print("\n=== REGIME MIXTURE-OF-EXPERTS — QQQ out-of-sample ===")
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
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report()
    elif a.fold is not None:
        run_fold(a.fold)
    else:
        launch()
