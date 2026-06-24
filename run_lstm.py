"""
Experiment B — a DIFFERENT MODEL: recurrent LSTM policy (RecurrentPPO).

Same hourly all-weather + cross-asset setup as the best feed-forward config, but
with an LSTM policy (temporal memory -> "more understanding over time, less noise")
and a higher learning rate. Tests whether a recurrent model captures the trend-
participation edge better than the MLP. Compared head-to-head vs B&H (and the prior
MLP base: QQQ +0.73 Sharpe / +32.5 P&L, SPY -0.10 / -0.3). Dev set only.

  python run_lstm.py            # launcher (parallel) + report
  python run_lstm.py --ticker QQQ --fold 3
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
OUT = HERE / "runs" / "phase3" / "lstm"
OUT.mkdir(parents=True, exist_ok=True)
TICKERS = ["QQQ", "SPY"]
K = 6
TIMESTEPS = int(os.environ.get("LSTM_TIMESTEPS", "150000"))


def cfg_for(ticker: str):
    from basket import ticker_cfg
    cfg = ticker_cfg(ticker)
    cfg.train.device = "cpu"
    cfg.train.total_timesteps = TIMESTEPS
    cfg.train.learning_rate = 5e-4          # higher LR (per request)
    cfg.train.ent_coef = 0.01
    cfg.reward.reward_mode = "money"
    cfg.reward.flat_bonus = 0.01
    cfg.reward.turnover_penalty_w = 0.4
    cfg.reward.allow_short = True
    cfg.env.window = 96
    cfg.env.use_regime_features = True
    cfg.env.use_cross_features = True
    cfg.env.short_only_in_down = True
    cfg.env.force_long_in_up = True
    return cfg


def train_lstm(df_tr, cfg):
    import warnings
    warnings.filterwarnings("ignore")
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from environment import CaptureTradingEnv
    venv = DummyVecEnv([lambda: CaptureTradingEnv(df_tr, cfg)])
    model = RecurrentPPO(
        "MlpLstmPolicy", venv, n_steps=2048, learning_rate=cfg.train.learning_rate,
        gamma=cfg.train.gamma, ent_coef=cfg.train.ent_coef, seed=cfg.train.seed,
        device="cpu", verbose=0)
    model.learn(total_timesteps=cfg.train.total_timesteps)
    return model


def eval_lstm(model, df_te, cfg):
    from agent import _max_drawdown, _sharpe
    from environment import CaptureTradingEnv
    env = CaptureTradingEnv(df_te, cfg)
    obs, _ = env.reset(seed=0)
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    step_pnls, equity = [], []
    prev_eq = 0.0
    prev_pos = 0
    trades = in_mkt = steps = 0
    done = False
    while not done:
        action, lstm_states = model.predict(
            obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, _r, term, trunc, info = env.step(int(action))
        episode_starts = np.array([term or trunc])
        step_pnls.append(info["equity"] - prev_eq)
        prev_eq = info["equity"]
        equity.append(info["equity"])
        if info["position"] != 0:
            in_mkt += 1
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
        "in_mkt": in_mkt / max(steps, 1),
    }


def run_one(ticker, fold):
    from run_cross import aligned_hourly
    from validate import fold_bounds
    out = OUT / f"{ticker}_f{fold}.csv"
    if out.exists():
        print(f"skip {ticker} f{fold}")
        return
    cfg = cfg_for(ticker)
    dev = aligned_hourly(ticker)
    n, window = len(dev), cfg.env.window
    tr_end, te_end = fold_bounds(n, K, window, fold)
    df_tr = dev.iloc[:tr_end].reset_index(drop=True)
    df_te = dev.iloc[tr_end:te_end].reset_index(drop=True)
    model = train_lstm(df_tr, cfg)
    m = eval_lstm(model, df_te, cfg)
    m.update(ticker=ticker, fold=fold, beats_bh=int(m["pnl"] > m["buy_hold_pnl"]))
    pd.DataFrame([m]).to_csv(out, index=False)
    print(f"DONE {ticker} f{fold}: LSTM P&L={m['pnl']:.2f} (Sh {m['sharpe']:.2f}) "
          f"vs B&H={m['buy_hold_pnl']:.2f}")


def launch():
    print(f"LSTM (RecurrentPPO) hourly all-weather+cross on {TICKERS} @ {TIMESTEPS}")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "3"; env["MKL_NUM_THREADS"] = "3"
    env["CT_TORCH_THREADS"] = "3"; env["PYTHONWARNINGS"] = "ignore"
    procs = [subprocess.Popen(
        [sys.executable, "-u", "run_lstm.py", "--ticker", t, "--fold", str(f)],
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
    print("\n=== LSTM (recurrent) hourly all-weather+cross — vs B&H (OOS) ===")
    for tk, s in d.groupby("ticker"):
        print(f"  {tk}: Sharpe {s.sharpe.mean():+.2f}  P&L {s.pnl.mean():8.2f}  "
              f"B&H {s.buy_hold_pnl.mean():8.2f}  beat-B&H {s.beats_bh.mean():.0%}  "
              f"trades {s.trades.mean():.0f}")
    print(f"\n  POOLED: beat-B&H {d.beats_bh.mean():.0%}  "
          f"mean Sharpe {d.sharpe.mean():+.2f}  "
          f"(MLP base was QQQ +0.73 / SPY -0.10)")


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
