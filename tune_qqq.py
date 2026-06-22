"""
tune_qqq.py — Phase E: parallel Optuna hyperparameter search on QQQ.

Goal: keep improving the money-reward agent on QQQ by searching PPO
hyperparameters AND the feature flags (position features, regime, AlphaTrend).
The study is persisted to a shared SQLite DB so many worker processes can run in
parallel (saturating the CPU) and the search resumes/improves across bursts.

Usage:
  python tune_qqq.py --worker --timeout 480     # one search worker (time-boxed)
  python tune_qqq.py --report                    # print best trial + leaderboard

Objective = mean out-of-sample Sharpe on QQQ over SEARCH_FOLDS (dev set only;
lockbox stays sealed). Each trial also records its buy-and-hold beat rate.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

RUNS = Path(__file__).with_name("runs") / "phase3"
RUNS.mkdir(parents=True, exist_ok=True)
DB = f"sqlite:///{(RUNS / 'tune_qqq.db').as_posix()}"
STUDY = "qqq_money"

TICKER = "QQQ"
K = 6                       # fold grid (same as validation)
# Search on a balanced pair: fold 1 (B&H DOWN) + fold 4 (B&H UP), so a do-nothing
# agent cannot win (it would lose the full up-move on fold 4). Env overrides allow
# a fast smoke (TUNE_TIMESTEPS / TUNE_FOLDS).
SEARCH_FOLDS = [int(x) for x in os.environ.get("TUNE_FOLDS", "1,4").split(",")]
SEARCH_TIMESTEPS = int(os.environ.get("TUNE_TIMESTEPS", "100000"))


def objective(trial) -> float:
    import optuna  # noqa
    from agent import evaluate, train
    from basket import ticker_cfg
    from lockbox import Lockbox, load_dev_ticker
    from validate import fold_bounds

    cfg = ticker_cfg(TICKER)
    cfg.train.total_timesteps = SEARCH_TIMESTEPS
    cfg.train.device = "cpu"
    cfg.reward.reward_mode = "money"

    # --- search space ---
    cfg.train.learning_rate = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    cfg.train.ent_coef = trial.suggest_float("ent_coef", 0.0, 0.02)
    cfg.train.n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])
    cfg.train.gamma = trial.suggest_categorical("gamma", [0.95, 0.99, 0.999])
    cfg.train.n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 20])
    cfg.train.net_arch = trial.suggest_categorical("net_arch", ["64,64", "128,128", "256,256"])
    cfg.reward.flat_bonus = trial.suggest_float("flat_bonus", 0.0, 0.03)  # small only (high => degenerate flat agent)
    cfg.env.use_position_features = trial.suggest_categorical("posfeat", [False, True])
    cfg.env.use_regime_features = trial.suggest_categorical("regime", [False, True])
    cfg.env.use_alphatrend_features = trial.suggest_categorical("alpha", [False, True])

    dev = load_dev_ticker(TICKER, Lockbox.load_or_build())
    n, window = len(dev), cfg.env.window
    sharpes, beats, pnls, excess, trades = [], [], [], [], []
    for i in SEARCH_FOLDS:
        tr_end, te_end = fold_bounds(n, K, window, i)
        df_tr = dev.iloc[:tr_end].reset_index(drop=True)
        df_te = dev.iloc[tr_end:te_end].reset_index(drop=True)
        if len(df_te) <= window + 2:
            continue
        model = train(df_tr, cfg)                  # test-eval only (fast)
        res = evaluate(model, df_te, cfg, f"f{i}")
        sharpes.append(res.sharpe)
        beats.append(int(res.raw_pnl > res.buy_hold_pnl))
        pnls.append(res.raw_pnl)
        excess.append(res.raw_pnl - res.buy_hold_pnl)   # excess over buy-and-hold
        trades.append(res.n_trades)
    if not excess:
        return -1e6
    trial.set_user_attr("beat_bh", float(np.mean(beats)))
    trial.set_user_attr("mean_sharpe", float(np.mean(sharpes)))
    trial.set_user_attr("mean_pnl", float(np.mean(pnls)))
    trial.set_user_attr("mean_trades", float(np.mean(trades)))
    # OBJECTIVE: mean excess P&L over buy-and-hold (punishes the do-nothing agent).
    return float(np.mean(excess))


def make_study():
    import time

    import optuna
    from optuna.samplers import TPESampler
    # Longer SQLite lock timeout so many parallel workers don't error out.
    storage = optuna.storages.RDBStorage(
        url=DB, engine_kwargs={"connect_args": {"timeout": 60}})
    # Unique per-process seed so parallel workers EXPLORE different configs
    # (a shared fixed seed makes every worker sample the identical trial).
    seed = (os.getpid() * 1000003 + int(time.time() * 1000)) % (2 ** 31)
    sampler = TPESampler(seed=seed, n_startup_trials=12)
    return optuna.create_study(
        direction="maximize", study_name=STUDY, storage=storage,
        load_if_exists=True, sampler=sampler)


def report():
    import optuna
    study = make_study()
    trials = [t for t in study.trials if t.value is not None]
    print(f"=== QQQ tuning study: {len(trials)} completed trials ===")
    if not trials:
        print("no completed trials yet.")
        return
    best = study.best_trial
    print(f"\nBEST  excess-over-B&H={best.value:.2f}  Sharpe={best.user_attrs.get('mean_sharpe',0):.3f}"
          f"  beat_B&H={best.user_attrs.get('beat_bh')}  mean_pnl={best.user_attrs.get('mean_pnl',0):.2f}"
          f"  trades={best.user_attrs.get('mean_trades',0):.0f}")
    print("  params:", best.params)
    print("\nTop 8 by EXCESS over buy-and-hold:")
    top = sorted(trials, key=lambda t: t.value, reverse=True)[:8]
    for t in top:
        print(f"  #{t.number:<3} excess={t.value:8.2f}  Sharpe={t.user_attrs.get('mean_sharpe',0):6.3f}"
              f"  beatBH={t.user_attrs.get('beat_bh',0):.2f}  trades={t.user_attrs.get('mean_trades',0):5.0f}  "
              f"lr={t.params.get('lr'):.1e} arch={t.params.get('net_arch')} "
              f"flat={t.params.get('flat_bonus'):.3f} reg={t.params.get('regime')} "
              f"alpha={t.params.get('alpha')} pos={t.params.get('posfeat')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--timeout", type=int, default=480)
    a = ap.parse_args()
    if a.report:
        report()
    elif a.worker:
        make_study().optimize(objective, timeout=a.timeout)
    else:
        report()
