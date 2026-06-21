"""
agent.py — Part 5: PPO training and evaluation/metrics.

Chronological split only (never shuffle time). Train on the earlier slice,
evaluate deterministically on a later slice, and report capture reward alongside
raw P&L, buy & hold, Sharpe, max drawdown, trade count and time-in-market.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from config import Config, default_config
from environment import CaptureTradingEnv


def split_data(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split — earliest data trains, latest tests."""
    n = len(df)
    cut = int(n * train_frac)
    train = df.iloc[:cut].reset_index(drop=True)
    test = df.iloc[cut:].reset_index(drop=True)
    return train, test


class _EnvMaker:
    """Picklable env factory so SubprocVecEnv works under Windows 'spawn'."""

    def __init__(self, df: pd.DataFrame, cfg: Config):
        self.df = df
        self.cfg = cfg

    def __call__(self) -> CaptureTradingEnv:
        return CaptureTradingEnv(self.df, self.cfg)


def _build_vecenv(df_train: pd.DataFrame, cfg: Config):
    """DummyVecEnv for n_envs<=1 (safe/serial); SubprocVecEnv for parallel rollouts."""
    n_envs = max(1, int(cfg.train.n_envs))
    makers = [_EnvMaker(df_train, cfg) for _ in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(makers)
    # n_steps is per-env in SB3; scale so total rollout size stays ~constant.
    return SubprocVecEnv(makers, start_method="spawn")


def train(df_train: pd.DataFrame, cfg: Config, verbose: int = 0) -> PPO:
    """Build and train a PPO agent on the capture-trading env; save the model."""
    tc = cfg.train
    n_envs = max(1, int(tc.n_envs))
    venv = _build_vecenv(df_train, cfg)
    # Keep the total rollout (n_steps * n_envs) near the configured n_steps so the
    # update batch size is comparable regardless of how many envs we parallelize.
    per_env_steps = max(256, tc.n_steps // n_envs)
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=per_env_steps,
        learning_rate=tc.learning_rate,
        gamma=tc.gamma,
        seed=tc.seed,
        device=tc.device,
        verbose=verbose,
    )
    model.learn(total_timesteps=tc.total_timesteps)
    model.save(tc.model_path)
    try:
        venv.close()
    except Exception:  # noqa: BLE001
        pass
    return model


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    label: str
    capture_reward: float
    raw_pnl: float
    buy_hold_pnl: float
    sharpe: float
    max_drawdown: float
    n_trades: int
    pct_in_market: float
    n_steps: int

    def block(self) -> str:
        return (
            f"--- {self.label} ---\n"
            f"  capture reward : {self.capture_reward:8.3f}\n"
            f"  raw P&L (px)   : {self.raw_pnl:8.3f}\n"
            f"  buy & hold P&L : {self.buy_hold_pnl:8.3f}\n"
            f"  Sharpe         : {self.sharpe:8.3f}\n"
            f"  max drawdown   : {self.max_drawdown:8.3f}\n"
            f"  trades         : {self.n_trades:8d}\n"
            f"  % in market    : {self.pct_in_market:8.1%}\n"
            f"  steps          : {self.n_steps:8d}"
        )


def _sharpe(step_pnls: np.ndarray) -> float:
    if step_pnls.size < 2:
        return 0.0
    sd = step_pnls.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(step_pnls.mean() / sd * np.sqrt(len(step_pnls)))


def _max_drawdown(equity_curve: np.ndarray) -> float:
    if equity_curve.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = equity_curve - running_max
    return float(-drawdowns.min())  # reported as a positive magnitude


def evaluate(model: PPO, df_eval: pd.DataFrame, cfg: Config, label: str) -> EvalResult:
    """Deterministic rollout of `model` over `df_eval`, returning metrics."""
    env = CaptureTradingEnv(df_eval, cfg)
    obs, _ = env.reset(seed=cfg.train.seed)

    rewards: list[float] = []
    step_pnls: list[float] = []
    equity_curve: list[float] = []
    in_market = 0
    n_trades = 0
    prev_equity = 0.0
    prev_position = 0
    done = False
    steps = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        rewards.append(reward)
        step_pnls.append(info["equity"] - prev_equity)
        prev_equity = info["equity"]
        equity_curve.append(info["equity"])
        if info["position"] != 0:
            in_market += 1
        if info["position"] != prev_position:
            n_trades += 1
        prev_position = info["position"]
        steps += 1
        done = term or trunc

    close = df_eval["close"].to_numpy()
    buy_hold = float(close[-1] - close[env.window + 1])

    return EvalResult(
        label=label,
        capture_reward=float(np.sum(rewards)),
        raw_pnl=float(prev_equity),
        buy_hold_pnl=buy_hold,
        sharpe=_sharpe(np.asarray(step_pnls)),
        max_drawdown=_max_drawdown(np.asarray(equity_curve)),
        n_trades=n_trades,
        pct_in_market=in_market / max(steps, 1),
        n_steps=steps,
    )


def random_baseline(df_eval: pd.DataFrame, cfg: Config, label: str = "RANDOM") -> EvalResult:
    """A uniformly-random-action baseline for the same slice."""
    env = CaptureTradingEnv(df_eval, cfg)
    obs, _ = env.reset(seed=cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    rewards, step_pnls, equity_curve = [], [], []
    in_market = n_trades = steps = 0
    prev_equity = 0.0
    prev_position = 0
    done = False
    while not done:
        obs, reward, term, trunc, info = env.step(int(rng.integers(0, 3)))
        rewards.append(reward)
        step_pnls.append(info["equity"] - prev_equity)
        prev_equity = info["equity"]
        equity_curve.append(info["equity"])
        if info["position"] != 0:
            in_market += 1
        if info["position"] != prev_position:
            n_trades += 1
        prev_position = info["position"]
        steps += 1
        done = term or trunc
    close = df_eval["close"].to_numpy()
    return EvalResult(
        label=label,
        capture_reward=float(np.sum(rewards)),
        raw_pnl=float(prev_equity),
        buy_hold_pnl=float(close[-1] - close[env.window + 1]),
        sharpe=_sharpe(np.asarray(step_pnls)),
        max_drawdown=_max_drawdown(np.asarray(equity_curve)),
        n_trades=n_trades,
        pct_in_market=in_market / max(steps, 1),
        n_steps=steps,
    )


if __name__ == "__main__":
    from data import make_synthetic

    cfg = default_config()
    cfg.train.total_timesteps = 30_000  # short smoke train
    df = make_synthetic(3000, seed=cfg.data.synthetic_seed)
    df_train, df_test = split_data(df, cfg.train.train_frac)

    print("=== agent.py self-test: short train on synthetic ===")
    rnd = random_baseline(df_test, cfg, "RANDOM (test)")
    print(rnd.block())
    print("\ntraining PPO ...")
    model = train(df_train, cfg, verbose=0)
    res_tr = evaluate(model, df_train, cfg, "TRAIN")
    res_te = evaluate(model, df_test, cfg, "TEST")
    print(res_tr.block())
    print(res_te.block())
    assert res_te.capture_reward > rnd.capture_reward, "agent did not beat random"
    print("\nOK — trained agent beat the random baseline on test")
