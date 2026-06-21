"""
environment.py — Part 4: the Gymnasium trading environment.

================================ LOOKAHEAD WALL ================================
The observation returned to the agent is built EXCLUSIVELY from data at or before
the current bar t: past log-returns and rolling features computed with past-only
windows, plus the agent's own current position. The oracle quantities
(swings / leg_range) are future-derived and are passed ONLY to CaptureReward to
compute the scalar reward. They never enter `_get_obs()`. See the assertion in
`_get_obs` and the test in tests/test_environment.py.
===============================================================================
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from config import Config, default_config
from reward import CaptureReward
from swings import build_leg_ranges


class CaptureTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.window = cfg.env.window
        self.df = df.reset_index(drop=True)
        self.n = len(self.df)
        if self.n <= self.window + 2:
            raise ValueError("DataFrame too short for the configured window.")

        self.close_px = self.df["close"].to_numpy(dtype=np.float64)
        self.high = self.df["high"].to_numpy(dtype=np.float64)
        self.low = self.df["low"].to_numpy(dtype=np.float64)

        # --- PAST-ONLY observation features (lookahead-safe) -----------------
        # log returns
        logret = np.zeros(self.n, dtype=np.float64)
        logret[1:] = np.log(self.close_px[1:] / self.close_px[:-1])
        self.logret = np.nan_to_num(logret, nan=0.0, posinf=0.0, neginf=0.0)

        s = pd.Series(self.logret)
        # momentum: rolling mean of past returns; volatility: rolling std.
        self.momentum = s.rolling(self.window, min_periods=1).mean().to_numpy()
        self.volatility = s.rolling(self.window, min_periods=1).std().fillna(0.0).to_numpy()
        # range-position: where close sits within its past window hi/lo (0..1).
        roll_hi = self.df["high"].rolling(self.window, min_periods=1).max().to_numpy()
        roll_lo = self.df["low"].rolling(self.window, min_periods=1).min().to_numpy()
        span = np.maximum(roll_hi - roll_lo, 1e-9)
        self.range_pos = (self.close_px - roll_lo) / span

        # --- ORACLE (reward only — behind the lookahead wall) ----------------
        self.leg_range = build_leg_ranges(self.df, cfg)
        self.reward_fn = CaptureReward(self.close_px, self.leg_range, cfg)

        # --- Spaces ----------------------------------------------------------
        # Action: 0 -> short(-1) (or flat if shorting disabled), 1 -> flat, 2 -> long(+1)
        self.action_space = spaces.Discrete(3)
        obs_dim = self.window + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.t = 0
        self.position = 0
        self.prev_position = 0
        self.equity = 0.0

    # ----------------------------------------------------------------------- #
    def _action_to_position(self, action: int) -> int:
        if action == 0:
            return -1 if self.cfg.reward.allow_short else 0
        if action == 2:
            return 1
        return 0

    def _get_obs(self) -> np.ndarray:
        """
        Build the observation from PAST/CURRENT data only.

        LOOKAHEAD ASSERTION: this method references self.logret, self.momentum,
        self.volatility, self.range_pos and self.position — none of which are
        derived from self.leg_range / future swings. Do not add oracle features.
        """
        t = self.t
        window_ret = self.logret[t - self.window + 1: t + 1]
        # scale returns so they are O(1) for the network
        scaled = window_ret * 100.0
        feats = np.array(
            [
                self.momentum[t] * 100.0,
                self.volatility[t] * 100.0,
                self.range_pos[t],
                float(self.position),
            ],
            dtype=np.float64,
        )
        obs = np.concatenate([scaled, feats]).astype(np.float32)
        # Numerical safety: no NaN/Inf may ever reach the agent.
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        assert obs.shape == self.observation_space.shape
        return obs

    # ----------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window + 1
        self.position = 0
        self.prev_position = 0
        self.equity = 0.0
        info = {"t": self.t, "position": self.position, "equity": self.equity}
        return self._get_obs(), info

    def step(self, action: int):
        self.prev_position = self.position
        self.position = self._action_to_position(int(action))

        reward = self.reward_fn.step_reward(self.t, self.position, self.prev_position)
        # Hard guard: a non-finite reward must never reach the agent.
        assert np.isfinite(reward), "non-finite reward produced"

        # Track raw equity in price units for reporting (not used as reward).
        price_pnl = self.position * (self.close_px[self.t + 1] - self.close_px[self.t])
        traded = abs(self.position - self.prev_position)
        cost = traded * self.cfg.reward.txn_cost_frac * self.close_px[self.t]
        self.equity += price_pnl - cost

        self.t += 1
        # Need t+1 for the next reward; truncate one bar early.
        truncated = self.t >= self.n - 1
        terminated = False

        info = {
            "t": self.t,
            "position": self.position,
            "prev_position": self.prev_position,
            "equity": self.equity,
        }
        obs = self._get_obs()
        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    from data import make_synthetic

    cfg = default_config()
    df = make_synthetic(2000, seed=cfg.data.synthetic_seed)
    env = CaptureTradingEnv(df, cfg)

    print("=== environment.py self-test ===")
    print("obs space :", env.observation_space.shape)
    print("act space :", env.action_space)

    # Random agent over a full episode -> should net negative (pays costs).
    obs, info = env.reset(seed=cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    total = 0.0
    done = False
    steps = 0
    while not done:
        a = int(rng.integers(0, 3))
        obs, r, term, trunc, info = env.step(a)
        total += r
        steps += 1
        done = term or trunc
    print(f"random agent: {steps} steps, total reward = {total:.3f}")
    print("OK — random agent ran an episode")
