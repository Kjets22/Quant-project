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
from indicators import atr_wilder
from reward import CaptureReward
from swings import build_leg_ranges

# Number of features in the position-awareness block (Enhancement 1).
N_POSITION_FEATURES = 9


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

        # ATR (past-only) — used to normalize the position-feature distances and,
        # later, to size brackets. Computed always (cheap); used only when enabled.
        self.atr = atr_wilder(self.high, self.low, self.close_px,
                              period=cfg.env.atr_period)

        # --- ORACLE (reward only — behind the lookahead wall) ----------------
        self.leg_range = build_leg_ranges(self.df, cfg)
        self.reward_fn = CaptureReward(self.close_px, self.leg_range, cfg)

        # --- Spaces ----------------------------------------------------------
        # Action: 0 -> short(-1) (or flat if shorting disabled), 1 -> flat, 2 -> long(+1)
        self.action_space = spaces.Discrete(3)
        self.use_position_features = bool(cfg.env.use_position_features)
        obs_dim = self.window + 4 + (N_POSITION_FEATURES if self.use_position_features else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.t = 0
        self.position = 0
        self.prev_position = 0
        self.equity = 0.0
        # Per-trade running state (Enhancement 1). All reset to 0 when flat.
        self.entry_price = 0.0
        self.bars_in_trade = 0
        self.mfe = 0.0   # max favorable excursion since entry (price units, >= 0)
        self.mae = 0.0   # max adverse excursion since entry (price units, <= 0)

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
        self.volatility, self.range_pos, self.position and (optionally) the
        position-feature block — none of which are derived from self.leg_range /
        future swings, and all of which use only bars <= t. Do not add oracle
        features.
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
        parts = [scaled, feats]
        if self.use_position_features:
            parts.append(self._position_features())
        obs = np.concatenate(parts).astype(np.float32)
        # Numerical safety: no NaN/Inf may ever reach the agent.
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        assert obs.shape == self.observation_space.shape
        return obs

    def _position_features(self) -> np.ndarray:
        """
        Position-awareness block (Enhancement 1) — all PAST-ONLY and normalized.

        Every value derives from self.position, self.entry_price, self.bars_in_trade,
        self.mfe/self.mae and self.atr[t] / self.close_px[t]; none of these depend on
        any bar > t (entry/MFE/MAE are updated from bars already observed). When flat,
        the entire block is zeros except the flat one-hot.
        """
        t = self.t
        d = float(self.position)                       # -1 / 0 / +1
        price = self.close_px[t]
        atr = self.atr[t]                              # floored > 0 in atr_wilder

        # position sign one-hot {short, flat, long}
        onehot = np.array([d < 0, d == 0, d > 0], dtype=np.float64)

        if self.position == 0:
            unreal = mfe = mae = give_back = entry_dist = 0.0
            bars = 0.0
        else:
            unreal = d * (price - self.entry_price)
            mfe = self.mfe
            mae = self.mae
            give_back = mfe - unreal                   # open profit surrendered (>= 0)
            entry_dist = d * (price - self.entry_price)  # == unreal by definition
            bars = self.bars_in_trade / 100.0

        block = np.array([
            onehot[0], onehot[1], onehot[2],
            bars,
            unreal / atr,
            mfe / atr,
            mae / atr,
            give_back / atr,
            entry_dist / atr,
        ], dtype=np.float64)
        return block

    def _update_trade_state_on_action(self, new_position: int) -> None:
        """Update entry/excursion bookkeeping when the target position changes."""
        if new_position == 0:
            # Flat: clear everything.
            self.entry_price = 0.0
            self.bars_in_trade = 0
            self.mfe = 0.0
            self.mae = 0.0
        elif new_position != self.position:
            # New entry or flip -> fresh trade, entered at this bar's close.
            self.entry_price = self.close_px[self.t]
            self.bars_in_trade = 0
            self.mfe = 0.0
            self.mae = 0.0
        # else: holding the same nonzero position -> keep entry/excursions.

    def _accrue_excursion(self) -> None:
        """After advancing to the current bar, fold its high/low into MFE/MAE."""
        if self.position == 0:
            return
        self.bars_in_trade += 1
        t = self.t
        if self.position > 0:
            fav_price, adv_price = self.high[t], self.low[t]
        else:
            fav_price, adv_price = self.low[t], self.high[t]
        fav = self.position * (fav_price - self.entry_price)
        adv = self.position * (adv_price - self.entry_price)
        self.mfe = max(self.mfe, fav)
        self.mae = min(self.mae, adv)

    # ----------------------------------------------------------------------- #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window + 1
        self.position = 0
        self.prev_position = 0
        self.equity = 0.0
        self.entry_price = 0.0
        self.bars_in_trade = 0
        self.mfe = 0.0
        self.mae = 0.0
        info = {"t": self.t, "position": self.position, "equity": self.equity}
        return self._get_obs(), info

    def step(self, action: int):
        self.prev_position = self.position
        new_position = self._action_to_position(int(action))
        # Update entry/excursion bookkeeping (compares against the OLD position).
        self._update_trade_state_on_action(new_position)
        self.position = new_position

        reward = self.reward_fn.step_reward(self.t, self.position, self.prev_position)
        # Hard guard: a non-finite reward must never reach the agent.
        assert np.isfinite(reward), "non-finite reward produced"

        # Track raw equity in price units for reporting (not used as reward).
        price_pnl = self.position * (self.close_px[self.t + 1] - self.close_px[self.t])
        traded = abs(self.position - self.prev_position)
        cost = traded * self.cfg.reward.txn_cost_frac * self.close_px[self.t]
        self.equity += price_pnl - cost

        self.t += 1
        # Fold the newly-current bar's high/low into MFE/MAE (past-only).
        self._accrue_excursion()
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
