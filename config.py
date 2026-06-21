"""
config.py — Part 0: all tunable parameters as grouped dataclasses.

Nothing here reads the network or touches global state. The API key is read
from the POLYGON_API_KEY environment variable at runtime (see `Config.api_key`),
never hardcoded. If the key is absent and `DataConfig.use_synthetic_if_no_key`
is True, the whole pipeline falls back to synthetic data and still self-tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv() -> None:
    """
    Minimal .env loader (no external dependency). Reads KEY=VALUE lines from a
    .env file next to this module and populates os.environ for keys that are not
    already set. This makes the API key available regardless of how the shell
    was launched (the Windows equivalent of `export` in a shell profile).
    """
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


@dataclass
class DataConfig:
    ticker: str = "AAPL"
    multiplier: int = 5
    timespan: str = "minute"          # Polygon aggregate bar size unit
    start_date: str = "2023-01-01"
    end_date: str = "2023-06-30"
    use_synthetic_if_no_key: bool = True
    synthetic_n: int = 4000           # default synthetic series length
    synthetic_seed: int = 7


@dataclass
class SwingConfig:
    # A swing low at index i is confirmed if low[i] is the local min over the
    # prior `confirm` candles AND is not beaten for the next `confirm` candles.
    # Symmetric for swing highs. Larger `confirm` => fewer, more significant swings.
    confirm: int = 5
    # Floor on the oracle leg range expressed as a fraction of price. Prevents
    # tiny choppy legs from making the normalized reward explode.
    min_leg_range_frac: float = 0.001


@dataclass
class RewardConfig:
    flat_bonus: float = 0.001         # small positive reward for sitting flat
    txn_cost_frac: float = 0.0002     # transaction cost per unit traded, frac of price
    reward_clip: float = 3.0          # final reward clipped to [-clip, +clip]
    allow_short: bool = True


@dataclass
class EnvConfig:
    window: int = 32                  # number of past log-returns in the observation
    # Enhancement 1: position-awareness state block (past-only, behind this flag).
    # When False the observation is byte-for-byte identical to the Phase-A control.
    use_position_features: bool = False
    atr_period: int = 14              # Wilder ATR period used to normalize distances


@dataclass
class TrainConfig:
    # NOTE: 50k is a smoke-test budget. Raise to 500k+ for any real run.
    total_timesteps: int = 50_000
    n_steps: int = 2048
    learning_rate: float = 3e-4
    gamma: float = 0.99
    seed: int = 42
    train_frac: float = 0.7
    model_path: str = "capture_ppo.zip"
    # Throughput controls (Phase A scaling):
    #   n_envs > 1  -> SubprocVecEnv parallel rollouts (uses multiple CPU cores)
    #   device      -> "auto" | "cpu" | "cuda" for the PPO policy network
    n_envs: int = 1
    device: str = "auto"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    swing: SwingConfig = field(default_factory=SwingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Optional override; if None we read POLYGON_API_KEY from the environment.
    polygon_api_key: str | None = None

    @property
    def api_key(self) -> str | None:
        """Return the Polygon key from config override or environment, else None."""
        return self.polygon_api_key or os.environ.get("POLYGON_API_KEY")


def default_config() -> Config:
    return Config()


if __name__ == "__main__":
    cfg = default_config()
    print("=== Config self-test ===")
    print("ticker        :", cfg.data.ticker)
    print("bar           :", f"{cfg.data.multiplier} {cfg.data.timespan}")
    print("confirm       :", cfg.swing.confirm)
    print("min_leg_frac  :", cfg.swing.min_leg_range_frac)
    print("reward_clip   :", cfg.reward.reward_clip)
    print("window        :", cfg.env.window)
    print("total_steps   :", cfg.train.total_timesteps)
    print("api_key set?  :", cfg.api_key is not None)
    print("OK")
