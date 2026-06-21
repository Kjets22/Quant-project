import warnings

import numpy as np

warnings.filterwarnings("ignore")

from stable_baselines3.common.env_checker import check_env

from config import default_config
from data import make_synthetic
from environment import CaptureTradingEnv


def test_check_env_passes():
    env = CaptureTradingEnv(make_synthetic(800, seed=7), default_config())
    check_env(env)  # raises on any spec violation


def test_random_agent_total_reward_negative():
    cfg = default_config()
    env = CaptureTradingEnv(make_synthetic(2000, seed=7), cfg)
    obs, _ = env.reset(seed=cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    total, done = 0.0, False
    while not done:
        obs, r, term, trunc, _ = env.step(int(rng.integers(0, 3)))
        total += r
        done = term or trunc
    assert total < 0  # reward is not trivially gameable


def test_no_nan_or_inf_in_obs_and_reward():
    cfg = default_config()
    env = CaptureTradingEnv(make_synthetic(1000, seed=2), cfg)
    obs, _ = env.reset(seed=1)
    assert np.all(np.isfinite(obs))
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(1)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(r)
        done = term or trunc


def test_observation_has_no_oracle_leakage():
    """
    Lookahead wall: scrambling leg_range must NOT change the observation, since
    the obs is built only from past/current price features and the position.
    """
    cfg = default_config()
    df = make_synthetic(1000, seed=5)
    env = CaptureTradingEnv(df, cfg)
    obs1, _ = env.reset(seed=0)
    obs_seq1 = [obs1]
    for _ in range(50):
        o, _, term, trunc, _ = env.step(1)
        obs_seq1.append(o)
        if term or trunc:
            break

    # Corrupt the oracle entirely; observations must be identical.
    env2 = CaptureTradingEnv(df, cfg)
    env2.leg_range = env2.leg_range * 1000.0 + 12345.0
    from reward import CaptureReward
    env2.reward_fn = CaptureReward(env2.close_px, env2.leg_range, cfg)
    obs2, _ = env2.reset(seed=0)
    obs_seq2 = [obs2]
    for _ in range(50):
        o, _, term, trunc, _ = env2.step(1)
        obs_seq2.append(o)
        if term or trunc:
            break

    for a, b in zip(obs_seq1, obs_seq2):
        assert np.array_equal(a, b), "observation changed when oracle changed!"
