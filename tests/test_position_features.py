import copy

import numpy as np

from config import default_config
from data import make_synthetic
from environment import N_POSITION_FEATURES, CaptureTradingEnv


def _cfg(use_pos):
    cfg = default_config()
    cfg.env.use_position_features = use_pos
    return cfg


def _action_cycle(i):
    # Deterministic pattern that creates entries, holds, flips and exits.
    return [2, 2, 2, 1, 0, 0, 2, 0, 0, 0][i % 10]


def test_obs_dim_grows_by_expected_amount():
    df = make_synthetic(600, seed=7)
    base = CaptureTradingEnv(df, _cfg(False)).observation_space.shape[0]
    grown = CaptureTradingEnv(df, _cfg(True)).observation_space.shape[0]
    assert grown - base == N_POSITION_FEATURES == 9


def test_backward_compatible_base_observation():
    """Flag OFF == byte-for-byte old obs; flag ON only APPENDS the block."""
    df = make_synthetic(600, seed=7)
    env_off = CaptureTradingEnv(df, _cfg(False))
    env_on = CaptureTradingEnv(df, _cfg(True))
    o_off, _ = env_off.reset(seed=0)
    o_on, _ = env_on.reset(seed=0)
    base = o_off.shape[0]
    assert np.array_equal(o_off, o_on[:base])
    for i in range(120):
        a = _action_cycle(i)
        o_off, _, t1, k1, _ = env_off.step(a)
        o_on, _, t2, k2, _ = env_on.step(a)
        assert np.array_equal(o_off, o_on[:base])  # base portion unchanged
        if t1 or k1:
            break


def test_no_nan_inf_with_position_features():
    df = make_synthetic(800, seed=3)
    env = CaptureTradingEnv(df, _cfg(True))
    obs, _ = env.reset(seed=1)
    assert np.all(np.isfinite(obs))
    done = False
    i = 0
    while not done:
        obs, r, term, trunc, _ = env.step(_action_cycle(i))
        assert np.all(np.isfinite(obs))
        assert np.isfinite(r)
        i += 1
        done = term or trunc


def test_position_features_are_causal():
    """Obs at bar t must not change when FUTURE bars are scrambled."""
    df = make_synthetic(700, seed=5)
    cut = 450
    df2 = df.copy()
    for col in ("open", "high", "low", "close"):
        df2.loc[cut:, col] = df2.loc[cut:, col] * 1.5  # keeps high>=low

    env1 = CaptureTradingEnv(df, _cfg(True))
    env2 = CaptureTradingEnv(df2, _cfg(True))
    o1, _ = env1.reset(seed=0)
    o2, _ = env2.reset(seed=0)
    assert np.array_equal(o1, o2)
    # Step only while the current bar index stays below the scramble point.
    for i in range(cut - env1.t - 2):
        a = _action_cycle(i)
        o1, _, *_ = env1.step(a)
        o2, _, *_ = env2.step(a)
        assert env1.t == env2.t
        if env1.t >= cut:
            break
        assert np.array_equal(o1, o2), f"obs diverged at t={env1.t} (future leak!)"


def test_trade_state_resets_on_flat():
    df = make_synthetic(400, seed=2)
    env = CaptureTradingEnv(df, _cfg(True))
    env.reset(seed=0)
    # Go long for a few bars.
    for _ in range(5):
        env.step(2)
    assert env.position == 1
    assert env.bars_in_trade >= 1
    assert env.entry_price > 0
    # Now flat -> all per-trade state cleared, block is the flat one-hot only.
    obs, *_ = env.step(1)
    assert env.position == 0
    assert env.bars_in_trade == 0
    assert env.entry_price == 0.0
    block = obs[-N_POSITION_FEATURES:]
    # one-hot flat = [0,1,0] then six zeros
    assert np.array_equal(block, np.array([0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32))
