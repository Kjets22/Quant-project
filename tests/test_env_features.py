import numpy as np

from config import default_config
from data import make_synthetic
from environment import (
    N_ALPHATREND_FEATURES,
    N_REGIME_FEATURES,
    CaptureTradingEnv,
)


def _cfg(**flags):
    cfg = default_config()
    for k, v in flags.items():
        setattr(cfg.env, k, v)
    return cfg


def _actions(i):
    return [2, 2, 1, 0, 2, 0, 0, 2, 1, 0][i % 10]


def test_obs_dim_growth():
    df = make_synthetic(800, seed=7)
    base = CaptureTradingEnv(df, _cfg()).observation_space.shape[0]
    a = CaptureTradingEnv(df, _cfg(use_alphatrend_features=True)).observation_space.shape[0]
    r = CaptureTradingEnv(df, _cfg(use_regime_features=True)).observation_space.shape[0]
    both = CaptureTradingEnv(
        df, _cfg(use_alphatrend_features=True, use_regime_features=True)
    ).observation_space.shape[0]
    assert a - base == N_ALPHATREND_FEATURES == 3
    assert r - base == N_REGIME_FEATURES == 4
    assert both - base == 7


def test_backward_compatible_base_obs():
    df = make_synthetic(800, seed=7)
    off = CaptureTradingEnv(df, _cfg())
    on = CaptureTradingEnv(df, _cfg(use_alphatrend_features=True, use_regime_features=True))
    o_off, _ = off.reset(seed=0)
    o_on, _ = on.reset(seed=0)
    base = o_off.shape[0]
    assert np.array_equal(o_off, o_on[:base])
    for i in range(80):
        a = _actions(i)
        o_off, _, t1, k1, _ = off.step(a)
        o_on, _, t2, k2, _ = on.step(a)
        assert np.array_equal(o_off, o_on[:base])
        if t1 or k1:
            break


def test_no_nan_with_blocks_on():
    df = make_synthetic(900, seed=3)
    env = CaptureTradingEnv(df, _cfg(use_alphatrend_features=True, use_regime_features=True))
    obs, _ = env.reset(seed=1)
    assert np.all(np.isfinite(obs))
    done = False
    i = 0
    while not done:
        obs, r, term, trunc, _ = env.step(_actions(i))
        assert np.all(np.isfinite(obs))
        i += 1
        done = term or trunc


def test_new_features_are_causal():
    """Scrambling FUTURE bars must not change the obs at the current bar."""
    df = make_synthetic(700, seed=5)
    cut = 450
    df2 = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        df2.loc[cut:, col] = df2.loc[cut:, col] * 1.4

    cfg = _cfg(use_alphatrend_features=True, use_regime_features=True)
    e1 = CaptureTradingEnv(df, cfg)
    e2 = CaptureTradingEnv(df2, cfg)
    o1, _ = e1.reset(seed=0)
    o2, _ = e2.reset(seed=0)
    assert np.array_equal(o1, o2)
    for i in range(cut - e1.t - 2):
        a = _actions(i)
        o1, _, *_ = e1.step(a)
        o2, _, *_ = e2.step(a)
        if e1.t >= cut:
            break
        assert np.array_equal(o1, o2), f"future leak at t={e1.t}"
