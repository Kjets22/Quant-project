import numpy as np

from regime import (
    realized_volatility,
    regime_label,
    trend_slope,
    vol_percentile,
)


def _close(n, seed):
    rng = np.random.default_rng(seed)
    return 100 + np.cumsum(rng.standard_normal(n))


def test_shapes_and_bounds():
    c = _close(800, 1)
    rv = realized_volatility(c, 32)
    ts = trend_slope(c, 32)
    vp = vol_percentile(rv, 256)
    rl = regime_label(vp, 0.7)
    for a in (rv, ts, vp, rl):
        assert a.shape == c.shape
        assert np.all(np.isfinite(a))
    assert rv.min() >= 0.0
    assert vp.min() >= 0.0 and vp.max() <= 1.0
    assert set(np.unique(rl)).issubset({0.0, 1.0})


def test_trend_slope_sign():
    n = 300
    up = np.linspace(100, 200, n)          # rising
    down = np.linspace(200, 100, n)        # falling
    assert trend_slope(up, 32)[-1] > 0
    assert trend_slope(down, 32)[-1] < 0


def test_all_features_are_causal():
    """Each regime feature at t must be invariant to bars after t."""
    c = _close(700, 4)
    cut = 450
    c2 = c.copy()
    c2[cut:] *= 1.7

    for fn in (lambda x: realized_volatility(x, 32),
               lambda x: trend_slope(x, 32)):
        a, b = fn(c), fn(c2)
        assert np.allclose(a[:cut], b[:cut]), "feature leaked future data"

    # vol_percentile depends on realized vol, which is causal -> check end to end.
    rvp = vol_percentile(realized_volatility(c, 32), 256)
    rvp2 = vol_percentile(realized_volatility(c2, 32), 256)
    assert np.allclose(rvp[:cut], rvp2[:cut])


def test_vol_percentile_monotone_intuition():
    # A vol spike at the end should rank near the top.
    rv = np.concatenate([np.full(300, 0.01), [0.05]])
    vp = vol_percentile(rv, 256)
    assert vp[-1] > 0.9
