import numpy as np

from alphatrend import alphatrend, crossover, crossunder, money_flow_index, rsi


def _series(n, seed):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.standard_normal(n))
    high = close + np.abs(rng.standard_normal(n))
    low = close - np.abs(rng.standard_normal(n))
    vol = 1e6 * (1 + rng.random(n))
    return high, low, close, vol


def test_mfi_and_rsi_bounded():
    h, l, c, v = _series(500, 1)
    mfi = money_flow_index(h, l, c, v, 14)
    r = rsi(c, 14)
    assert np.all((mfi >= 0) & (mfi <= 100))
    assert np.all((r >= 0) & (r <= 100))
    assert not np.any(np.isnan(mfi)) and not np.any(np.isnan(r))


def test_alphatrend_finite_and_atr_positive():
    h, l, c, v = _series(500, 2)
    at, atr, mom = alphatrend(h, l, c, v, 14, 1.0)
    assert at.shape == c.shape
    assert np.all(np.isfinite(at))
    assert np.all(atr > 0)


def test_alphatrend_is_causal():
    """AlphaTrend[t] must not change when bars after t are altered (no lookahead)."""
    h, l, c, v = _series(400, 3)
    cut = 250
    at_full, _, _ = alphatrend(h, l, c, v, 14, 1.0)

    h2, l2, c2, v2 = h.copy(), l.copy(), c.copy(), v.copy()
    h2[cut:] *= 3.0; l2[cut:] *= 0.2; c2[cut:] *= 2.0; v2[cut:] *= 5.0
    at_scr, _, _ = alphatrend(h2, l2, c2, v2, 14, 1.0)

    assert np.allclose(at_full[:cut], at_scr[:cut])


def test_alphatrend_ratchets():
    """In a steadily rising bullish series the line should be non-decreasing-ish."""
    n = 200
    c = np.linspace(100, 150, n)
    h = c + 0.5
    l = c - 0.5
    v = np.full(n, 1e6)
    at, _, _ = alphatrend(h, l, c, v, 14, 1.0)
    # Mostly monotone up: very few downticks in a pure uptrend.
    downticks = int((np.diff(at) < -1e-9).sum())
    assert downticks <= n // 10


def test_crossover_helpers():
    a = np.array([1, 2, 3, 2, 1, 2], float)
    b = np.array([2, 2, 2, 2, 2, 2], float)
    assert crossover(a, b)[2]      # a:2->3 crosses above 2 at i=2
    assert crossunder(a, b)[4]     # a:2->1 crosses below 2 at i=4 (a[3]=2 is not < 2)
    assert not crossunder(a, b)[3]
