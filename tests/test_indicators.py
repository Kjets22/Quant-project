import numpy as np

from indicators import atr_wilder, true_range


def test_true_range_values():
    high = np.array([10, 11, 12, 11, 13], dtype=float)
    low = np.array([9, 9.5, 10, 10, 11], dtype=float)
    close = np.array([9.5, 10.5, 11, 10.5, 12], dtype=float)
    tr = true_range(high, low, close)
    assert abs(tr[0] - 1.0) < 1e-9      # high-low (no prior close)
    assert abs(tr[1] - 1.5) < 1e-9      # 11-9.5
    assert abs(tr[2] - 2.0) < 1e-9      # 12-10


def test_atr_positive_and_no_nan():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.standard_normal(500))
    high = close + np.abs(rng.standard_normal(500))
    low = close - np.abs(rng.standard_normal(500))
    atr = atr_wilder(high, low, close, period=14)
    assert atr.shape == close.shape
    assert np.all(atr > 0)
    assert not np.any(np.isnan(atr))


def test_atr_is_causal():
    """ATR[t] must not change when bars AFTER t are altered."""
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.standard_normal(300))
    high = close + np.abs(rng.standard_normal(300))
    low = close - np.abs(rng.standard_normal(300))
    cut = 200
    atr_full = atr_wilder(high, low, close, period=14)

    high2, low2, close2 = high.copy(), low.copy(), close.copy()
    high2[cut:] *= 3.0
    low2[cut:] *= 0.3
    close2[cut:] *= 2.0
    atr_scrambled = atr_wilder(high2, low2, close2, period=14)

    assert np.allclose(atr_full[:cut], atr_scrambled[:cut])
