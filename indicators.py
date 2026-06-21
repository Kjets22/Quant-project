"""
indicators.py — past-only technical indicators (lookahead-safe).

Convention used throughout this project: a decision is made at the CLOSE of a
completed bar t, observing that bar's full OHLC, and the position is held to the
close of bar t+1 (the reward uses t+1; the observation never does). Under this
"bar-close decision" convention, using high[t]/low[t]/close[t] is past-only — bar
t is fully observed at decision time. ATR[t] therefore depends only on TR[0..t].
"""

from __future__ import annotations

import numpy as np


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Wilder's True Range. TR[0] falls back to high[0]-low[0] (no prior close)."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return tr


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14, floor_frac: float = 1e-6) -> np.ndarray:
    """
    Wilder-smoothed ATR, past-only and strictly positive.

    Seed: ATR over the first `period` bars is the expanding mean of TR (so there
    are no NaNs at the head). From index `period` onward use Wilder recursion
    ATR[i] = (ATR[i-1]*(period-1) + TR[i]) / period. Floored at floor_frac*price
    so it can safely divide.
    """
    period = max(1, int(period))
    tr = true_range(high, low, close)
    n = len(tr)
    atr = np.empty(n, dtype=np.float64)
    if n == 0:
        return atr

    # Expanding mean for the seed region (past-only, no NaN).
    csum = np.cumsum(tr)
    for i in range(min(period, n)):
        atr[i] = csum[i] / (i + 1)
    # Wilder recursion afterwards.
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    floor = floor_frac * np.asarray(close, dtype=np.float64)
    return np.maximum(atr, np.maximum(floor, 1e-9))


if __name__ == "__main__":
    # Tiny hand-checkable example.
    high = np.array([10, 11, 12, 11, 13], dtype=float)
    low = np.array([9, 9.5, 10, 10, 11], dtype=float)
    close = np.array([9.5, 10.5, 11, 10.5, 12], dtype=float)
    tr = true_range(high, low, close)
    atr = atr_wilder(high, low, close, period=3)
    print("=== indicators.py self-test ===")
    print("TR :", np.round(tr, 4))
    print("ATR:", np.round(atr, 4))
    assert np.all(atr > 0)
    assert not np.any(np.isnan(atr))
    # TR[0] = 10-9 = 1 ; TR[1]=max(1.5, |11-9.5|, |9.5-9.5|)=1.5 ; TR[2]=max(2,1.5,0.5)=2
    assert abs(tr[0] - 1.0) < 1e-9 and abs(tr[1] - 1.5) < 1e-9 and abs(tr[2] - 2.0) < 1e-9
    print("OK")
