"""
alphatrend.py — AlphaTrend indicator + MFI/RSI (past-only, lookahead-safe).

Port of KivancOzbilgic's AlphaTrend (TradingView, MPL-2.0) for use as an ML
OBSERVATION feature. AlphaTrend is a volatility-scaled trailing line driven by a
momentum gate (MFI when volume is available, else RSI):

  ATR   = SMA(TrueRange, period)              # simple MA, as in the original
  upT   = low  - ATR*coeff                     # trailing support  (bullish)
  downT = high + ATR*coeff                     # trailing resistance (bearish)
  if momentum>=50:  AT = max(prev_AT, upT)     # ratchets up, never loosens
  else:             AT = min(prev_AT, downT)   # ratchets down

Everything is CAUSAL: AT[t] uses only bars <= t plus its own previous value, so
it satisfies the lookahead wall and may enter the observation. Its key value as
a feature is the MFI (volume) channel, which the price-only features don't carry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import true_range


def money_flow_index(high, low, close, volume, period: int = 14) -> np.ndarray:
    """Money Flow Index in [0, 100], past-only (rolling sums over `period`)."""
    high = np.asarray(high, float); low = np.asarray(low, float)
    close = np.asarray(close, float); volume = np.asarray(volume, float)
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    dtp = np.diff(tp, prepend=tp[0])
    pos = np.where(dtp > 0, rmf, 0.0)
    neg = np.where(dtp < 0, rmf, 0.0)
    pos_s = pd.Series(pos).rolling(period, min_periods=1).sum().to_numpy()
    neg_s = pd.Series(neg).rolling(period, min_periods=1).sum().to_numpy()
    ratio = pos_s / np.maximum(neg_s, 1e-12)
    return np.where(neg_s <= 1e-12, 100.0, 100.0 - 100.0 / (1.0 + ratio))


def rsi(close, period: int = 14) -> np.ndarray:
    """Wilder RSI in [0, 100], past-only."""
    close = np.asarray(close, float)
    d = np.diff(close, prepend=close[0])
    gain = np.where(d > 0, d, 0.0)
    loss = np.where(d < 0, -d, 0.0)
    ag = pd.Series(gain).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
    al = pd.Series(loss).ewm(alpha=1.0 / period, adjust=False).mean().to_numpy()
    rs = ag / np.maximum(al, 1e-12)
    return np.where(al <= 1e-12, 100.0, 100.0 - 100.0 / (1.0 + rs))


def alphatrend(high, low, close, volume, period: int = 14, coeff: float = 1.0,
               novolume: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (alphatrend_line, atr, momentum) — all causal arrays.

    `momentum` is the MFI (or RSI) series used for the gate, returned so callers
    can feed the raw volume-momentum value as its own feature.
    """
    high = np.asarray(high, float); low = np.asarray(low, float)
    close = np.asarray(close, float); volume = np.asarray(volume, float)
    tr = true_range(high, low, close)
    atr = pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()  # SMA of TR
    upT = low - atr * coeff
    downT = high + atr * coeff
    mom = rsi(close, period) if novolume else money_flow_index(high, low, close, volume, period)
    bullish = mom >= 50.0

    n = len(close)
    at = np.empty(n, dtype=np.float64)
    prev = float(close[0])  # robust seed (Pine seeds 0; we start on price to avoid a warmup spike)
    for i in range(n):
        if bullish[i]:
            cur = prev if upT[i] < prev else upT[i]      # max(prev, upT)
        else:
            cur = prev if downT[i] > prev else downT[i]  # min(prev, downT)
        at[i] = cur
        prev = cur
    return at, atr, mom


def crossover(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Boolean array: a crosses above b at index i (a[i]>b[i] and a[i-1]<=b[i-1])."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = np.zeros(len(a), dtype=bool)
    out[1:] = (a[1:] > b[1:]) & (a[:-1] <= b[:-1])
    return out


def crossunder(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = np.zeros(len(a), dtype=bool)
    out[1:] = (a[1:] < b[1:]) & (a[:-1] >= b[:-1])
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 600
    close = 100 + np.cumsum(rng.standard_normal(n))
    high = close + np.abs(rng.standard_normal(n))
    low = close - np.abs(rng.standard_normal(n))
    vol = 1e6 * (1 + rng.random(n))
    at, atr, mfi = alphatrend(high, low, close, vol, period=14, coeff=1.0)
    print("=== alphatrend.py self-test ===")
    print("AT finite:", np.all(np.isfinite(at)), "| ATR>0:", np.all(atr > 0))
    print("MFI in [0,100]:", float(mfi.min()), float(mfi.max()))
    at2 = at[2:]; lag = at[:-2]
    print("buy signals:", int(crossover(at[2:], at[:-2]).sum()),
          "sell signals:", int(crossunder(at[2:], at[:-2]).sum()))
    assert np.all(np.isfinite(at)) and np.all((mfi >= 0) & (mfi <= 100))
    print("OK")
