"""
regime.py — Phase C: causal regime-awareness features (lookahead-safe).

The control's weakness was fold-to-fold inconsistency (some folds strongly
negative). Regime features give the agent a past-only read on "what kind of
market am I in" so it can behave differently in calm vs. volatile/trending
regimes. Every value here uses only bars <= t.

Features:
  * realized_volatility(close, window)   — rolling std of log returns
  * trend_slope(close, window)           — slope of an OLS line fit to log price
                                            over the past `window` bars (per-bar,
                                            vectorized; normalized growth rate)
  * vol_percentile(vol, lookback)        — rank of current vol within the past
                                            `lookback` vols, in [0, 1]
  * regime_label(vol_pct, hi)            — 1.0 if high-risk (vol_pct >= hi) else 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def log_returns(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, float)
    r = np.zeros_like(close)
    r[1:] = np.log(close[1:] / close[:-1])
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def realized_volatility(close: np.ndarray, window: int = 32) -> np.ndarray:
    r = pd.Series(log_returns(close))
    return r.rolling(window, min_periods=2).std().fillna(0.0).to_numpy()


def trend_slope(close: np.ndarray, window: int = 32) -> np.ndarray:
    """
    Per-bar slope of an OLS line fit to log(close) over the trailing `window`
    bars (units: log-price per bar). Vectorized via sliding windows; the head
    (< window bars) is back-filled with the first valid slope. Past-only.
    """
    y = np.log(np.maximum(np.asarray(close, float), 1e-12))
    n = len(y)
    if n < window:
        return np.zeros(n)
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    sxx = float((xc * xc).sum())                 # constant denominator
    sw = sliding_window_view(y, window)          # (n-window+1, window)
    slope_valid = (sw @ xc) / sxx                # OLS slope per window
    out = np.empty(n, dtype=float)
    out[window - 1:] = slope_valid
    out[:window - 1] = slope_valid[0]            # back-fill head
    return out


def vol_percentile(vol: np.ndarray, lookback: int = 256) -> np.ndarray:
    """Rank of current vol within the trailing `lookback` window, in [0,1]. Causal."""
    vol = np.asarray(vol, float)
    n = len(vol)
    if n < lookback:
        # Expanding rank over what we have so far.
        out = np.zeros(n)
        for t in range(n):
            w = vol[: t + 1]
            out[t] = float((w <= vol[t]).mean())
        return out
    sw = sliding_window_view(vol, lookback)       # (n-lookback+1, lookback)
    pct_valid = (sw <= sw[:, -1:]).mean(axis=1)   # fraction <= current
    out = np.empty(n, dtype=float)
    out[lookback - 1:] = pct_valid
    # Head: expanding rank.
    for t in range(lookback - 1):
        w = vol[: t + 1]
        out[t] = float((w <= vol[t]).mean())
    return out


def regime_label(vol_pct: np.ndarray, hi: float = 0.7) -> np.ndarray:
    """1.0 in a high-risk (high-vol) regime, else 0.0. Causal threshold."""
    return (np.asarray(vol_pct, float) >= hi).astype(float)


N_REGIMES = 3  # 0 = up-trend, 1 = down-trend, 2 = chop


def _debounce(reg: np.ndarray, min_run: int) -> np.ndarray:
    """Causal hysteresis: adopt a new regime only after it persists min_run bars."""
    out = reg.copy()
    cur = int(reg[0])
    run_val, run_len = int(reg[0]), 1
    for t in range(1, len(reg)):
        if reg[t] == run_val:
            run_len += 1
        else:
            run_val, run_len = int(reg[t]), 1
        if run_len >= min_run:
            cur = run_val
        out[t] = cur
    return out


def regime_id(close: np.ndarray, slope_window: int = 32,
              thresh_window: int = 512, min_run: int = 1) -> np.ndarray:
    """
    Causal 3-regime classifier from the trend slope:
      0 = up-trend   (slope >  +eps)
      1 = down-trend (slope <  -eps)
      2 = chop       (|slope| <= eps)
    where eps is a PAST-ONLY rolling median of |slope| (adapts per asset). Uses
    only bars <= t, so it is lookahead-safe. `min_run` applies causal hysteresis
    so the regime is sticky (avoids whipsaw from frequent flips).
    """
    slope = trend_slope(close, slope_window)
    abs_slope = np.abs(slope)
    eps = (pd.Series(abs_slope).rolling(thresh_window, min_periods=8)
           .median().bfill().to_numpy())
    out = np.full(len(close), 2, dtype=np.int64)   # default chop
    out[slope > eps] = 0
    out[slope < -eps] = 1
    if min_run > 1:
        out = _debounce(out, min_run)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 1000
    close = 100 + np.cumsum(rng.standard_normal(n))
    rv = realized_volatility(close, 32)
    ts = trend_slope(close, 32)
    vp = vol_percentile(rv, 256)
    rl = regime_label(vp, 0.7)
    print("=== regime.py self-test ===")
    print("realized_vol finite:", np.all(np.isfinite(rv)), "min/max:",
          round(float(rv.min()), 5), round(float(rv.max()), 5))
    print("trend_slope finite:", np.all(np.isfinite(ts)))
    print("vol_pct in [0,1]:", round(float(vp.min()), 3), round(float(vp.max()), 3))
    print("high-risk fraction:", round(float(rl.mean()), 3))
    assert np.all(np.isfinite(rv)) and np.all(np.isfinite(ts))
    assert vp.min() >= 0.0 and vp.max() <= 1.0
    print("OK")
