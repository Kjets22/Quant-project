"""
trend_test.py — does a simple trend filter (long in uptrend, FLAT in downtrend)
beat buy-and-hold net of cost? (training-free)

Rule: position = 1 (long) if close > SMA(N) else 0 (flat). Held to next bar.
This is classic time-series momentum / trend-following: keep the upside, sit out
the downtrends. Tested on 1h and 1day bars for QQQ and TLT (dev set, OOS-style).
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from lockbox import Lockbox, load_dev_ticker

TXN = 0.0002


def resample(dev, rule):
    g = dev.set_index("timestamp").resample(rule).agg(
        close=("close", "last")).dropna()
    return g["close"].to_numpy(float)


def pnl(close, signal):
    rets = np.diff(close)
    price = close[:-1]
    sig = signal.astype(float)
    gross = float((sig * rets).sum())
    traded = np.abs(np.diff(np.concatenate([[0.0], sig])))
    cost = float((traded * TXN * price).sum())
    return gross - cost, int((np.diff(sig) != 0).sum())


def sharpe(close, signal):
    rets = np.diff(close)
    sig = signal.astype(float)
    pl = sig * rets
    if pl.std() < 1e-9:
        return 0.0
    return float(pl.mean() / pl.std() * np.sqrt(len(pl)))


def analyze(ticker):
    dev = load_dev_ticker(ticker, Lockbox.load_or_build())
    print(f"\n================  {ticker}  ================")
    for rule, name in (("15min", "15m"), ("30min", "30m"), ("60min", "1h"), ("1D", "1day")):
        close = resample(dev, rule)
        n = len(close)
        bh, _ = pnl(close, np.ones(n - 1))
        bh_sh = sharpe(close, np.ones(n - 1))
        print(f"\n  --- {name} ({n} bars) ---   Buy&Hold P&L={bh:8.2f}  Sharpe={bh_sh:+.2f}")
        s = pd.Series(close)
        for N in (20, 50, 100, 200):
            sma = s.rolling(N).mean().shift(1)          # causal (uses only past)
            sig = (close > sma.to_numpy()).astype(float)[:-1]   # long/flat over next bar
            sig = np.nan_to_num(sig)
            p, tr = pnl(close, sig)
            sh = sharpe(close, sig)
            beat = "  <- BEATS B&H" if p > bh else ""
            print(f"     trend long/flat SMA{N:>3}: P&L={p:8.2f}  Sharpe={sh:+.2f}  "
                  f"trades={tr:>4}{beat}")


if __name__ == "__main__":
    for tk in ("QQQ", "TLT"):
        analyze(tk)
    print("\nIf 'long/flat trend' beats B&H (esp. on TLT) with FEW trades, that is a")
    print("real, robust edge: capture uptrends, sit out downtrends. This is what we")
    print("should make the RL agent learn (daily/hourly, long-or-flat, trend-aware).")
