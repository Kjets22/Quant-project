"""
diagnose_tf.py — is there more SIGNAL at longer timeframes? (training-free)

Resamples the cached 5-min data to 5min / 1h / 1day and, for each, reports return
predictability (OOS linear R^2 + direction accuracy) and whether simple momentum
beats buy-and-hold NET OF COST. This tells us whether a longer bar is worth RL
training BEFORE we spend the compute.
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


def resample(dev: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = dev.set_index("timestamp").resample(rule).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def predictability(close: np.ndarray):
    r = np.diff(np.log(close))
    if len(r) < 100:
        return None
    ac1 = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    L = 10
    X = np.stack([r[i:len(r) - L + i] for i in range(L)], axis=1)
    y = r[L:]
    cut = int(len(y) * 0.7)
    A = np.hstack([X[:cut], np.ones((cut, 1))])
    coef, *_ = np.linalg.lstsq(A, y[:cut], rcond=None)
    pred = np.hstack([X[cut:], np.ones((len(y) - cut, 1))]) @ coef
    yte = y[cut:]
    r2 = 1 - float(((yte - pred) ** 2).sum()) / float(((yte - yte.mean()) ** 2).sum())
    acc = float((np.sign(pred) == np.sign(yte)).mean())
    return ac1, r2, acc, r.std() * 1e4


def strat_pnl(close, signal):
    """signal[i] = position held over bar i->i+1 (length len(close)-1)."""
    rets = np.diff(close)
    price = close[:-1]
    sig = signal.astype(float)
    gross = float((sig * rets).sum())
    traded = np.abs(np.diff(np.concatenate([[0.0], sig])))
    cost = float((traded * TXN * price).sum())
    return gross - cost, int((np.diff(sig) != 0).sum())


def analyze(ticker: str) -> None:
    dev = load_dev_ticker(ticker, Lockbox.load_or_build())
    print(f"\n================  {ticker}  ================")
    print(f"{'tf':>6} {'bars':>7} {'std(bps)':>9} {'ac1':>8} {'OOS R^2':>9} "
          f"{'dir acc':>8} | {'B&H':>9} {'Mom1':>9} {'Mom(k)':>9} {'flat':>6}")
    for rule, name, k in (("5min", "5min", 12), ("60min", "1h", 6), ("1D", "1day", 5)):
        g = resample(dev, rule)
        close = g["close"].to_numpy(float)
        pr = predictability(close)
        if pr is None:
            continue
        ac1, r2, acc, stdbps = pr
        rets = np.diff(close)                       # length n-1
        bh, _ = strat_pnl(close, np.ones(len(rets)))
        mom1, _ = strat_pnl(close, np.concatenate([[0.0], np.sign(rets[:-1])]))
        roll = pd.Series(rets).rolling(k).sum().shift(1)
        momk, _ = strat_pnl(close, np.sign(roll).fillna(0.0).to_numpy())
        flat = 0.0
        beat = "  <-momentum beats B&H!" if max(mom1, momk) > bh else ""
        print(f"{name:>6} {len(close):>7} {stdbps:>9.1f} {ac1:>+8.3f} {r2:>+9.5f} "
              f"{acc:>7.1%} | {bh:>9.2f} {mom1:>9.2f} {momk:>9.2f} {flat:>6.1f}{beat}")


if __name__ == "__main__":
    for tk in ("QQQ", "TLT"):
        analyze(tk)
    print("\nIf OOS R^2 / direction accuracy rise at 1h/1day and momentum starts")
    print("beating B&H net of cost, a longer-bar RL agent has a real target.")
