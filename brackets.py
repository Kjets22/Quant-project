"""
brackets.py — does a 2:1 take-profit/stop-loss bracket make money? (SPY, QQQ)

Triple-barrier test on real daily candles: from each entry, set TP = 2x the SL
distance (2:1), a time barrier, and walk intraday highs/lows to see which is hit
first (ties -> SL first, conservative). Break-even win rate at 2:1 is 33.3%.

We compare entry signals:
  * all-long  : a long bracket every day (captures market DRIFT = beta)
  * random    : random long/short (no edge -> should sit at ~33% = break-even)
  * trend     : long only when above SMA200 (the one thing that 'worked')
If a signal can't push the win rate meaningfully above 33% out-of-sample, the
bracket adds no edge -- and layering options on a ~0 edge only adds premium cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

SL_VOL = 1.5     # stop  = 1.5 * daily vol
TP_VOL = 3.0     # target= 3.0 * daily vol  (=> 2:1)
MAX_H = 20       # time barrier (trading days)
COST = 0.0002    # round-trip-ish underlying cost (bps), charged per trade


def daily(ticker):
    p = Path(f"data_cache/{ticker}_5minute_2021-06-01_2026-06-01.csv")
    df = pd.read_csv(p, parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample("1D").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna()
    return g.reset_index()


def triple_barrier(high, low, close, entries, side):
    """side: +1 long / -1 short. Returns win, loss, timeout counts and total R."""
    r = np.diff(np.log(close))
    vol = pd.Series(r).rolling(21).std().reindex(range(len(close))).ffill().to_numpy()
    win = loss = to = 0
    R = 0.0
    for i in entries:
        v = vol[i]
        if not np.isfinite(v) or v <= 0:
            continue
        P = close[i]
        sl_d, tp_d = SL_VOL * v * P, TP_VOL * v * P
        if side > 0:
            sl, tp = P - sl_d, P + tp_d
        else:
            sl, tp = P + sl_d, P - tp_d
        outcome = 0
        for j in range(i + 1, min(i + MAX_H + 1, len(close))):
            hit_sl = (low[j] <= sl) if side > 0 else (high[j] >= sl)
            hit_tp = (high[j] >= tp) if side > 0 else (low[j] <= tp)
            if hit_sl:                      # ties: stop first (conservative)
                outcome = -1
                break
            if hit_tp:
                outcome = +1
                break
        if outcome == 1:
            win += 1
            R += TP_VOL / SL_VOL            # +2 R
        elif outcome == -1:
            loss += 1
            R += -1.0
        else:
            j = min(i + MAX_H, len(close) - 1)
            to += 1
            R += side * (close[j] - P) / sl_d
        R -= COST * P / sl_d                # cost in R units
    n = win + loss + to
    return n, win, loss, to, R


def run(ticker):
    d = daily(ticker)
    high, low, close = (d[c].to_numpy() for c in ("high", "low", "close"))
    n = len(close)
    sma200 = pd.Series(close).rolling(200).mean().shift(1).to_numpy()
    rng = np.random.default_rng(0)
    valid = np.arange(220, n - 1)

    sigs = {
        "all-long  (side+1)": (valid, +1),
        "trend long>SMA200 ": (valid[close[valid] > sma200[valid]], +1),
        "random long       ": (rng.choice(valid, size=len(valid) // 2, replace=False), +1),
        "anti-trend short  ": (valid[close[valid] > sma200[valid]], -1),
    }
    print(f"\n===== {ticker} — 2:1 bracket (TP {TP_VOL} / SL {SL_VOL} vol, {MAX_H}d) =====")
    print(f"  break-even win rate at 2:1 = 33.3%")
    print(f"  {'signal':20} {'trades':>7} {'win%':>6} {'exp(R/trade)':>13} {'verdict':>9}")
    for name, (entries, side) in sigs.items():
        ntot, win, loss, to, R = triple_barrier(high, low, close, entries, side)
        if ntot == 0:
            continue
        wr = win / ntot
        exp = R / ntot
        verdict = "edge?" if (wr > 0.36 and exp > 0.03) else "no edge"
        print(f"  {name:20} {ntot:>7} {wr:>6.1%} {exp:>13.3f} {verdict:>9}")
    print("  (win% near 33% => the bracket is break-even; >>33% only from market DRIFT,")
    print("   which is just B&H beta -- and options would give that drift back as premium.)")


if __name__ == "__main__":
    for tk in ("SPY", "QQQ"):
        run(tk)


