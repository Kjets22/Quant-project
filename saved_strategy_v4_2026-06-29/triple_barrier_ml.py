"""
triple_barrier_ml.py — can an ML find bracket ENTRIES that beat the break-even
win rate? (the guide's triple-barrier + meta-labeling method, on SPY & QQQ candles)

For each bar: set target = +TP*ATR, stop = -SL*ATR, time barrier H. Label which is
hit first (intraday highs/lows; ties -> stop). Break-even win rate = SL/(SL+TP).
Train LightGBM on CAUSAL features to predict 'target first', then on out-of-sample
data take only the high-confidence entries and measure their actual win rate +
expectancy net of cost. If the ML-selected win rate clears break-even OOS, that's a
real learnable edge. If it sits at break-even, the geometry wins (no edge).
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
import lightgbm as lgb

H = 24            # time barrier (hourly bars ~ 3-4 sessions)
COST_BPS = 1.0    # round-trip underlying cost, bps of price


def hourly(ticker):
    p = Path(f"data_cache/{ticker}_5minute_2021-06-01_2026-06-01.csv")
    df = pd.read_csv(p, parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample("60min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def atr(h, l, c, n=24):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n).mean().bfill().to_numpy()


def features(h, l, c, v):
    s = pd.Series(c)
    r = np.diff(np.log(c), prepend=np.log(c[0]))
    f = pd.DataFrame({
        "ret1": r,
        "ret6": pd.Series(r).rolling(6).sum().to_numpy(),
        "ret24": pd.Series(r).rolling(24).sum().to_numpy(),
        "rsi": _rsi(c, 14),
        "vol24": pd.Series(r).rolling(24).std().to_numpy(),
        "sma20d": (c - s.rolling(20).mean()) / s.rolling(20).std(),
        "sma50d": (c - s.rolling(50).mean()) / s.rolling(50).std(),
        "rangepos": (c - pd.Series(l).rolling(24).min()) /
                    (pd.Series(h).rolling(24).max() - pd.Series(l).rolling(24).min() + 1e-9),
        "atrpct": atr(h, l, c) / c,
        "volz": (pd.Series(v) - pd.Series(v).rolling(48).mean()) / (pd.Series(v).rolling(48).std() + 1e-9),
    })
    return f


def _rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0)).rolling(n).mean()
    dn = pd.Series(np.where(d < 0, -d, 0)).rolling(n).mean()
    return (100 - 100 / (1 + up / (dn + 1e-9))).to_numpy()


def label(h, l, c, A, tp, sl):
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        a = A[i]
        up, dn = c[i] + tp * a, c[i] - sl * a
        for j in range(i + 1, min(i + H + 1, n)):
            if l[j] <= dn:          # stop first (tie -> stop)
                y[i] = 0; break
            if h[j] >= up:
                y[i] = 1; break
    return y


def run(ticker, tp, sl):
    d = hourly(ticker)
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = features(h, l, c, v)
    y = label(h, l, c, A, tp, sl)
    m = X.notna().all(axis=1) & np.isfinite(y)
    X, y, Ai, ci = X[m], y[m].astype(int), A[m], c[m]
    cut = int(len(y) * 0.6)
    Xtr, ytr = X.iloc[:cut], y[:cut]
    Xte, yte = X.iloc[cut:], y[cut:]
    Ate, cte = Ai[cut:], ci[cut:]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    be = sl / (sl + tp)
    # take only high-confidence longs (top 30% predicted P(target first))
    thr = np.quantile(proba, 0.70)
    sel = proba >= thr
    base_wr = yte.mean()
    sel_wr = yte[sel].mean() if sel.sum() else float("nan")
    # expectancy in $ for the selected trades (win=+tp*ATR, loss=-sl*ATR) net of cost
    cost = COST_BPS / 1e4 * cte[sel]
    pnl = np.where(yte[sel] == 1, tp * Ate[sel], -sl * Ate[sel]) - cost
    exp = pnl.mean() if sel.sum() else float("nan")
    print(f"  {ticker} TP:SL={tp}:{sl}  break-even={be:.0%}  | "
          f"base win%={base_wr:.1%}  ML-selected win%={sel_wr:.1%}  "
          f"exp$/trade={exp:+.3f}  ({'EDGE' if sel_wr > be + 0.03 and exp > 0 else 'no edge'})")


if __name__ == "__main__":
    print("Triple-barrier + ML entry selection (hourly, OOS 40%):")
    for tk in ("SPY", "QQQ"):
        for tp, sl in ((1, 1), (2, 1), (1.5, 1)):
            run(tk, tp, sl)
    print("\nREAD: ML-selected win% must clear break-even by a clear margin AND exp$>0.")
    print("If selected ~= break-even, the geometry holds -> no learnable entry edge.")
