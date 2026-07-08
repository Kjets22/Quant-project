"""
timeframe_test.py — does a finer candle (15 / 30 min) beat the 1-hour bar?

Same strategy engine (base + S/R features, top-7% selectivity, walk-forward, 3 bps,
H = 24 bars), just resampled to different bar sizes from the 5-min cache, pooled across
the 8-name basket. Reports trades, win%, statistical z, total return%, mean bps/trade,
and avg expected move (ATR%) so we can see how move size shrinks at finer bars.

STANDALONE study. Does NOT touch the frozen v1 (1:1) / v2 (1.5:1) snapshots or the
working files. Run:  python timeframe_test.py 1        (just 1:1)
                     python timeframe_test.py 1.5      (just 1.5:1)
                     python timeframe_test.py 1,1.5    (both)
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb

from triple_barrier_ml import atr, features, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features

COST_BPS = 3.0
SEL_Q = 0.93
HBAR = 24
TIMEFRAMES = (60, 30, 15)


def bars(ticker, minutes):
    df = pd.read_csv(f"data_cache/{ticker}_5minute_2021-06-01_2026-06-01.csv",
                     parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{minutes}min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def pooled(ticker, minutes, tp, sl):
    d = bars(ticker, minutes)
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, tp, sl)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    n = len(idx)
    K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    tr = win = 0
    sumret = summove = 0.0
    for k in range(K):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
        thr = np.quantile(clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1], SEL_Q)
        proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
        i = bnds[k]
        while i < bnds[k + 1] - 1:
            if proba[i - bnds[k]] < thr:
                i += 1
                continue
            a = Av[i]
            up, dn = cv[i] + tp * a, cv[i] - sl * a
            res, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if lv[j] <= dn:
                    res = 0; break
                if hv[j] >= up:
                    res = 1; break
                j += 1
            if res is None:
                res = 1 if cv[min(j, n - 1)] > cv[i] else 0
            tr += 1; win += res
            sumret += (tp * a if res == 1 else -sl * a) / cv[i] - COST_BPS / 1e4
            summove += a / cv[i]
            i = j + 1
    return tr, win, sumret, summove


def run(minutes, tp, sl):
    TT = TW = 0
    SR = SM = 0.0
    for tk in TICKERS:
        try:
            t, w, sr, sm = pooled(tk, minutes, tp, sl)
        except Exception as e:
            print(f"    [skip {tk} {minutes}m] {e}")
            continue
        TT += t; TW += w; SR += sr; SM += sm
    z = (TW - 0.5 * TT) / np.sqrt(0.25 * TT) if TT else 0
    return TT, TW / TT, z, SR * 100, SR / TT * 1e4, SM / TT * 100


def main():
    ratios = sys.argv[1] if len(sys.argv) > 1 else "1,1.5"
    pairs = [(float(r), 1.0) for r in ratios.split(",")]
    print("Finer candles vs 1-hour — same engine, top-7%, 3 bps, H=24 bars, basket pooled\n")
    for tp, sl in pairs:
        be = sl / (sl + tp)
        print(f"=== target:stop = {tp:g}:{sl:g}  (break-even win% = {be:.0%}) ===")
        print(f"  {'candle':>7} {'trades':>7} {'win%':>6} {'z':>6} {'avgMove%':>9} "
              f"{'mean bps':>9} {'total%':>8}")
        for minutes in TIMEFRAMES:
            n, wr, z, tot, bps, mv = run(minutes, tp, sl)
            print(f"  {str(minutes)+'min':>7} {n:>7} {wr:>6.1%} {z:>+6.2f} {mv:>9.2f} "
                  f"{bps:>+9.1f} {tot:>+8.0f}", flush=True)
        print()
    print("READ: finer candles add trades but shrink the move (avgMove%). If mean bps and z")
    print("hold up at 30/15 min it's an improvement; if they fade, costs eat the smaller moves.")


if __name__ == "__main__":
    main()
