"""
triple_barrier_breadth.py — the asset-class test (the guide's key anti-overfit rule:
"a pattern found only on one security is likely a false discovery").

Run the 1:1 triple-barrier + ML entry edge across a basket of liquid names,
walk-forward, non-overlapping trades. If the ML-selected win rate holds ~>50%
BROADLY (most names + a strongly significant pooled z over thousands of independent
trades), it's a real (tiny) systematic edge. If it scatters around 50%, the
SPY/QQQ 52% was noise.
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

from triple_barrier_ml import atr, features, hourly, label
from triple_barrier_validate import trade_fold

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "JPM", "XLE", "TLT"]


def pooled(ticker):
    if not Path(f"data_cache/{ticker}_5minute_2021-06-01_2026-06-01.csv").exists():
        return None
    d = hourly(ticker)
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = features(h, l, c, v)
    y = label(h, l, c, A, 1.0, 1.0)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    n = len(idx)
    K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    tr = win = 0
    pnl = 0.0
    fpos = 0
    for k in range(K):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
        thr = np.quantile(clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1], 0.70)
        nt, w, p = trade_fold(clf, thr, hv, lv, cv, Av, Xv, bnds[k], bnds[k + 1])
        tr += nt; win += w; pnl += p
        if nt and w / nt > 0.5:
            fpos += 1
    return tr, win, pnl, fpos, K


def main():
    print("=== 1:1 triple-barrier ML edge across the asset class (walk-forward) ===")
    print(f"  {'ticker':>6} {'trades':>7} {'win%':>6} {'z':>6} {'pnl$':>8} {'folds>50':>9}")
    TT = TW = 0
    TP = 0.0
    for tk in TICKERS:
        r = pooled(tk)
        if r is None:
            print(f"  {tk:>6}  (no data)")
            continue
        tr, win, pnl, fpos, K = r
        wr = win / tr
        z = (win - 0.5 * tr) / np.sqrt(0.25 * tr)
        TT += tr; TW += win; TP += pnl
        print(f"  {tk:>6} {tr:>7} {wr:>6.1%} {z:>+6.2f} {pnl:>8.2f} {fpos:>6}/{K}")
    wr = TW / TT
    z = (TW - 0.5 * TT) / np.sqrt(0.25 * TT)
    print(f"\n  POOLED ALL: trades={TT}  win%={wr:.2%}  z-vs-50%={z:+.2f}  total pnl=${TP:.0f}")
    print(f"  z>3 over {TT} independent trades AND most names>50% => a real systematic edge.")
    print(f"  z~1-2 / scattered => SPY-QQQ 52% was multiple-testing noise.")


if __name__ == "__main__":
    main()
