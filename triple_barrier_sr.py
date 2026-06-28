"""
triple_barrier_sr.py — do S/R features help the bracket ML find an edge?

A/B: base candle features vs base + S/R features, at 1:1 AND 2:1, across the
8-name basket, walk-forward, non-overlapping trades, charged a realistic 3 bps.
Win rate is cost-independent (does S/R raise it above break-even?); P&L at 3 bps
says whether it's tradeable. The imported S/R README's own caveat: synthetic
edge may not transfer -- so this measures it on REAL data, honestly.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import os

import numpy as np
import pandas as pd
import lightgbm as lgb

from triple_barrier_ml import atr, features, hourly, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features

COST_BPS = float(os.environ.get("COST_BPS", "3.0"))
SEL_Q = float(os.environ.get("SEL_Q", "0.70"))   # selectivity: take top (1-SEL_Q) entries


def feat(d, use_sr):
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    X = features(h, l, c, v)
    if use_sr:
        X = pd.concat([X.reset_index(drop=True), sr_features(d).reset_index(drop=True)], axis=1)
    return X


def trade_fold(clf, thr, h, l, c, A, X, i0, i1, tp, sl):
    proba = clf.predict_proba(X.iloc[i0:i1])[:, 1]
    wins = pnl = ntr = 0
    i = i0
    while i < i1 - 1:
        if proba[i - i0] < thr:
            i += 1
            continue
        a = A[i]
        up, dn = c[i] + tp * a, c[i] - sl * a
        out = None
        j = i + 1
        while j < min(i + 24 + 1, len(c)):
            if l[j] <= dn:
                out = 0; break
            if h[j] >= up:
                out = 1; break
            j += 1
        if out is None:
            out = 1 if c[min(j, len(c) - 1)] > c[i] else 0
        ntr += 1; wins += out
        pnl += (tp * a if out == 1 else -sl * a) - COST_BPS / 1e4 * c[i]
        i = j + 1
    return ntr, wins, pnl


def pooled(ticker, tp, sl, use_sr):
    d = hourly(ticker)
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = feat(d, use_sr)
    y = label(h, l, c, A, tp, sl)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    n = len(idx)
    K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    tr = win = 0
    pnl = 0.0
    for k in range(K):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
        thr = np.quantile(clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1], SEL_Q)
        nt, w, p = trade_fold(clf, thr, hv, lv, cv, Av, Xv, bnds[k], bnds[k + 1], tp, sl)
        tr += nt; win += w; pnl += p
    return tr, win, pnl


def run(tp, sl, use_sr):
    TT = TW = 0
    TP = 0.0
    for tk in TICKERS:
        try:
            t, w, p = pooled(tk, tp, sl, use_sr)
        except Exception:
            continue
        TT += t; TW += w; TP += p
    wr = TW / TT
    z = (TW - 0.5 * TT) / np.sqrt(0.25 * TT)
    be = sl / (sl + tp)
    return wr, z, TP, TT, be


if __name__ == "__main__":
    print(f"Triple-barrier ML, base vs +S/R features, basket of {len(TICKERS)}, "
          f"cost {COST_BPS} bps:")
    print(f"  {'TP:SL':>6} {'features':>10} {'breakeven':>9} {'win%':>6} {'z':>6} {'pnl$':>8}")
    for tp, sl in ((1, 1), (2, 1)):
        for use_sr in (False, True):
            wr, z, pnl, tt, be = run(tp, sl, use_sr)
            tag = "base+S/R" if use_sr else "base"
            print(f"  {tp}:{sl:<4} {tag:>10} {be:>9.0%} {wr:>6.1%} {z:>+6.2f} {pnl:>8.0f}")
    print("\n  READ: S/R 'helps' only if base+S/R win% clearly > base AND pnl turns positive")
    print("  at 3 bps. Win% is cost-free; pnl@3bps is the tradeability test.")
