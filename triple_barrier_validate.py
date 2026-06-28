"""
triple_barrier_validate.py — does the 1:1 ML entry edge SURVIVE validation?

Realistic, NON-overlapping trades (enter only when flat), walk-forward (expanding
window, threshold chosen on TRAIN only -> no lookahead), across regimes. Reports
per-fold win rate + a binomial z-score vs 50% on the independent trades, plus net
expectancy. >50% consistently AND significant = a real learnable edge; noisy /
insignificant = the single-split 54% was luck.
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

from triple_barrier_ml import atr, features, hourly, label, H

import os
COST_BPS = float(os.environ.get("TB_COST_BPS", "1.0"))
TP = SL = 1.0


def trade_fold(clf, thr, h, l, c, A, X, i0, i1):
    """Walk test bars [i0,i1); enter long when flat & proba>=thr; hold to barrier."""
    proba = clf.predict_proba(X.iloc[i0:i1])[:, 1]
    wins = pnl = ntr = 0
    i = i0
    while i < i1 - 1:
        if proba[i - i0] < thr:
            i += 1
            continue
        a = A[i]
        up, dn = c[i] + TP * a, c[i] - SL * a
        out = None
        j = i + 1
        while j < min(i + H + 1, len(c)):
            if l[j] <= dn:
                out = 0; break
            if h[j] >= up:
                out = 1; break
            j += 1
        if out is None:
            out = 1 if c[min(j, len(c) - 1)] > c[i] else 0
        ntr += 1
        wins += out
        pnl += (TP * a if out == 1 else -SL * a) - COST_BPS / 1e4 * c[i]
        i = j + 1                      # non-overlapping: resume after exit
    return ntr, wins, pnl


def run(ticker):
    d = hourly(ticker)
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = features(h, l, c, v)
    y = label(h, l, c, A, TP, SL)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv = X.iloc[idx].reset_index(drop=True)
    yv = y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    n = len(idx)

    K = 5
    start = int(n * 0.4)
    bnds = np.linspace(start, n, K + 1).astype(int)
    print(f"\n=== {ticker} 1:1 triple-barrier, walk-forward (non-overlapping trades) ===")
    tot_tr = tot_win = 0
    tot_pnl = 0.0
    wrs = []
    for k in range(K):
        tr_end = bnds[k]
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:tr_end], yv[:tr_end])
        thr = np.quantile(clf.predict_proba(Xv.iloc[:tr_end])[:, 1], 0.70)  # train-only
        ntr, wins, pnl = trade_fold(clf, thr, hv, lv, cv, Av, Xv, bnds[k], bnds[k + 1])
        wr = wins / ntr if ntr else float("nan")
        wrs.append(wr)
        tot_tr += ntr; tot_win += wins; tot_pnl += pnl
        print(f"  fold {k+1}: trades={ntr:3d}  win%={wr:5.1%}  pnl=${pnl:7.2f}")
    wr = tot_win / tot_tr
    z = (tot_win - 0.5 * tot_tr) / np.sqrt(0.25 * tot_tr) if tot_tr else 0
    print(f"  POOLED: trades={tot_tr}  win%={wr:.1%} (break-even 50%)  "
          f"z-vs-50%={z:+.2f}  total pnl=${tot_pnl:.2f}  folds>50%={sum(x>0.5 for x in wrs)}/{K}")
    print(f"    z>1.96 => significant; folds>50% must be consistent (e.g. 4-5/5).")


if __name__ == "__main__":
    for tk in ("SPY", "QQQ"):
        run(tk)
    print("\nVERDICT logic: real edge = win% clearly>50% on MOST folds AND z>~2 AND pnl>0.")
