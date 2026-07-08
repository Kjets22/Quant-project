"""
exit_model6.py — sweep the exit aggressiveness to find ~90% accuracy and report the total.
FIXED accounting (exit only BEFORE the resolving bar, so cuts are above the stop). Less
reluctant model (pos_weight lower) + rising thresholds cut MORE trades -> precision falls
from ~100% toward lower. 'accuracy' = % of cuts that were NOT target-bound winners. Tracks
the costly mistakes (target winners cut during their dip). Standalone.
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
from triple_barrier_breadth import TICKERS as DEV
from sr_features import sr_features

MIN, TP, SL = 15, 4.0, 1.0
SEL_Q, HBAR, COST = 0.93, 24, 3.0 / 1e4
POS_W = 1.0
THRS = [0.10, 0.18, 0.26, 0.34, 0.42, 0.50]


def prep(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    d = df.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna().reset_index()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    return h, l, c, A, X.to_numpy(float), label(h, l, c, A, TP, SL)


def walk_trade(i, h, l, c, A, n):
    a = A[i]; up, dn = c[i] + TP * a, c[i] - SL * a
    j = i + 1
    while j < min(i + HBAR + 1, n):
        if l[j] <= dn:
            return "S", j
        if h[j] >= up:
            return "T", j
        j += 1
    return "X", min(j, n - 1)


def base_ret(res, ex, i, c, A):
    a = A[i]
    if res == "T":
        return TP * a / c[i] - COST
    if res == "S":
        return -SL * a / c[i] - COST
    return (c[ex] - c[i]) / c[i] - COST


def tstate(i, j, h, l, c, A):
    a = A[i]
    return [(c[j] - c[i]) / a, (j - i) / HBAR, (h[i + 1:j + 1].max() - c[i]) / a,
            (l[i + 1:j + 1].min() - c[i]) / a, (c[i] + TP * a - c[j]) / a, (c[j] - (c[i] - SL * a)) / a]


def main():
    exit_X, exit_y, store = [], [], []
    for tk in DEV:
        h, l, c, A, Xnp, y = prep(tk)
        n = len(c)
        fin = np.isfinite(Xnp).all(axis=1) & np.isfinite(y)
        cut = int(n * 0.6)
        idxtr = np.where(fin & (np.arange(n) < cut))[0]
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(Xnp[idxtr], y[idxtr].astype(int))
        proba = np.full(n, -1.0); vi = np.where(fin)[0]
        proba[vi] = clf.predict_proba(Xnp[vi])[:, 1]
        thr = np.quantile(proba[idxtr], SEL_Q)
        i = 0
        while i < cut - 1:
            if not fin[i] or proba[i] < thr:
                i += 1; continue
            res, ex = walk_trade(i, h, l, c, A, n)
            tgt = 1 if res == "T" else 0
            for j in range(i + 1, ex + 1):
                if np.isfinite(Xnp[j]).all():
                    exit_X.append(np.concatenate([Xnp[j], tstate(i, j, h, l, c, A)])); exit_y.append(tgt)
            i = ex + 1
        tests = []
        i = cut
        while i < n - 1:
            if not fin[i] or proba[i] < thr:
                i += 1; continue
            res, ex = walk_trade(i, h, l, c, A, n)
            tests.append((i, res, ex)); i = ex + 1
        store.append((h, l, c, A, Xnp, tests))

    exclf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31,
                               min_child_samples=80, subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=2.0, scale_pos_weight=POS_W, verbose=-1)
    exclf.fit(np.array(exit_X), np.array(exit_y))

    pre = []
    for h, l, c, A, Xnp, tests in store:
        for i, res, ex in tests:
            b = base_ret(res, ex, i, c, A)
            rows, cjs = [], []
            for j in range(i + 1, ex):
                if np.isfinite(Xnp[j]).all():
                    rows.append(np.concatenate([Xnp[j], tstate(i, j, h, l, c, A)])); cjs.append(c[j])
            ptgt = exclf.predict_proba(np.array(rows))[:, 1] if rows else np.array([])
            pre.append((b, c[i], np.array(cjs), ptgt, res))

    base = np.array([p[0] for p in pre])
    print(f"=== EXIT MODEL v6 (sweep accuracy) on v4 [DEV] — {len(pre)} trades, baseline {base.sum()*100:+.0f}% ===\n")
    print(f"  {'P(tgt)<':>8} {'#cut':>6} {'accuracy':>9} {'TARGET-winners cut':>19} {'total%':>8}")
    for THR in THRS:
        rets, ncut, tgt_cut = [], 0, 0
        for b, ci, cjs, ptgt, res in pre:
            er = b
            w = np.where(ptgt < THR)[0]
            if len(w):
                er = (cjs[w[0]] - ci) / ci - COST
                ncut += 1
                if res == "T":
                    tgt_cut += 1
            rets.append(er)
        r = np.array(rets)
        acc = 1 - tgt_cut / ncut if ncut else 1
        print(f"  {THR:>8} {ncut:>6} {acc:>9.0%} {tgt_cut:>19} {r.sum()*100:>+8.0f}")
    print("\n  Watch the row near ~90% accuracy: even a handful of TARGET-winners cut (each worth")
    print("  ~4 ATR forgone) outweighs the small saves on losers. That's the 4:1 asymmetry at work.")


if __name__ == "__main__":
    main()
