"""
ml_accuracy.py — try to make the model MORE ACCURATE (raise the win rate of the selected
top-7% trades) without overfitting. Three variants on 30-min/1.5:1, walk-forward (dev):
  - baseline  : the current single LightGBM
  - ensemble  : average of 3 LightGBMs (different seeds) -> variance reduction
  - tuned     : more trees + leaves + stronger regularization
Accuracy = OOS win% of the selected trades. A real improvement raises win% AND mean bps,
and must then hold on fresh names. Standalone; touches no frozen snapshot.
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

MIN, TP, SL = 30, 1.5, 1.0
SEL_Q, HBAR, COST_BPS = 0.93, 24, 3.0
BASE = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
TUNED = dict(n_estimators=600, learning_rate=0.02, num_leaves=31, min_child_samples=80,
             subsample=0.7, colsample_bytree=0.7, reg_lambda=3.0, verbose=-1)


def probas(Xtr, ytr, Xte, variant):
    if variant == "ensemble":
        ptr = np.zeros(len(Xtr)); pte = np.zeros(len(Xte))
        for s in range(3):
            clf = lgb.LGBMClassifier(random_state=1 + 7 * s, **BASE)
            clf.fit(Xtr, ytr)
            ptr += clf.predict_proba(Xtr)[:, 1]; pte += clf.predict_proba(Xte)[:, 1]
        return ptr / 3, pte / 3
    p = TUNED if variant == "tuned" else BASE
    clf = lgb.LGBMClassifier(**p)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xtr)[:, 1], clf.predict_proba(Xte)[:, 1]


def walk(proba, thr, hv, lv, cv, Av, i0, i1, n):
    rets, i = [], i0
    while i < i1 - 1:
        if proba[i - i0] < thr:
            i += 1; continue
        a = Av[i]; up, dn = cv[i] + TP * a, cv[i] - SL * a
        res, j = None, i + 1
        while j < min(i + HBAR + 1, n):
            if lv[j] <= dn:
                res = 0; break
            if hv[j] >= up:
                res = 1; break
            j += 1
        if res is None:
            res = 1 if cv[min(j, n - 1)] > cv[i] else 0
        rets.append(((TP * a if res == 1 else -SL * a) / cv[i] - COST_BPS / 1e4, res))
        i = j + 1
    return rets


def main():
    variants = ["baseline", "ensemble", "tuned"]
    agg = {x: [] for x in variants}
    for tk in TICKERS:
        try:
            df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
        except Exception:
            continue
        d = df.set_index("timestamp").resample(f"{MIN}min").agg(
            high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum")).dropna().reset_index()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        y = label(h, l, c, A, TP, SL)
        m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
        idx = np.where(m)[0]
        Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
        hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
        n = len(idx); K = 5
        bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
        for k in range(K):
            Xtr, ytr = Xv.iloc[:bnds[k]], yv[:bnds[k]]
            Xte = Xv.iloc[bnds[k]:bnds[k + 1]]
            for x in variants:
                ptr, pte = probas(Xtr, ytr, Xte, x)
                agg[x] += walk(pte, np.quantile(ptr, SEL_Q), hv, lv, cv, Av, bnds[k], bnds[k + 1], n)
        print(f"  ...{tk} done", flush=True)
    print("\nMODEL ACCURACY (30-min/1.5:1, top-7%, walk-forward dev). Higher win% = more accurate.\n")
    print(f"  {'variant':>10} {'trades':>7} {'win%':>6} {'mean bps':>9} {'total%':>8}")
    for x in variants:
        a = np.array(agg[x])
        r, w = a[:, 0], a[:, 1]
        print(f"  {x:>10} {len(a):>7} {w.mean():>6.2%} {r.mean()*1e4:>+9.1f} {r.sum()*100:>+8.0f}")
    print("\n  baseline win% is the bar. If ensemble/tuned beats it on win% AND mean bps, it's a real")
    print("  accuracy gain -> next step is to confirm it on the fresh holdout names.")


if __name__ == "__main__":
    main()
