"""
make_better.py — improvement attempt: CONVICTION-WEIGHTED SIZING on the validated
30-min/1.5:1. Same signals (no change to WHICH trades we take), but size each trade by
the model's confidence (predicted probability). This adds NO new signal -> almost no
room to overfit; it only helps if the model's probability is genuinely informative
(higher proba -> higher actual win rate). Validated on dev AND the fresh holdout.

  python make_better.py          # dev names
  python make_better.py fresh    # fresh holdout
"""

from __future__ import annotations

import math
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

FRESH = ["IWM", "GLD", "META", "XOM", "KO"]
MIN = 30
TP, SL = 1.5, 1.0
SEL_Q = 0.93
HBAR = 24
COST_BPS = 3.0


def bars(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                     parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum")).dropna()
    return g.reset_index()


def trades_of(ticker):
    d = bars(ticker)
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    n = len(idx)
    K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    rets, probs = [], []
    for k in range(K):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
        ptr = clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1]
        thr = np.quantile(ptr, SEL_Q)
        proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
        i = bnds[k]
        while i < bnds[k + 1] - 1:
            p = proba[i - bnds[k]]
            if p < thr:
                i += 1
                continue
            a = Av[i]
            up, dn = cv[i] + TP * a, cv[i] - SL * a
            out, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if lv[j] <= dn:
                    out = 0; break
                if hv[j] >= up:
                    out = 1; break
                j += 1
            if out is None:
                out = 1 if cv[min(j, n - 1)] > cv[i] else 0
            rets.append((TP * a if out == 1 else -SL * a) / cv[i] - COST_BPS / 1e4)
            probs.append(p)
            i = j + 1
    return np.array(rets), np.array(probs)


def stats(r, w, per_year):
    rw = w * r
    return rw.mean() * 1e4, rw.mean() / rw.std() * math.sqrt(per_year), rw.sum() * 100


def main():
    mode = "fresh" if (len(sys.argv) > 1 and sys.argv[1] == "fresh") else "dev"
    names = FRESH if mode == "fresh" else DEV
    allr, allp = [], []
    for tk in names:
        try:
            r, p = trades_of(tk)
        except Exception as e:
            print(f"  [skip {tk}] {e}")
            continue
        allr.append(r); allp.append(p)
    r = np.concatenate(allr); p = np.concatenate(allp)
    n = len(r); per_year = n / 3.0

    print(f"=== IMPROVEMENT: conviction-weighted sizing, 30-min/1.5:1 [{mode.upper()}] ===")
    print(f"  {n} trades\n")
    # calibration: is higher model-proba => higher win rate?
    q = np.asarray(pd.qcut(p, 3, labels=["low conv", "mid conv", "high conv"]))
    print("  calibration (within the top-7% we already take):")
    print(f"    {'bucket':>10} {'trades':>7} {'win%':>6} {'mean bps':>9}")
    for b in ["low conv", "mid conv", "high conv"]:
        msk = (q == b)
        print(f"    {b:>10} {msk.sum():>7} {(r[msk]>0).mean():>6.1%} {r[msk].mean()*1e4:>+9.1f}")

    # equal weight vs conviction weight (weights normalized to mean 1 -> same capital)
    w_eq = np.ones(n)
    w_cv = p / p.mean()
    w_cv2 = (p ** 2) / (p ** 2).mean()   # more aggressive tilt to high conviction
    print("\n  sizing scheme        mean bps   Sharpe   total%")
    for tag, w in (("equal weight (base)", w_eq),
                   ("conviction ^1", w_cv),
                   ("conviction ^2", w_cv2)):
        bps, sh, tot = stats(r, w, per_year)
        print(f"    {tag:>20} {bps:>+8.1f} {sh:>8.2f} {tot:>+8.0f}")
    print("\n  IMPROVED only if conviction schemes beat equal-weight on BOTH dev and fresh,")
    print("  and the calibration win% rises low->high. Otherwise proba isn't informative enough.")


if __name__ == "__main__":
    main()
