"""
v4_tune.py — (A) v4 (15-min/4:1) over the LAST 2 WEEKS, and (B) can we raise the 4:1 edge by
being MORE selective (top 7% -> 4% -> 2% conviction)? Part B is on the full walk-forward (dev).
Higher selectivity should lift win% above the 20% break-even if the model is well-calibrated.
Standalone; touches no frozen snapshot.
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

from triple_barrier_ml import atr, features, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features

MIN, TP, SL = 15, 4.0, 1.0
HBAR, COST_BPS = 24, 3.0
SEL_LEVELS = [0.93, 0.96, 0.98]            # top 7%, 4%, 2%
TW_START, TW_END = pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-29")
END = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)


def load5(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    p = Path(f"data_cache/{tk}_recent_2026-06-01_{END.date()}.csv")
    if p.exists():
        rec = pd.read_csv(p, parse_dates=["timestamp"])
        df = (pd.concat([df, rec], ignore_index=True)
                .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    return df


def bars15(df):
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def walk_trades(proba, thr, h, l, c, A, i0, i1, n):
    rets, i = [], i0
    while i < i1 - 1:
        if proba[i - i0] < thr:
            i += 1; continue
        a = A[i]; up, dn = c[i] + TP * a, c[i] - SL * a
        res, j = None, i + 1
        while j < min(i + HBAR + 1, n):
            if l[j] <= dn:
                res = 0; break
            if h[j] >= up:
                res = 1; break
            j += 1
        if res is None:
            res = 1 if c[min(j, n - 1)] > c[i] else 0
        rets.append((TP * a if res == 1 else -SL * a) / c[i] - COST_BPS / 1e4)
        i = j + 1
    return rets


def part_a():
    print(f"=== A) LAST 2 WEEKS ({TW_START.date()} .. {(TW_END-pd.Timedelta(days=3)).date()}), v4 top-7% ===")
    rows = []
    for tk in TICKERS:
        try:
            d = bars15(load5(tk))
        except Exception:
            continue
        ts = pd.to_datetime(d["timestamp"]).to_numpy()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        y = label(h, l, c, A, TP, SL)
        fv = X.notna().all(axis=1).to_numpy(); n = len(c)
        tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(TW_START)))[0]
        if len(tr) < 500:
            continue
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr].astype(int))
        thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], 0.93)
        fwd = np.where(fv & (ts >= np.datetime64(TW_START)) & (ts < np.datetime64(TW_END)))[0]
        if len(fwd) == 0:
            continue
        proba = {int(ix): float(p) for ix, p in zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
        i, last = int(fwd[0]), int(fwd[-1])
        while i <= last:
            if proba.get(i, -1) < thr:
                i += 1; continue
            a = A[i]; up, dn = c[i] + TP * a, c[i] - SL * a
            res, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if l[j] <= dn:
                    res = 0; break
                if h[j] >= up:
                    res = 1; break
                j += 1
            if res is None and j >= n:
                rows.append((tk, str(pd.Timestamp(ts[i]))[5:16], "OPEN", None)); i = n; continue
            if res is None:
                res = 1 if c[min(j, n - 1)] > c[i] else 0
            rows.append((tk, str(pd.Timestamp(ts[i]))[5:16], "TARGET" if res == 1 else "STOP",
                         round(((TP if res == 1 else -SL) * a) / c[i] * 100 - COST_BPS / 100, 2)))
            i = j + 1
    cl = [r for r in rows if r[3] is not None]
    wins = sum(r[2] == "TARGET" for r in cl)
    tot = sum(r[3] for r in cl)
    print(f"  trades={len(cl)}  wins={wins}  win%={(wins/len(cl)*100 if cl else 0):.0f}%  "
          f"(break-even 20%)  total={tot:+.2f}%  open={sum(r[2]=='OPEN' for r in rows)}")
    for r in sorted(cl, key=lambda x: -(x[3] or 0))[:6]:
        print(f"    best: {r[0]:>5} {r[1]}  {r[2]}  {r[3]:+.2f}%")
    print()


def part_b():
    print("=== B) RAISE THE EDGE: v4 (15-min/4:1) at tighter selectivity, full walk-forward (dev) ===")
    agg = {q: [] for q in SEL_LEVELS}
    for tk in TICKERS:
        try:
            d = bars15(pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"]))
        except Exception:
            continue
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
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, verbose=-1)
            clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
            ptr = clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1]
            proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
            for q in SEL_LEVELS:
                thr = np.quantile(ptr, q)
                agg[q] += walk_trades(proba, thr, hv, lv, cv, Av, bnds[k], bnds[k + 1], n)
    print(f"  {'selectivity':>12} {'trades':>7} {'win%':>6} {'mean bps':>9} {'total%':>8}")
    for q in SEL_LEVELS:
        r = np.array(agg[q])
        print(f"  {'top '+str(round((1-q)*100,1))+'%':>12} {len(r):>7} {(r>0).mean():>6.1%} "
              f"{r.mean()*1e4:>+9.1f} {r.sum()*100:>+8.0f}")
    print("\n  If win% rises with selectivity AND mean bps improves, tighter is a real edge boost")
    print("  (then validate the best on fresh names). If win% rises but total falls, it's just fewer trades.")


if __name__ == "__main__":
    part_a()
    part_b()
