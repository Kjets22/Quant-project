"""
flexible_profit.py — "flexible profit" test: instead of a FIXED target, let winners RUN
with a trailing stop. Entry signal = the validated model (trained on the balanced 1:1
directional label, top-7%). Exit = initial stop at -1 ATR, then trail the stop 1.5 ATR
below the high-water-mark, so strong trends run far and weak ones cut quickly.

Reports full validated period, this month, and last week at 30-min and 15-min, with the
average winner vs loser (the "flexible" payoff). Standalone; touches no frozen snapshot.
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

SEL_Q, HBAR, COST_BPS = 0.93, 24, 3.0
SL_INIT, TRAIL = 1.0, 1.5         # initial stop / trailing distance, in ATR
MONTH_START = pd.Timestamp("2026-06-01")
LWS, LWE = pd.Timestamp("2026-06-22"), pd.Timestamp("2026-06-29")
END = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
_C5 = {}


def load5(tk):
    if tk in _C5:
        return _C5[tk]
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    p = Path(f"data_cache/{tk}_recent_2026-06-01_{END.date()}.csv")
    if p.exists():
        rec = pd.read_csv(p, parse_dates=["timestamp"])
        df = (pd.concat([df, rec], ignore_index=True)
                .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    _C5[tk] = df
    return df


def resample(df, mins):
    g = df.set_index("timestamp").resample(f"{mins}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def trail_ret(i, h, l, c, A, n):
    entry, a = c[i], A[i]
    stop, hwm, j = entry - SL_INIT * a, c[i], i + 1
    while j < min(i + HBAR + 1, n):
        if h[j] > hwm:
            hwm = h[j]
        ns = hwm - TRAIL * a
        if ns > stop:
            stop = ns
        if l[j] <= stop:
            return (stop - entry) / entry - COST_BPS / 1e4, j
        j += 1
    ex = min(j, n - 1)
    return (c[ex] - entry) / entry - COST_BPS / 1e4, ex


def _walk_take(proba, thr, h, l, c, A, i0, i1, n):
    rets, i = [], i0
    while i < i1 - 1:
        if proba[i - i0] < thr:
            i += 1; continue
        r, j = trail_ret(i, h, l, c, A, n)
        rets.append(r); i = j + 1
    return rets


def prep(tk, mins):
    d = resample(load5(tk), mins)
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, 1.0, 1.0)          # balanced directional label for the entry model
    return ts, h, l, c, A, X, y


def gen_walk(tk, mins):
    ts, h, l, c, A, X, y = prep(tk, mins)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av = h[idx], l[idx], c[idx], A[idx]
    n = len(idx); K = 5
    bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
    out = []
    for k in range(K):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(Xv.iloc[:bnds[k]], yv[:bnds[k]])
        thr = np.quantile(clf.predict_proba(Xv.iloc[:bnds[k]])[:, 1], SEL_Q)
        proba = clf.predict_proba(Xv.iloc[bnds[k]:bnds[k + 1]])[:, 1]
        out += _walk_take(proba, thr, hv, lv, cv, Av, bnds[k], bnds[k + 1], n)
    return out


def gen_window(tk, mins, start, end):
    ts, h, l, c, A, X, y = prep(tk, mins)
    fv = X.notna().all(axis=1).to_numpy(); n = len(c)
    tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(start)))[0]
    if len(tr) < 500:
        return []
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q)
    fwd = np.where(fv & (ts >= np.datetime64(start)) & (ts < np.datetime64(end)))[0]
    if len(fwd) == 0:
        return []
    proba = {int(ix): float(p) for ix, p in zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
    out, i, last = [], int(fwd[0]), int(fwd[-1])
    while i <= last:
        if proba.get(i, -1) < thr:
            i += 1; continue
        r, j = trail_ret(i, h, l, c, A, n)
        out.append(r); i = j + 1
    return out


def report(rets, tag):
    if not rets:
        print(f"  {tag:>34}: no trades"); return
    r = np.array(rets)
    w = r[r > 0]; lo = r[r <= 0]
    aw = w.mean() * 100 if len(w) else 0
    al = lo.mean() * 100 if len(lo) else 0
    print(f"  {tag:>34}: n={len(r):>4} win {(r>0).mean():>5.1%}  avgWin {aw:>+5.2f}% avgLoss {al:>+5.2f}%"
          f"  mean {r.mean()*1e4:>+5.1f}bps  total {r.sum()*100:>+6.0f}%")


def main():
    print(f"FLEXIBLE PROFIT (trailing stop: init -{SL_INIT} ATR, trail {TRAIL} ATR), top-7%, 3 bps\n")
    for mins in (30, 15):
        print(f"--- {mins}-min ---")
        full = []
        for tk in TICKERS:
            try:
                full += gen_walk(tk, mins)
            except Exception as e:
                print(f"   [skip {tk}] {e}")
        report(full, "full validated period")
        mon = []
        for tk in TICKERS:
            mon += gen_window(tk, mins, MONTH_START, END)
        report(mon, "this month (Jun)")
        lw = []
        for tk in TICKERS:
            lw += gen_window(tk, mins, LWS, LWE)
        report(lw, "last week (Jun 22-26)")
        print()
    print("Compare to FIXED-target saved configs (full period): v3 30m/1.5:1 +114%, v4 15m/4:1 +129%.")
    print("Flexible 'wins' should be bigger than losses (winners run); does mean bps beat the fixed ones?")


if __name__ == "__main__":
    main()
