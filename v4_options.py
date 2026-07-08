"""
v4_options.py — make the 15-min/4:1 strategy (v4) trade OPTIONS instead of shares.

v4's edge: short INTRADAY holds (~hours) and big 4:1 winning moves. Short holds mean the
option barely decays (theta is tiny) -- the opposite of the 30-min test -- so this is the
config most likely to survive option frictions. Buys a deep-ITM weekly call at REALISTIC
implied vol (= realized x VRP), buy ask / sell bid. Reports both the full validated period
and this month, stock vs options. Standalone; touches no frozen snapshot.
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
from triple_barrier_breadth import TICKERS
from sr_features import sr_features
from options_strategy import opt_ret

MIN, TP, SL = 15, 4.0, 1.0
SEL_Q, HBAR, COST_BPS = 0.93, 24, 3.0
VRP, HS = 1.25, 0.005
MONTH_START = pd.Timestamp("2026-06-01")
END = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)


def bars(tk):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    from pathlib import Path
    p = Path(f"data_cache/{tk}_recent_2026-06-01_{END.date()}.csv")
    if p.exists():
        rec = pd.read_csv(p, parse_dates=["timestamp"])
        df = (pd.concat([df, rec], ignore_index=True)
                .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    g = df.set_index("timestamp").resample(f"{MIN}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def _prep(tk):
    d = bars(tk)
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr(h, l, c)
    daily = d.set_index("timestamp")["close"].resample("1D").last().dropna()
    iv = min(max(float(np.log(daily / daily.shift(1)).std() * math.sqrt(252)), 0.08), 0.80) * VRP
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    y = label(h, l, c, A, TP, SL)
    return ts, h, l, c, A, X, y, iv


def _trade(i, h, l, c, A, ts, n, iv):
    a = A[i]; up, dn = c[i] + TP * a, c[i] - SL * a
    res, j = None, i + 1
    while j < min(i + HBAR + 1, n):
        if l[j] <= dn:
            res = 0; break
        if h[j] >= up:
            res = 1; break
        j += 1
    ex = j if (res is not None and j < n) else min(j, n - 1)
    if res is None and j >= n:
        return None, n
    if res is None:
        res = 1 if c[ex] > c[i] else 0
    Sx = c[i] + TP * a if res == 1 else c[i] - SL * a
    dt = (pd.Timestamp(ts[ex]) - pd.Timestamp(ts[i])).total_seconds() / 86400.0
    rec = {"S0": c[i], "Sx": Sx, "a": a, "sig": iv, "dt": max(dt, 1 / 48), "win": res}
    return rec, j


def gen_walk(tk):
    ts, h, l, c, A, X, y, iv = _prep(tk)
    m = (X.notna().all(axis=1) & np.isfinite(y)).to_numpy()
    idx = np.where(m)[0]
    Xv, yv = X.iloc[idx].reset_index(drop=True), y[idx].astype(int)
    hv, lv, cv, Av, tsv = h[idx], l[idx], c[idx], A[idx], ts[idx]
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
        i = bnds[k]
        while i < bnds[k + 1] - 1:
            if proba[i - bnds[k]] < thr:
                i += 1; continue
            rec, j = _trade(i, hv, lv, cv, Av, tsv, n, iv)
            if rec is None:
                break
            out.append(rec); i = j + 1
    return out


def gen_month(tk):
    ts, h, l, c, A, X, y, iv = _prep(tk)
    fv = X.notna().all(axis=1).to_numpy(); n = len(c)
    tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(MONTH_START)))[0]
    if len(tr) < 500:
        return []
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q)
    fwd = np.where(fv & (ts >= np.datetime64(MONTH_START)) & (ts < np.datetime64(END)))[0]
    if len(fwd) == 0:
        return []
    proba = {int(ix): float(p) for ix, p in zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
    out = []
    i, last = int(fwd[0]), int(fwd[-1])
    while i <= last:
        if proba.get(i, -1) < thr:
            i += 1; continue
        rec, j = _trade(i, h, l, c, A, ts, n, iv)
        if rec is None:
            break
        out.append(rec); i = j + 1
    return out


def report(trades, label_):
    if not trades:
        print(f"  {label_}: no trades"); return
    sret = np.array([((TP if t["win"] else -SL) * t["a"]) / t["S0"] - COST_BPS / 1e4 for t in trades])
    am = np.array([t["a"] / t["S0"] for t in trades])
    thr10 = np.quantile(am, 0.90)
    print(f"  {label_}: {len(trades)} trades, win {np.mean([t['win'] for t in trades]):.0%}, "
          f"avg hold {np.mean([t['dt'] for t in trades])*24:.1f}h")
    print(f"    STOCK            total {sret.sum()*100:+.1f}%   ({sret.mean()*1e4:+.1f} bps/trade)")
    for nm, sub in (("OPTIONS all", trades), ("OPTIONS top-10% move",
                                              [t for t, x in zip(trades, am) if x >= thr10])):
        r = np.array([opt_ret(t, "call", 0.95, 7, HS) for t in sub])
        print(f"    {nm:>20} total {r.sum()*100:+.0f}%   ({r.mean()*100:+.2f}%/trade, win {(r>0).mean():.0%}, n={len(r)})")


def main():
    print(f"v4 = 15-min / 4:1 as OPTIONS (deep-ITM weekly, implied vol = realized x {VRP}, buy ask/sell bid)\n")
    walk = []
    for tk in TICKERS:
        try:
            walk += gen_walk(tk)
        except Exception as e:
            print(f"  [skip {tk}] {e}")
    report(walk, "FULL VALIDATED PERIOD (walk-forward)")
    print()
    mon = []
    for tk in TICKERS:
        try:
            mon += gen_month(tk)
        except Exception:
            pass
    report(mon, f"THIS MONTH ({MONTH_START.date()}..{(END-pd.Timedelta(days=1)).date()})")
    print("\n  Options 'total%' is on PREMIUM (leveraged) -- not comparable 1:1 to stock notional;")
    print("  you would size it far smaller. Positive net/trade is the real test.")


if __name__ == "__main__":
    main()
