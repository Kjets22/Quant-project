"""
v6_options_real.py — pick v6's option structure from REAL data (or reject it again).

The 2026-07-15 deep dive tested v6 on SPY only, with ATM/OTM strikes, and found every
bucket flat-to-negative: v6 grinds a modest ATR target over ~8 days, and theta eats the
move. Before putting a v6 options twin on the paper account (user request), re-test it
properly: ALL v6 tickers, REAL Polygon 5-min option bars, entry at the signal bar (next
open if off-hours), exit when the STOCK leg exits, 1%/side cost — and include DEEPER ITM
and LONGER expiries, which are the two levers that cut theta for a slow strategy.

Structures: strike {ATM, 3% ITM, 7% ITM} x expiry {2-3w, 4-6w, 8-12w}.
"""

from __future__ import annotations

import datetime as dt
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb

from alpaca_bot2 import prep, _barriers, MIN_ATR_PCT
from triple_barrier_breadth import TICKERS
from vc_options_real import contracts_near, day_close
from qqq_options_real import bars_for, et_date

TRAIN_END, SIM_END = "2025-07-14", "2026-07-01"
EFF_COST = 5.0 / 1e4
TP, SL, H, SELQ, MINS = 7.0, 1.0, 96, 0.93, 60      # v6's live config
BUCKETS = [("2-3w", 12, 25), ("4-6w", 26, 45), ("8-12w", 50, 90)]
STRIKES = [("ATM", 1.00), ("ITM3", 0.97), ("ITM7", 0.93)]
HAIRCUT = 0.01


def gen_trades(tk):
    ts, h, l, c, A, X, valid, sp, gp = prep(tk, MINS, "trend", "atr")
    sp, gp = _barriers("atr", c, A, TP, SL, sp, gp)
    ok = valid & np.isfinite(A) & (A / np.maximum(c, 1e-9) >= MIN_ATR_PCT)
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        if not ok[i]:
            continue
        for j in range(i + 1, min(i + H + 1, n)):
            if l[j] <= sp[i]:
                y[i] = 0; break
            if h[j] >= gp[i]:
                y[i] = 1; break
    fv = (X.notna().all(axis=1)).to_numpy() & ok
    tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(TRAIN_END)))[0]
    tr = tr[:-H] if len(tr) > H else tr
    if len(tr) < 500 or np.nansum(y[tr]) < 10:
        return []
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SELQ)
    idx = np.where(fv & (ts >= np.datetime64(TRAIN_END))
                   & (ts < np.datetime64(SIM_END)))[0]
    if len(idx) == 0:
        return []
    proba = {int(ix): float(p) for ix, p in
             zip(idx, clf.predict_proba(X.iloc[idx])[:, 1])}
    out = []
    i, last = int(idx[0]), int(idx[-1])
    while i <= last:
        if proba.get(i, -1.0) < thr:
            i += 1; continue
        res, j = None, i + 1
        while j < min(i + H + 1, n):
            if l[j] <= sp[i]:
                res = 0; break
            if h[j] >= gp[i]:
                res = 1; break
            j += 1
        ex = min(j, n - 1)
        S1 = float(gp[i] if res == 1 else (sp[i] if res == 0 else c[ex]))
        out.append(dict(tk=tk, t0=pd.Timestamp(ts[i]), t1=pd.Timestamp(ts[ex]),
                        S0=float(c[i]), S1=S1,
                        stock_ret=(S1 - c[i]) / c[i] - EFF_COST))
        i = ex + 1
    return out, ts, c


def replay(trades, lo, hi, mult, ts_all, c_all):
    fills, skipped = [], 0
    for t in trades:
        d0, d1 = et_date(t["t0"]), et_date(t["t1"])
        tgt_k = t["S0"] * mult
        cons = [x for x in contracts_near(t["tk"], d0, t["S0"] * 0.88,
                                          t["S0"] * 1.05)
                if lo <= (dt.date.fromisoformat(x["exp"]) - d0).days <= hi]
        if not cons:
            skipped += 1; continue
        cons.sort(key=lambda x: (abs(x["K"] - tgt_k),
                                 dt.date.fromisoformat(x["exp"])))
        got = None
        for con in cons[:3]:
            exp = dt.date.fromisoformat(con["exp"])
            bars = bars_for(con["ticker"], str(d0), str(max(d1, exp)))
            if not bars:
                continue
            bt = np.array([b["t"] for b in bars], dtype=np.int64)
            t0ms, t1ms = (int(t["t0"].value // 1_000_000),
                          int(t["t1"].value // 1_000_000))
            i0 = int(np.searchsorted(bt, t0ms))
            if i0 >= len(bars) or bt[i0] - t0ms > 24 * 3600_000:
                continue
            entry = bars[i0]["c"] if bt[i0] == t0ms else bars[i0]["o"]
            if entry <= 0.05:
                continue
            if d1 > exp:
                Sx = day_close(ts_all, c_all, exp)
                if Sx is None:
                    continue
                ex_px = max(Sx - con["K"], 0.0)
            else:
                i1 = int(np.searchsorted(bt, t1ms))
                ex_px = (bars[i1]["c"] if i1 < len(bars) and bt[i1] == t1ms
                         else bars[min(i1, len(bars) - 1)]["o"])
            got = (entry, ex_px)
            break
        if got is None:
            skipped += 1; continue
        e, x = got
        fills.append((x * (1 - HAIRCUT) - e * (1 + HAIRCUT)) / (e * (1 + HAIRCUT)))
    return fills, skipped


def main():
    print(f"v6 OPTIONS — real intraday replay, {TRAIN_END}..{SIM_END}, all v6 tickers")
    allt, tsmap = [], {}
    for tk in TICKERS:
        r = gen_trades(tk)
        if not r:
            continue
        trades, ts_all, c_all = r
        tsmap[tk] = (ts_all, c_all)
        allt += trades
        sys.stdout.flush()
    sr = np.array([t["stock_ret"] for t in allt])
    print(f"\nSTOCK leg: {len(allt)} trades, win {(sr > 0).mean():.0%}, "
          f"avg {sr.mean() * 1e4:+.0f}bps, total {sr.sum() * 100:+.1f}% "
          f"(${sr.sum() * 1000:+,.0f} at $1k/trade)\n")
    print(f"  {'expiry':>7} {'strike':>6} {'n':>4} {'skip':>5} {'win%':>6} "
          f"{'avg/trade':>10} {'total':>9} {'$1k/trade':>11}")
    best = None
    for bname, lo, hi in BUCKETS:
        for sname, mult in STRIKES:
            fills, sk = [], 0
            for tk in tsmap:
                f, s = replay([t for t in allt if t["tk"] == tk], lo, hi, mult,
                              *tsmap[tk])
                fills += f; sk += s
            if len(fills) < 10:
                print(f"  {bname:>7} {sname:>6}  too few fills ({len(fills)})")
                continue
            r = np.array(fills)
            tot = r.sum() * 100
            print(f"  {bname:>7} {sname:>6} {len(r):>4} {sk:>5} {(r > 0).mean():>6.0%} "
                  f"{r.mean() * 100:>+9.1f}% {tot:>+8.0f}% ${r.sum() * 1000:>+10,.0f}")
            if best is None or r.mean() > best[0]:
                best = (r.mean(), bname, sname, lo, hi, mult, len(r))
            sys.stdout.flush()
    if best:
        print(f"\nBEST: {best[1]} {best[2]} -> {best[0] * 100:+.1f}%/trade "
              f"over {best[6]} trades (DTE {best[3]}-{best[4]}, strike x{best[5]})")
        print("Positive => worth trading live. Negative => v6 stays stock-only.")


if __name__ == "__main__":
    main()
