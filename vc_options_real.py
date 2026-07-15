"""
vc_options_real.py — the REAL intraday options test for vC (30xATR / 3xATR drift-rider).

Upgrades the daily-chain result (+6-12%/trade on SPY) to actual traded 5-min option
prices, on ALL 8 tickers vC trades, over the final year:
  - vC signals regenerated per ticker (model trained strictly before the window),
  - BUY the call at the first real option bar at/after the signal (signals in extended
    hours -> next market open; counted as 'delayed'),
  - SELL at the first real option bar at/after the stock's target/stop/time exit
    (intrinsic at expiry if the option died first),
  - expiries ~1-2w and ~3-6w, strikes ATM and +3% OTM, spread haircut 0% / 1% per side.

Uses the existing Polygon key; everything cached in data_cache/options/.
Research only; places no orders.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb

from alpaca_bot2 import prep, _barriers, MIN_ATR_PCT
from triple_barrier_breadth import TICKERS
import options_data_polygon as odp
from qqq_options_real import bars_for, et_date

TRAIN_END = "2025-07-14"
SIM_END = "2026-07-01"
EFF_COST = 5.0 / 1e4
TP, SL, H, SELQ = 30.0, 3.0, 96, 0.93
BUCKETS = [("~1-2w", 5, 14), ("~3-6w", 15, 45)]
STYLES = ["ATM", "OTM3"]
HAIRCUTS = [0.0, 0.01]
CACHE = Path("data_cache/options")


def gen_trades(tk):
    ts, h, l, c, A, X, valid, sp, gp = prep(tk, 60, "trend", "atr")
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
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SELQ)
    idx = np.where(fv & (ts >= np.datetime64(TRAIN_END)) & (ts < np.datetime64(SIM_END)))[0]
    if len(idx) == 0:
        return []
    proba = {int(ix): float(p) for ix, p in
             zip(idx, clf.predict_proba(X.iloc[idx])[:, 1])}
    trades = []
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
        trades.append(dict(tk=tk, t0=pd.Timestamp(ts[i]), t1=pd.Timestamp(ts[ex]),
                           S0=float(c[i]), S1=S1,
                           stock_ret=(S1 - c[i]) / c[i] - EFF_COST))
        i = ex + 1
    return trades, ts, c


_CONS = {}
def contracts_near(ul, day, klo, khi):
    key = f"{ul}_{day}_{int(klo)}_{int(khi)}"
    cf = CACHE / f"cons_{key}.json"
    if key in _CONS:
        return _CONS[key]
    if cf.exists():
        _CONS[key] = json.loads(cf.read_text()); return _CONS[key]
    s, k = odp._session_key()
    out = []
    for expired in ("true", "false"):
        url = "https://api.polygon.io/v3/reference/options/contracts"
        params = {"underlying_ticker": ul, "contract_type": "call",
                  "expiration_date.gte": str(day),
                  "expiration_date.lte": str(day + dt.timedelta(days=50)),
                  "strike_price.gte": klo, "strike_price.lte": khi,
                  "expired": expired, "limit": 1000}
        while True:
            j = odp._get(s, k, url, params)
            out += [{"ticker": r["ticker"], "K": float(r["strike_price"]),
                     "exp": r["expiration_date"]} for r in j.get("results", [])]
            nxt = j.get("next_url")
            if not nxt:
                break
            url, params = nxt, {}
    cf.write_text(json.dumps(out))
    _CONS[key] = out
    return out


def day_close(ts, c, day):
    m = np.array([et_date(pd.Timestamp(t)) == day for t in ts])
    if not m.any():
        return None
    return float(c[np.where(m)[0][-1]])


def replay(trades, lo, hi, style, ts_all, c_all):
    out, skipped, delayed = [], 0, 0
    for t in trades:
        d0, d1 = et_date(t["t0"]), et_date(t["t1"])
        tgt_k = t["S0"] * (1.03 if style == "OTM3" else 1.0)
        cons = contracts_near(t["tk"], d0, t["S0"] * 0.94, t["S0"] * 1.08)
        cand = [c for c in cons
                if lo <= (dt.date.fromisoformat(c["exp"]) - d0).days <= hi]
        if not cand:
            skipped += 1; continue
        cand.sort(key=lambda c: (abs(c["K"] - tgt_k), dt.date.fromisoformat(c["exp"])))
        filled = None
        for con in cand[:3]:
            exp = dt.date.fromisoformat(con["exp"])
            bars = bars_for(con["ticker"], str(d0), str(max(d1, exp)))
            if not bars:
                continue
            bt = np.array([b["t"] for b in bars], dtype=np.int64)
            t0ms = int(t["t0"].value // 1_000_000)
            t1ms = int(t["t1"].value // 1_000_000)
            i0 = int(np.searchsorted(bt, t0ms))
            if i0 >= len(bars) or bt[i0] - t0ms > 24 * 3600_000:
                continue                          # nothing within a day -> illiquid
            was_delayed = bt[i0] - t0ms > 30 * 60_000
            entry = bars[i0]["c"] if bt[i0] == t0ms else bars[i0]["o"]
            if entry <= 0.03:
                continue
            if d1 > exp:                          # option died before the stock exit
                Sx = day_close(ts_all, c_all, exp)
                if Sx is None:
                    continue
                exit_px = max(Sx - con["K"], 0.0)
            else:
                i1 = int(np.searchsorted(bt, t1ms))
                if i1 < len(bars):
                    exit_px = bars[i1]["c"] if bt[i1] == t1ms else bars[i1]["o"]
                elif d1 >= exp:
                    Sx = day_close(ts_all, c_all, exp)
                    exit_px = max(Sx - con["K"], 0.0) if Sx else None
                else:
                    exit_px = bars[-1]["c"]       # after last print -> last real price
                if exit_px is None:
                    continue
            filled = (entry, exit_px, t["stock_ret"])
            if was_delayed:
                delayed += 1
            break
        if filled is None:
            skipped += 1; continue
        out.append(filled)
    return out, skipped, delayed


def main():
    print(f"vC OPTIONS — REAL intraday replay (Polygon 5-min traded prices), "
          f"{TRAIN_END}..{SIM_END}, all vC tickers")
    all_trades, tsmap = [], {}
    for tk in TICKERS:
        r = gen_trades(tk)
        if not r:
            continue
        trades, ts_all, c_all = r
        tsmap[tk] = (ts_all, c_all)
        all_trades += trades
        sys.stdout.flush()
    sr = np.array([t["stock_ret"] for t in all_trades])
    print(f"\nALL TICKERS: {len(all_trades)} vC stock trades | STOCK: "
          f"win {(sr > 0).mean():.0%} avg {sr.mean() * 1e4:+.0f}bps "
          f"total {sr.sum() * 100:+.1f}%  ($1k/trade => ${sr.sum() * 1000:+,.0f})")
    print(f"  {'expiry':>6} {'strike':>6} {'cost':>5} {'n':>4} {'skip':>4} {'dly':>4} "
          f"{'win%':>5} {'avg/trade':>10} {'median':>8} {'total':>8} {'$1k/trade':>11}")
    for bname, lo, hi in BUCKETS:
        for style in STYLES:
            fills, skipped, delayed = [], 0, 0
            for tk in TICKERS:
                if tk not in tsmap:
                    continue
                tt = [t for t in all_trades if t["tk"] == tk]
                f, s, d = replay(tt, lo, hi, style, *tsmap[tk])
                fills += f; skipped += s; delayed += d
            if len(fills) < 10:
                print(f"  {bname:>6} {style:>6}   too few fills (n={len(fills)})")
                continue
            for hc in HAIRCUTS:
                r = np.array([(x * (1 - hc) - e * (1 + hc)) / (e * (1 + hc))
                              for e, x, _ in fills])
                print(f"  {bname:>6} {style:>6} {hc * 100:>4.1f}% {len(r):>4} "
                      f"{skipped:>4} {delayed:>4} {(r > 0).mean():>5.0%} "
                      f"{r.mean() * 100:>+9.1f}% {np.median(r) * 100:>+7.1f}% "
                      f"{r.sum() * 100:>+7.0f}% ${r.sum() * 1000:>+9,.0f}")
            sys.stdout.flush()
    print("\n'dly' = entries where the signal fired off-hours and the option was bought")
    print("at the next market open. Fills are real traded 5-min option prices.")


if __name__ == "__main__":
    main()
