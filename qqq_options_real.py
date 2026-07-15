"""
qqq_options_real.py — do vQ / vQ2 / vR port to OPTIONS? REAL intraday QQQ option data.

The user's hypothesis: short holds (1-2h) + leverage => options should multiply returns.
Test with REAL traded option prices (Polygon 5-min aggregates on actual QQQ contracts):
  - generate each strategy's stock trades over the final year (model trained before),
  - at the signal bar, BUY the ATM call at that bar's real traded price ("at market"),
  - when the STOCK hits its target/stop/time exit, SELL the option at that bar's real
    traded price (intrinsic at expiry if the option died first),
  - expiries: 0-2 days, ~1 week, ~2-3 weeks; spread sensitivity 0% and 1% per side.

Uses the existing Polygon key (same one that built the SPY chain). Everything cached in
data_cache/options/. Research only; places no orders.
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

from alpaca_bot2 import prep, _barriers
from qqq_tournament import MODELS as TOURN_MODELS
import options_data_polygon as odp

TRAIN_END = "2025-07-14"
SIM_END = "2026-07-01"
EFF_COST = 5.0 / 1e4
BUCKETS = [("0-2d", 0, 2), ("~1w", 3, 9), ("~2-3w", 10, 21)]
HAIRCUTS = [0.0, 0.01]                       # per-side, fraction of option price
#             name   mode      tp     sl    H  gate
STRATS = [("vQ",  "dollar", 2.0,   2.0,  12, ("conf", 0.90), "lgbm"),
          ("vQ2", "dollar", 2.5,   2.0,  24, ("conf", 0.90), "histgb"),
          ("vR",  "pct",    0.004, 0.002, 24, ("q", 0.97),   "lgbm")]
CACHE = Path("data_cache/options")
AGGS = CACHE / "aggs5"
AGGS.mkdir(parents=True, exist_ok=True)


def gen_trades(name, mode, tp, sl, H, sel, mname, rth_only=False):
    ts, h, l, c, A, X, valid, sp, gp = prep("QQQ", 5, "full", mode)
    sp, gp = _barriers(mode, c, A, tp, sl, sp, gp)
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        for j in range(i + 1, min(i + H + 1, n)):
            if l[j] <= sp[i]:
                y[i] = 0; break
            if h[j] >= gp[i]:
                y[i] = 1; break
    fv = (X.notna().all(axis=1) & np.isfinite(A)).to_numpy()
    tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(TRAIN_END)))[0][:-H]
    clf = (TOURN_MODELS["histgb"]() if mname == "histgb" else
           lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                              min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                              reg_lambda=1.0, verbose=-1))
    clf.fit(X.iloc[tr], y[tr].astype(int))
    ptr = clf.predict_proba(X.iloc[tr])[:, 1]
    if rth_only:                      # gate re-based on market-hours bars only
        ptr = ptr[rth_mask(ts[tr])]
    thr = (0.5 + np.quantile(np.abs(ptr - 0.5), sel[1]) if sel[0] == "conf"
           else np.quantile(ptr, sel[1]))
    idx = np.where(fv & (ts >= np.datetime64(TRAIN_END)) & (ts < np.datetime64(SIM_END)))[0]
    proba = {int(ix): float(p) for ix, p in zip(idx, clf.predict_proba(X.iloc[idx])[:, 1])}
    trades = []
    rmask = rth_mask(ts) if rth_only else None
    i, last = int(idx[0]), int(idx[-1])
    while i <= last:
        if proba.get(i, -1.0) < thr or (rth_only and not rmask[i]):
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
        trades.append(dict(strat=name, t0=pd.Timestamp(ts[i]), t1=pd.Timestamp(ts[ex]),
                           S0=float(c[i]), S1=S1,
                           stock_ret=(S1 - c[i]) / c[i] - EFF_COST))
        i = ex + 1
    return trades, ts, c


from zoneinfo import ZoneInfo
_ET = ZoneInfo("America/New_York")


def to_et(t):
    return t.tz_localize("UTC").astimezone(_ET)


def et_date(t):
    return to_et(t).date()


def is_rth(t):
    """Entry falls when the options market is open (9:30-16:00 ET)."""
    et = to_et(t)
    return (et.hour, et.minute) >= (9, 30) and et.hour < 16


def rth_mask(ts):
    idx = pd.DatetimeIndex(ts).tz_localize("UTC").tz_convert(_ET)
    m = idx.hour * 60 + idx.minute
    return np.asarray((m >= 570) & (m < 960))


_CONS = {}
def contracts_near(day, klo, khi):
    """All QQQ calls expiring within 25 days of `day`, strikes in [klo,khi]."""
    key = f"{day}_{int(klo)}_{int(khi)}"
    cf = CACHE / f"qqq_cons_{key}.json"
    if key in _CONS:
        return _CONS[key]
    if cf.exists():
        _CONS[key] = json.loads(cf.read_text()); return _CONS[key]
    s, k = odp._session_key()
    out = []
    for expired in ("true", "false"):
        url = "https://api.polygon.io/v3/reference/options/contracts"
        params = {"underlying_ticker": "QQQ", "contract_type": "call",
                  "expiration_date.gte": str(day),
                  "expiration_date.lte": str(day + dt.timedelta(days=25)),
                  "strike_price.gte": klo, "strike_price.hi": khi,
                  "strike_price.lte": khi, "expired": expired, "limit": 1000}
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


def bars_for(ticker, d0, d1):
    cf = AGGS / f"{ticker.replace(':', '_')}_{d0}_{d1}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    s, k = odp._session_key()
    try:
        j = odp._get(s, k,
                     f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{d0}/{d1}",
                     {"limit": 50000, "adjusted": "true"})
    except RuntimeError:
        return []
    bars = j.get("results", []) or []
    cf.write_text(json.dumps(bars))
    return bars


def day_close(ts, c, day):
    """Last stock price on `day` (ET) at/before 20:00 UTC — for expiry settlement."""
    m = np.array([et_date(pd.Timestamp(t)) == day for t in ts])
    if not m.any():
        return None
    return float(c[np.where(m)[0][-1]])


def replay(trades, lo, hi, ts_all, c_all):
    """Buy ATM call (DTE in [lo,hi]) at the entry bar's real price; sell at exit bar."""
    out, skipped = [], 0
    for t in trades:
        d0, d1 = et_date(t["t0"]), et_date(t["t1"])
        cons = contracts_near(d0, t["S0"] * 0.97, t["S0"] * 1.03)
        cand = [c for c in cons
                if lo <= (dt.date.fromisoformat(c["exp"]) - d0).days <= hi]
        if not cand:
            skipped += 1; continue
        cand.sort(key=lambda c: (abs(c["K"] - t["S0"]),
                                 dt.date.fromisoformat(c["exp"])))
        filled = None
        for con in cand[:3]:                     # nearest 3 strikes before giving up
            exp = dt.date.fromisoformat(con["exp"])
            bars = bars_for(con["ticker"], str(d0), str(max(d1, exp)))
            if not bars:
                continue
            bt = np.array([b["t"] for b in bars], dtype=np.int64)
            t0ms = int(t["t0"].value // 1_000_000)
            t1ms = int(t["t1"].value // 1_000_000)
            i0 = int(np.searchsorted(bt, t0ms))
            if i0 >= len(bars) or bt[i0] - t0ms > 30 * 60_000:
                continue                          # no trades near entry -> too illiquid
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
                if i1 >= len(bars):
                    i1 = len(bars) - 1            # after last trade -> last real price
                exit_px = bars[i1]["c"] if bt[i1] <= t1ms + 30 * 60_000 else None
                if exit_px is None:
                    nxt = [b for b in bars[i1:] if b["t"] > t1ms]
                    if nxt:
                        exit_px = nxt[0]["o"]
                    elif d1 >= exp:
                        Sx = day_close(ts_all, c_all, exp)
                        exit_px = max(Sx - con["K"], 0.0) if Sx else None
                if exit_px is None:
                    continue
            filled = (entry, exit_px, t["stock_ret"])
            break
        if filled is None:
            skipped += 1; continue
        out.append(filled)
    return out, skipped


def main():
    print("QQQ OPTIONS — REAL intraday replay (Polygon 5-min traded prices), "
          f"{TRAIN_END}..{SIM_END}")
    for name, mode, tp, sl, H, sel, mname in STRATS:
        trades, ts_all, c_all = gen_trades(name, mode, tp, sl, H, sel, mname)
        sr = np.array([t["stock_ret"] for t in trades])
        print(f"\n{name}: {len(trades)} stock trades | STOCK: win {(sr > 0).mean():.0%} "
              f"avg {sr.mean() * 1e4:+.1f}bps total {sr.sum() * 100:+.2f}%  "
              f"($1k/trade => ${sr.sum() * 1000:+,.0f})")
        rth = [t for t in trades if is_rth(t["t0"])]
        print(f"  !! only {len(rth)}/{len(trades)} signals fire while the options market "
              f"is open ({len(rth) / max(len(trades), 1):.0%}) — the rest are "
              f"pre/post-market, options CANNOT be bought there")
        if len(rth) < 5:
            # as-is the strategy can't trade options at all; test the closest thing
            # that could: same model, signals restricted to market hours
            trades = gen_trades(name, mode, tp, sl, H, sel, mname, rth_only=True)[0]
            if len(trades) < 5:
                print("  RTH-only variant also produces too few trades — options N/A")
                continue
            sr = np.array([t["stock_ret"] for t in trades])
            print(f"  RTH-ONLY VARIANT (gate re-based on market-hours bars): "
                  f"{len(trades)} trades | STOCK: win {(sr > 0).mean():.0%} "
                  f"avg {sr.mean() * 1e4:+.1f}bps total {sr.sum() * 100:+.2f}%")
        else:
            sr = np.array([t["stock_ret"] for t in rth])
            print(f"  market-hours subset STOCK: win {(sr > 0).mean():.0%} "
                  f"avg {sr.mean() * 1e4:+.1f}bps total {sr.sum() * 100:+.2f}%")
            trades = rth
        print(f"  {'expiry':>6} {'cost':>6} {'n':>4} {'skip':>4} {'win%':>5} "
              f"{'avg/trade':>10} {'median':>8} {'total':>8} {'$1k/trade P&L':>14}")
        for bname, lo, hi in BUCKETS:
            fills, skipped = replay(trades, lo, hi, ts_all, c_all)
            if len(fills) < 5:
                print(f"  {bname:>6}    too few fills (n={len(fills)}, skip={skipped})")
                continue
            for hc in HAIRCUTS:
                r = np.array([(x * (1 - hc) - e * (1 + hc)) / (e * (1 + hc))
                              for e, x, _ in fills])
                print(f"  {bname:>6} {hc * 100:>5.1f}% {len(r):>4} {skipped:>4} "
                      f"{(r > 0).mean():>5.0%} {r.mean() * 100:>+9.1f}% "
                      f"{np.median(r) * 100:>+7.1f}% {r.sum() * 100:>+7.0f}% "
                      f"${r.sum() * 1000:>+12,.0f}")
        sys.stdout.flush()
    print("\nfills are REAL traded 5-min option prices (entry at the signal bar, exit at")
    print("the stock's target/stop bar); 'cost' = extra per-side haircut for the spread.")
    print("$1k/trade puts $1,000 of premium on every signal (fractional contracts).")


if __name__ == "__main__":
    main()
