"""
options_deep_dive.py — which stable strategies port to OPTIONS (long calls), and at
what expiry? REAL SPY chain (actual ask-in / bid-out), no Black-Scholes fantasy fills.

Part 1 — real replay: v3/v4/v6/v7/vC signals on SPY over the chain window
(2025-04-01..2026-06-11, model trained strictly before). Each stock trade is re-run as
a call purchase: entry = signal-day ASK, exit = exit-day BID (intrinsic if expired).
Grid: DTE buckets {4-11, 12-25, 26-59, 60-120} x strikes {ITM d~.88, ATM, OTM +3%}.
Same-day round trips (chain is one snapshot/day) use a delta-approximation and are
counted separately.

Part 2 — measured cost structure: ATM premium %, spread % of mid, delta by DTE bucket,
then each QQQ strategy's (win%, +move, -move) run through that real cost arithmetic.

Research only; places no orders. Standalone; touches no frozen snapshot.
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

from alpaca_bot2 import prep, _barriers, MIN_ATR_PCT
from qqq_tournament import MODELS as TOURN_MODELS

WIN0, WIN1 = pd.Timestamp("2025-04-01"), pd.Timestamp("2026-06-11")
EFF_COST = 5.0 / 1e4
DTE_BUCKETS = [("~1w", 4, 11), ("~2-3w", 12, 25), ("~1-2m", 26, 59), ("~2-4m", 60, 120)]
STYLES = ["ITM88", "ATM", "OTM3"]
#           name  mins hbar  mode     tp    sl  feat     model
SPY_STRATS = [("v3", 30, 24, "atr",   1.5, 1.0, "sr",    "lgbm"),
              ("v4", 15, 24, "atr",   4.0, 1.0, "sr",    "lgbm"),
              ("v6", 60, 96, "atr",   7.0, 1.0, "trend", "lgbm"),
              ("v7", 60, 96, "struct", 10.0, 1.0, "trend", "lgbm"),
              ("vC", 60, 96, "atr",  30.0, 3.0, "trend", "lgbm")]
# QQQ family: (name, win%, +move%, -move%, hold_hours, trades/yr) from final-year stats
QQQ_STRATS = [("vQ",  0.613, 0.28, 0.28, 1, 106), ("vQ2", 0.684, 0.35, 0.28, 2, 19),
              ("vA",  0.686, 0.21, 0.28, 4, 172), ("vP",  0.595, 0.28, 0.28, 8, 279),
              ("vR",  0.513, 0.40, 0.20, 2, 150), ("vS",  0.525, 0.50, 0.40, 8, 278)]


def load_chain():
    o = pd.read_parquet("data_cache/options/chain_SPY_2025-04-01_2026-06-20.parquet")
    o = o[(o["type"] == "C") & (o["bid"] > 0) & (o["ask"] > 0.05)].copy()
    o["d"] = o["date"].dt.normalize()
    by_day = {dd: g for dd, g in o.groupby("d")}
    quote = {(r.d, r.expiry, r.strike): (float(r.bid), float(r.ask)) for r in o.itertuples()}
    return by_day, quote


def gen_trades(strat, mins, hbar, mode, tp, sl, featmode):
    """Non-overlapping SPY trades in the chain window; model trained strictly before."""
    ts, h, l, c, A, X, valid, stop_px, tgt_px = prep("SPY", mins, featmode, mode)
    stop_px, tgt_px = _barriers(mode, c, A, tp, sl, stop_px, tgt_px)
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        if not valid[i]:
            continue
        for j in range(i + 1, min(i + hbar + 1, n)):
            if l[j] <= stop_px[i]:
                y[i] = 0; break
            if h[j] >= tgt_px[i]:
                y[i] = 1; break
    fv = (X.notna().all(axis=1) & np.isfinite(A) & valid).to_numpy()
    tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(WIN0)))[0]
    tr = tr[:-hbar] if len(tr) > hbar else tr
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], 0.93)
    idx = np.where(fv & (ts >= np.datetime64(WIN0)) & (ts < np.datetime64(WIN1)))[0]
    proba = {int(ix): float(p) for ix, p in
             zip(idx, clf.predict_proba(X.iloc[idx])[:, 1])}
    trades = []
    i, last = int(idx[0]), int(idx[-1])
    while i <= last:
        if (proba.get(i, -1.0) < thr or A[i] / c[i] < MIN_ATR_PCT):
            i += 1; continue
        res, j = None, i + 1
        while j < min(i + hbar + 1, n):
            if l[j] <= stop_px[i]:
                res = 0; break
            if h[j] >= tgt_px[i]:
                res = 1; break
            j += 1
        ex = min(j, n - 1)
        S_exit = float(tgt_px[i] if res == 1 else (stop_px[i] if res == 0 else c[ex]))
        trades.append(dict(strat=strat, ets=pd.Timestamp(ts[i]), xts=pd.Timestamp(ts[ex]),
                           S0=float(c[i]), S1=S_exit,
                           stock_ret=(S_exit - c[i]) / c[i] - EFF_COST))
        i = ex + 1
    return trades


def pick(g, dd, lo, hi, style, S0):
    dte = (g["expiry"] - dd).dt.days
    cand = g[(dte >= lo) & (dte <= hi)]
    if style == "ITM88":
        cand = cand[cand["delta"].between(0.78, 0.96)]
        if not len(cand):
            return None
        return cand.iloc[(cand["delta"] - 0.88).abs().to_numpy().argmin()]
    tgt_k = S0 * (1.03 if style == "OTM3" else 1.0)
    if not len(cand):
        return None
    # nearest strike at the nearest usable expiry
    e0 = sorted(cand["expiry"].unique())[0]
    ce = cand[cand["expiry"] == e0]
    return ce.iloc[(ce["strike"] - tgt_k).abs().to_numpy().argmin()]


def replay(trades, by_day, quote, lo, hi, style):
    rets, sameday = [], 0
    for t in trades:
        dd, ed = t["ets"].normalize(), t["xts"].normalize()
        g = by_day.get(dd)
        if g is None:
            continue
        row = pick(g, dd, lo, hi, style, t["S0"])
        if row is None:
            continue
        ask, bid = float(row["ask"]), float(row["bid"])
        if dd == ed:                                    # same-day round trip: delta approx
            dS = t["S1"] - float(row["underlying_price"])
            exit_px = max(bid + float(row["delta"]) * dS, 0.0)
            rets.append((exit_px - ask) / ask)
            sameday += 1
            continue
        exq = quote.get((ed, row["expiry"], row["strike"]))
        if exq is not None:
            exit_px = exq[0]
        elif ed >= row["expiry"]:
            # option died before the stock trade ended: settle at intrinsic using the
            # underlying price AT EXPIRY (not the later stock exit the option never saw)
            S_exp = None
            for k in range(0, 6):
                g2 = by_day.get(row["expiry"] - pd.Timedelta(days=k))
                if g2 is not None:
                    S_exp = float(g2["underlying_price"].iloc[0]); break
            if S_exp is None:
                continue
            exit_px = max(S_exp - float(row["strike"]), 0.0)
        else:                                           # missing snapshot: nearest later day
            exit_px = None
            for k in range(1, 4):
                exq = quote.get((ed + pd.Timedelta(days=k), row["expiry"], row["strike"]))
                if exq is not None:
                    exit_px = exq[0]; break
            if exit_px is None:
                continue
        rets.append((exit_px - ask) / ask)
    return rets, sameday


def part1():
    by_day, quote = load_chain()
    print("=" * 78)
    print("PART 1 — REAL replay on SPY (ask in / bid out), window "
          f"{WIN0.date()}..{WIN1.date()}")
    print("=" * 78)
    for strat, mins, hbar, mode, tp, sl, feat, _m in SPY_STRATS:
        trades = gen_trades(strat, mins, hbar, mode, tp, sl, feat)
        if not trades:
            print(f"\n{strat}: no signals in window"); continue
        sr = np.array([t["stock_ret"] for t in trades])
        print(f"\n{strat}  ({tp}x/{sl}x {mode}, {mins}-min, H={hbar}) — "
              f"{len(trades)} SPY trades | STOCK: win {(sr > 0).mean():.0%}, "
              f"avg {sr.mean() * 1e4:+.0f}bps, total {sr.sum() * 100:+.1f}%")
        print(f"  {'expiry':>7} {'strike':>6} {'n':>4} {'win%':>6} {'avg/trade':>10} "
              f"{'total':>8} {'sameday':>8}")
        for bname, lo, hi in DTE_BUCKETS:
            for style in STYLES:
                r, sd = replay(trades, by_day, quote, lo, hi, style)
                if len(r) < 5:
                    continue
                r = np.array(r)
                print(f"  {bname:>7} {style:>6} {len(r):>4} {(r > 0).mean():>6.0%} "
                      f"{r.mean() * 100:>+9.1f}% {r.sum() * 100:>+7.0f}% {sd:>8}")


def part2():
    by_day, _ = load_chain()
    print("\n" + "=" * 78)
    print("PART 2 — measured ATM cost structure (real chain medians) -> QQQ-family math")
    print("=" * 78)
    rows = []
    for bname, lo, hi in DTE_BUCKETS:
        prem, spr, dlt = [], [], []
        for dd, g in by_day.items():
            S = float(g["underlying_price"].iloc[0])
            row = pick(g, dd, lo, hi, "ATM", S)
            if row is None:
                continue
            mid = (row["bid"] + row["ask"]) / 2
            prem.append(mid / S * 100)
            spr.append((row["ask"] - row["bid"]) / mid * 100)
            dlt.append(float(row["delta"]))
        rows.append((bname, np.median(prem), np.median(spr), np.median(dlt)))
        print(f"  {bname:>6}: ATM premium {np.median(prem):.2f}% of spot | "
              f"spread {np.median(spr):.1f}% of mid | delta {np.median(dlt):.2f}")
    print(f"\n  QQQ family through that cost structure (expected % of premium per trade):")
    print(f"  {'strat':>5} {'edge(spot)':>10} | " +
          " | ".join(f"{b[0]:>7}" for b in rows))
    for name, pw, up, dn, hold_h, tpy in QQQ_STRATS:
        edge_spot = pw * up - (1 - pw) * dn              # % of spot per trade, pre-cost
        line = f"  {name:>5} {edge_spot:>+9.3f}% | "
        cells = []
        for bname, prem, spr, dlt in rows:
            # option edge = delta * spot-edge, as % of premium, minus the full spread
            # (theta for these short holds is second-order vs the spread; noted)
            per = (dlt * edge_spot / prem * 100) - spr
            cells.append(f"{per:>+6.1f}%")
        print(line + " | ".join(cells))
    print("\n  NOTE: spot-edge uses each strategy's final-year win%/geometry; costs are")
    print("  REAL measured medians (SPY chain as proxy for QQQ). Positive = the edge")
    print("  survives the spread at that expiry; theta and IV-crush would take more.")


if __name__ == "__main__":
    part1()
    part2()
