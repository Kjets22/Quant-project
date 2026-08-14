"""
strat_options_test.py — would an OPTIONS twin of a given stock strategy be profitable?

  python strat_options_test.py v4
  python strat_options_test.py v6
  python strat_options_test.py v7

Built after the 2026-08-03 lookahead bug (option legs transacted at the bar START,
55 min before the signal existed, inflating vC's edge by ~32%). Here both legs
transact at the bar CLOSE — the instant the stock leg trades — for ANY bar size.

Every structure is judged on more than a point estimate, because that is exactly
what produced the fake v6 "winner": a cell that was one expiry date, 83% Thursdays,
with 3 trades carrying 188% of the P&L. A structure must clear ALL of:
  n >= 40 fills          (not a survivorship sliver)
  mean > 0 after 1%/side
  t-stat >= 1.5
  bootstrap P(mean>0) >= 90%
  still positive after dropping the best 5 trades
  >= 60% of tickers positive
Anything else is reported as NOT VIABLE, with the reason.
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

from alpaca_bot2 import CONFIGS, MIN_ATR_PCT, prep, _barriers
from qqq_tournament import MODELS as TOURN_MODELS
from vc_options_real import contracts_near, day_close
from qqq_options_real import bars_for, et_date

TRAIN_END, SIM_END = "2025-07-14", "2026-07-01"
EFF_COST, HC = 5.0 / 1e4, 0.01
BUCKETS = [("~1-2w", 5, 14), ("~2-4w", 15, 30), ("~5-8w", 35, 60)]
STYLES = [("ATM", 1.00), ("OTM3", 1.03), ("ITM3", 0.97)]


def cfg_for(name):
    for c in CONFIGS:
        if c[0] == name:
            return c
    raise SystemExit(f"unknown strategy {name}")


def gen_trades(name):
    _, tks, mins, hbar, mode, tp, sl, featmode, sel, _ = cfg_for(name)
    bar = pd.Timedelta(minutes=mins)
    out, tsmap = [], {}
    for tk in tks:
        try:
            ts, h, l, c, A, X, valid, sp, gp = prep(tk, mins, featmode, mode)
        except Exception as e:
            print(f"  [prep warn {tk}: {e}]")
            continue
        sp, gp = _barriers(mode, c, A, tp, sl, sp, gp)
        ok = valid & np.isfinite(A)
        if mode not in ("dollar", "pct"):
            ok &= (A / np.maximum(c, 1e-9) >= MIN_ATR_PCT)
        n = len(c)
        y = np.full(n, np.nan)
        for i in range(n - 1):
            if not ok[i]:
                continue
            for j in range(i + 1, min(i + hbar + 1, n)):
                if l[j] <= sp[i]:
                    y[i] = 0; break
                if h[j] >= gp[i]:
                    y[i] = 1; break
        fv = X.notna().all(axis=1).to_numpy() & ok
        tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(TRAIN_END)))[0]
        tr = tr[:-hbar] if len(tr) > hbar else tr
        if len(tr) < 500 or np.nansum(y[tr]) < 20:
            continue
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8,
                                 colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr].astype(int))
        ptr = clf.predict_proba(X.iloc[tr])[:, 1]
        thr = (0.5 + np.quantile(np.abs(ptr - 0.5), sel[1]) if sel[0] == "conf"
               else np.quantile(ptr, sel[1]))
        idx = np.where(fv & (ts >= np.datetime64(TRAIN_END))
                       & (ts < np.datetime64(SIM_END)))[0]
        if not len(idx):
            continue
        proba = {int(ix): float(p) for ix, p in
                 zip(idx, clf.predict_proba(X.iloc[idx])[:, 1])}
        tsmap[tk] = (ts, c)
        i, last = int(idx[0]), int(idx[-1])
        while i <= last:
            if proba.get(i, -1.0) < thr:
                i += 1; continue
            res, j = None, i + 1
            while j < min(i + hbar + 1, n):
                if l[j] <= sp[i]:
                    res = 0; break
                if h[j] >= gp[i]:
                    res = 1; break
                j += 1
            ex = min(j, n - 1)
            S1 = float(gp[i] if res == 1 else (sp[i] if res == 0 else c[ex]))
            # LOOKAHEAD FIX: ts[] are bar STARTS; the signal is the bar CLOSE
            out.append(dict(tk=tk, t0=pd.Timestamp(ts[i]) + bar,
                            t1=pd.Timestamp(ts[ex]) + bar, S0=float(c[i]), S1=S1,
                            stock_ret=(S1 - c[i]) / c[i] - EFF_COST))
            i = ex + 1
    return out, tsmap


def replay(trades, lo, hi, mult, tsmap):
    fills, skipped = [], 0
    for t in trades:
        d0, d1 = et_date(t["t0"]), et_date(t["t1"])
        want = t["S0"] * mult
        cons = [c for c in contracts_near(t["tk"], d0, t["S0"] * 0.90,
                                          t["S0"] * 1.10, cap=hi + 10)
                if lo <= (dt.date.fromisoformat(c["exp"]) - d0).days <= hi]
        if not cons:
            skipped += 1; continue
        cons.sort(key=lambda c: (abs(c["K"] - want), dt.date.fromisoformat(c["exp"])))
        got = None
        for con in cons[:3]:
            exp = dt.date.fromisoformat(con["exp"])
            bars = bars_for(con["ticker"], str(d0), str(max(d1, exp)))
            if not bars:
                continue
            bt = np.array([b["t"] for b in bars], dtype=np.int64)
            t0ms, t1ms = (int(t["t0"].value // 1e6), int(t["t1"].value // 1e6))
            i0 = int(np.searchsorted(bt, t0ms))
            if i0 >= len(bars) or bt[i0] - t0ms > 30 * 60_000:
                continue                      # no print within 30 min -> illiquid
            entry = bars[i0]["c"] if bt[i0] == t0ms else bars[i0]["o"]
            if entry <= 0.05:
                continue
            ts_all, c_all = tsmap[t["tk"]]
            if d1 > exp:                       # option died first -> intrinsic
                Sx = day_close(ts_all, c_all, exp)
                if Sx is None:
                    continue
                ex_px = max(Sx - con["K"], 0.0)
            else:
                i1 = int(np.searchsorted(bt, t1ms))
                if i1 < len(bars) and bt[i1] - t1ms <= 30 * 60_000:
                    ex_px = bars[i1]["c"] if bt[i1] == t1ms else bars[i1]["o"]
                elif d1 >= exp:
                    Sx = day_close(ts_all, c_all, exp)
                    ex_px = max(Sx - con["K"], 0.0) if Sx else None
                else:
                    ex_px = bars[min(i1, len(bars) - 1)]["c"]
                if ex_px is None:
                    continue
            got = (entry, ex_px, t["tk"])
            break
        if got is None:
            skipped += 1; continue
        fills.append(got)
    return fills, skipped


def judge(fills):
    r = np.array([(x * (1 - HC) - e * (1 + HC)) / (e * (1 + HC)) for e, x, _ in fills])
    tks = [f[2] for f in fills]
    n = len(r)
    if n < 5:
        return r, {"viable": False, "why": f"only {n} fills"}
    sd = r.std(ddof=1)
    t = r.mean() / sd * np.sqrt(n) if sd > 0 else 0.0
    rng = np.random.default_rng(0)
    boot = np.array([rng.choice(r, n, replace=True).mean() for _ in range(4000)])
    pboot = float((boot > 0).mean())
    drop5 = float(np.sort(r)[::-1][5:].sum()) if n > 5 else -1
    byt = {}
    for ret, tk in zip(r, tks):
        byt.setdefault(tk, []).append(ret)
    pos_tk = sum(1 for v in byt.values() if sum(v) > 0)
    frac_tk = pos_tk / max(len(byt), 1)
    fails = []
    if n < 40:
        fails.append(f"n={n}<40")
    if r.mean() <= 0:
        fails.append("mean<=0")
    if t < 1.5:
        fails.append(f"t={t:.2f}<1.5")
    if pboot < 0.90:
        fails.append(f"P(>0)={pboot:.0%}<90%")
    if drop5 <= 0:
        fails.append("negative without top-5")
    if frac_tk < 0.60:
        fails.append(f"{pos_tk}/{len(byt)} tickers positive")
    return r, {"viable": not fails, "why": "; ".join(fails) or "PASSES ALL GATES",
               "t": t, "pboot": pboot, "drop5": drop5, "frac_tk": frac_tk,
               "pos_tk": pos_tk, "ntk": len(byt)}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "v6"
    trades, tsmap = gen_trades(name)
    sr = np.array([t["stock_ret"] for t in trades])
    print(f"\n{name} OPTIONS TEST — {TRAIN_END}..{SIM_END}, lookahead-corrected\n")
    print(f"STOCK leg: {len(trades)} trades, win {(sr > 0).mean():.0%}, "
          f"avg {sr.mean() * 1e4:+.0f}bps, total {sr.sum() * 100:+.1f}%")
    if sr.mean() <= 0:
        print("  !! the STOCK signal itself is negative on this window — an options\n"
              "     twin amplifies a negative expectancy. Reported for completeness.")
    print(f"\n{'expiry':>7} {'strike':>6} {'n':>4} {'skip':>5} {'win%':>5} "
          f"{'mean':>8} {'median':>8} {'t':>6} {'P(>0)':>6} {'drop5':>9}  verdict")
    winners = []
    for bname, lo, hi in BUCKETS:
        for sname, mult in STYLES:
            fills, sk = replay(trades, lo, hi, mult, tsmap)
            r, v = judge(fills)
            if len(r) < 5:
                print(f"{bname:>7} {sname:>6} {len(r):>4} {sk:>5}   -- {v['why']}")
                continue
            print(f"{bname:>7} {sname:>6} {len(r):>4} {sk:>5} {(r > 0).mean():>5.0%} "
                  f"{r.mean() * 100:>+7.1f}% {np.median(r) * 100:>+7.1f}% "
                  f"{v['t']:>+6.2f} {v['pboot']:>6.0%} "
                  f"{v['drop5'] * 100:>+8.0f}%  "
                  f"{'VIABLE' if v['viable'] else 'no: ' + v['why']}")
            if v["viable"]:
                winners.append((bname, sname, r.mean(), v))
    print()
    if winners:
        for b, s, m, v in sorted(winners, key=lambda z: -z[2]):
            print(f"VIABLE: {b} {s} -> {m * 100:+.1f}%/trade, t={v['t']:+.2f}, "
                  f"P(>0)={v['pboot']:.0%}, {v['pos_tk']}/{v['ntk']} tickers positive")
    else:
        print(f"NO VIABLE STRUCTURE for {name} — stays stock-only.")


if __name__ == "__main__":
    main()
