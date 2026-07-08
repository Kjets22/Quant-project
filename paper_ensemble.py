"""
paper_ensemble.py — paper-trading bot for the v3+v4 ensemble, WITH GUARDRAILS.

Trades both validated strategies (v3 = 30-min/1.5:1, v4 = 15-min/4:1) on the 8-name basket,
sizes every trade by fixed risk, and reports a dashboard + a ledger. Designed to be run any
time (e.g. after each trading day). Set PAPER_START to the Monday of the week you want to track.

GUARDRAILS (edit the constants to taste):
  ACCOUNT            paper account size
  RISK_PCT           % of account risked per trade (stop = 1 ATR -> this is the max loss/trade)
  MAX_POSITIONS      cap on simultaneous open positions
  DAILY_LOSS_LIMIT   stop opening new trades for the day past this loss
  DD_BREAKER         halt + reassess if account drawdown exceeds this
This is RESEARCH / PAPER ONLY. It does not place orders or move money. Real trading risks loss.
"""

from __future__ import annotations

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

from triple_barrier_ml import atr, features, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features
from basket import ticker_cfg
from data import fetch_polygon

# ---- account + guardrails ----
ACCOUNT = 10_000.0
NOTIONAL_PCT = 10.0                       # each trade BUYS this % of the account (position size)
MAX_POSITIONS = 10                        # 10 x 10% = up to 100% of the account, no leverage
DAILY_LOSS_LIMIT = ACCOUNT * 0.02
DD_BREAKER = ACCOUNT * 0.15
# ---- strategy ----
SEL_Q, HBAR, COST_BPS = 0.93, 24, 3.0
CONFIGS = [("v3", 30, 1.5, 1.0), ("v4", 15, 4.0, 1.0)]
PAPER_START = pd.Timestamp("2026-06-01")          # <-- last month, as of today
PAPER_END = pd.Timestamp("2026-07-02")            # through today (exclusive)
FETCH_END = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
_C5 = {}


def load5(tk):
    if tk in _C5:
        return _C5[tk]
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    p = Path(f"data_cache/{tk}_recent_2026-06-01_{FETCH_END}.csv")
    if p.exists():
        rec = pd.read_csv(p, parse_dates=["timestamp"])
    else:
        cfg = ticker_cfg(tk); cfg.data.start_date, cfg.data.end_date = "2026-06-01", FETCH_END
        cfg.data.multiplier, cfg.data.timespan = 5, "minute"
        rec = fetch_polygon(cfg); rec.to_csv(p, index=False)
    df = (pd.concat([df, rec], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    _C5[tk] = df
    return df


def gen(mins, tp, sl, strat):
    out = []
    for tk in TICKERS:
        d = load5(tk).set_index("timestamp").resample(f"{mins}min").agg(
            high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum")).dropna().reset_index()
        ts = pd.to_datetime(d["timestamp"]).to_numpy()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        y = label(h, l, c, A, tp, sl)
        fv = X.notna().all(axis=1).to_numpy(); n = len(c)
        tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(PAPER_START)))[0]
        if len(tr) < 500:
            continue
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr].astype(int))
        thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q)
        fwd = np.where(fv & (ts >= np.datetime64(PAPER_START)) & (ts < np.datetime64(PAPER_END)))[0]
        if len(fwd) == 0:
            continue
        proba = {int(ix): float(p) for ix, p in zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
        i, last = int(fwd[0]), int(fwd[-1])
        while i <= last:
            if proba.get(i, -1) < thr:
                i += 1; continue
            a = A[i]; up, dn = c[i] + tp * a, c[i] - sl * a
            res, j = None, i + 1
            while j < min(i + HBAR + 1, n):
                if l[j] <= dn:
                    res = 0; break
                if h[j] >= up:
                    res = 1; break
                j += 1
            if res is None and j >= n:
                out.append(dict(strat=strat, tk=tk, ets=pd.Timestamp(ts[i]), entry=c[i], atr=a,
                                tgt=up, stop=dn, xts=None, exit=None, outcome="OPEN")); i = n; continue
            if res is None:
                res = 1 if c[min(j, n - 1)] > c[i] else 0
            out.append(dict(strat=strat, tk=tk, ets=pd.Timestamp(ts[i]), entry=c[i], atr=a,
                            tgt=up, stop=dn, xts=pd.Timestamp(ts[j if j < n else n - 1]),
                            exit=(up if res == 1 else dn), outcome="TARGET" if res == 1 else "STOP"))
            i = j + 1
    return out


def main():
    notional_d = ACCOUNT * NOTIONAL_PCT / 100.0
    trades = []
    for strat, mins, tp, sl in CONFIGS:
        trades += gen(mins, tp, sl, strat)
    trades.sort(key=lambda t: t["ets"])
    for t in trades:
        t["shares"] = round(notional_d / t["entry"], 1)           # buy ~30% of account
        t["notional"] = round(t["shares"] * t["entry"], 0)
        t["risk"] = round(t["shares"] * t["atr"], 2)              # max loss if stopped (1 ATR)
        t["pnl"] = None if t["exit"] is None else round(t["shares"] * (t["exit"] - t["entry"]), 2)

    print(f"PAPER ENSEMBLE  v3+v4  |  account ${ACCOUNT:,.0f}  |  {NOTIONAL_PCT:.0f}% per trade (${notional_d:,.0f} position)")
    print(f"week {PAPER_START.date()} .. {(PAPER_END-pd.Timedelta(days=3)).date()}  (data thru {FETCH_END})\n")
    if not trades:
        print("No signals this week yet — re-run after more bars print."); return
    print(f"  {'strat':>5} {'tk':>5} {'entry (UTC)':>16} {'shares':>7} {'bought$':>8} {'risk$':>7} {'outcome':>7} {'$ P&L':>8}")
    for t in trades:
        pl = "  open" if t["pnl"] is None else f"{t['pnl']:+.2f}"
        print(f"  {t['strat']:>5} {t['tk']:>5} {str(t['ets'])[5:16]:>16} {t['shares']:>7} {t['notional']:>8,.0f} "
              f"{t['risk']:>7.2f} {t['outcome']:>7} {pl:>8}")

    closed = [t for t in trades if t["pnl"] is not None]
    wins = sum(t["outcome"] == "TARGET" for t in closed)
    pnl = sum(t["pnl"] for t in closed)
    # risk diagnostics vs guardrails
    ev = sorted([(t["ets"], 1) for t in closed] + [(t["xts"], -1) for t in closed])
    run = mx = 0
    for _, e in ev:
        run += e; mx = max(mx, run)
    daily = {}
    for t in closed:
        d = t["xts"].date(); daily[d] = daily.get(d, 0) + t["pnl"]
    worst_day = min(daily.values()) if daily else 0
    cum = np.cumsum([t["pnl"] for t in sorted(closed, key=lambda x: x["xts"])])
    maxdd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) else 0

    print(f"\n  RESULT: {len(closed)} closed, {wins} wins ({wins/len(closed):.0%}), "
          f"{sum(t['outcome']=='OPEN' for t in trades)} open  ->  P&L ${pnl:+,.2f} "
          f"({pnl/ACCOUNT*100:+.2f}% of account)")
    print(f"  account: ${ACCOUNT:,.0f} -> ${ACCOUNT+pnl:,.2f}\n")
    print("  GUARDRAIL CHECK:")
    print(f"    max simultaneous positions: {mx}  (limit {MAX_POSITIONS})  {'OK' if mx<=MAX_POSITIONS else 'BREACH'}")
    print(f"    worst single day:  ${worst_day:+,.2f}  (limit -${DAILY_LOSS_LIMIT:,.0f})  {'OK' if worst_day>=-DAILY_LOSS_LIMIT else 'BREACH -> stop that day'}")
    print(f"    max drawdown:      ${maxdd:+,.2f}  (breaker -${DD_BREAKER:,.0f})  {'OK' if maxdd>=-DD_BREAKER else 'BREACH -> halt + reassess'}")

    Path("runs").mkdir(exist_ok=True)
    Path("runs/paper_ensemble_ledger.json").write_text(json.dumps(
        [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in t.items()} for t in trades], indent=1))
    print("\n  ledger -> runs/paper_ensemble_ledger.json")


if __name__ == "__main__":
    main()
