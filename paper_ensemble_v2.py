"""
paper_ensemble_v2.py — IMPROVED copy of the paper bot (originals untouched). Fixes the 6
issues found in the audit:

 1. TRAIN-LABEL LEAKAGE: v1 trained on all bars before PAPER_START, but the last 24 bars'
    labels resolve DURING the paper window -> the "frozen" model had peeked. v2 embargoes
    the last HBAR training bars.
 2. ATR .bfill() LOOK-AHEAD: warmup bars were backfilled with FUTURE ATR values. v2 leaves
    them NaN and drops them.
 3. GUARDRAILS NOW ENFORCED, NOT JUST REPORTED: v1 simulated every ticker independently and
    only checked caps afterwards. v2 runs a chronological portfolio pass that actually
    SKIPS entries when at MAX_POSITIONS, when the ticker is already held, or when the
    daily loss limit / drawdown breaker has tripped.
 4. NOISE-TRADE FILTER: skip signals whose ATR < MIN_ATR_PCT of price (kills the sub-$1-risk
    TLT chop where the 3 bps cost eats the trade).
 5. SLIPPAGE: adds SLIP_BPS per side on top of commission (v1 assumed perfect fills).
 6. ONE POSITION PER TICKER across both strategies (v1 could hold v3-NVDA and v4-NVDA at
    once = double exposure).

Usage:  python paper_ensemble_v2.py 2026-06-29 2026-07-06     (start, end-exclusive)
Research/paper only — does not place orders.
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

from triple_barrier_ml import features, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features
from basket import ticker_cfg
from data import fetch_polygon

# ---- account + ENFORCED guardrails ----
ACCOUNT = 10_000.0
NOTIONAL_PCT = 10.0
MAX_POSITIONS = 10
DAILY_LOSS_LIMIT = ACCOUNT * 0.02
DD_BREAKER = ACCOUNT * 0.15
ONE_PER_TICKER = True
# ---- realism ----
SLIP_BPS = 1.0                     # per side, on top of COST_BPS
COST_BPS = 3.0                     # round-trip commission/spread baseline
MIN_ATR_PCT = 0.0012               # skip signals with ATR < 0.12% of price (noise floor)
# ---- strategy (unchanged from validated v3/v4) ----
SEL_Q, HBAR = 0.93, 24
CONFIGS = [("v3", 30, 1.5, 1.0), ("v4", 15, 4.0, 1.0)]
_C5 = {}


def atr_fixed(h, l, c, n=24):
    """ATR without the .bfill() look-ahead: warmup stays NaN and is dropped."""
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def load5(tk, fetch_end):
    if tk in _C5:
        return _C5[tk]
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    p = Path(f"data_cache/{tk}_recent_2026-06-01_{fetch_end}.csv")
    if p.exists():
        rec = pd.read_csv(p, parse_dates=["timestamp"])
    else:
        cfg = ticker_cfg(tk); cfg.data.start_date, cfg.data.end_date = "2026-06-01", fetch_end
        cfg.data.multiplier, cfg.data.timespan = 5, "minute"
        rec = fetch_polygon(cfg); rec.to_csv(p, index=False)
    df = (pd.concat([df, rec], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    _C5[tk] = df
    return df


def candidates(strat, mins, tp, sl, start, end, fetch_end):
    """Per (strategy, ticker) signal stream with embargoed training. Non-overlapping."""
    out = []
    for tk in TICKERS:
        d = load5(tk, fetch_end).set_index("timestamp").resample(f"{mins}min").agg(
            high=("high", "max"), low=("low", "min"), close=("close", "last"),
            volume=("volume", "sum")).dropna().reset_index()
        ts = pd.to_datetime(d["timestamp"]).to_numpy()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr_fixed(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        y = label(h, l, c, A, tp, sl)
        fv = (X.notna().all(axis=1) & np.isfinite(A)).to_numpy()
        n = len(c)
        tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(start)))[0]
        tr = tr[:-HBAR] if len(tr) > HBAR else tr          # FIX 1: embargo
        if len(tr) < 500:
            continue
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr].astype(int))
        thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q)
        fwd = np.where(fv & (ts >= np.datetime64(start)) & (ts < np.datetime64(end)))[0]
        if len(fwd) == 0:
            continue
        proba = {int(ix): float(p) for ix, p in zip(fwd, clf.predict_proba(X.iloc[fwd])[:, 1])}
        i, last = int(fwd[0]), int(fwd[-1])
        while i <= last:
            if proba.get(i, -1) < thr or A[i] / c[i] < MIN_ATR_PCT:   # FIX 4: noise floor
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
                                tgt=up, stop=dn, xts=None, exit=None, outcome="OPEN"))
                i = n; continue
            if res is None:
                res = 1 if c[min(j, n - 1)] > c[i] else 0
            ex_j = j if j < n else n - 1
            out.append(dict(strat=strat, tk=tk, ets=pd.Timestamp(ts[i]), entry=c[i], atr=a,
                            tgt=up, stop=dn, xts=pd.Timestamp(ts[ex_j]),
                            exit=(up if res == 1 else dn),
                            outcome="TARGET" if res == 1 else "STOP"))
            i = j + 1
    return out


def main():
    start = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2026-06-29")
    end = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else start + pd.Timedelta(days=7)
    fetch_end = str((pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date())
    cand = []
    for strat, mins, tp, sl in CONFIGS:
        cand += candidates(strat, mins, tp, sl, start, end, fetch_end)
    cand.sort(key=lambda t: t["ets"])

    # ---- chronological PORTFOLIO pass: guardrails ENFORCED (FIX 3, 6) ----
    notional = ACCOUNT * NOTIONAL_PCT / 100.0
    eff_cost = (COST_BPS + 2 * SLIP_BPS) / 1e4                       # FIX 5
    open_pos, taken, skipped = [], [], {"cap": 0, "ticker": 0, "daily": 0, "dd": 0}
    daily_pnl, cum, peak = {}, 0.0, 0.0
    halted = False
    for t in cand:
        # close positions whose exit is before this entry
        for p in [p for p in open_pos if p["xts"] is not None and p["xts"] <= t["ets"]]:
            open_pos.remove(p)
        if halted:
            skipped["dd"] += 1; continue
        d = t["ets"].date()
        if daily_pnl.get(d, 0.0) <= -DAILY_LOSS_LIMIT:
            skipped["daily"] += 1; continue
        if len(open_pos) >= MAX_POSITIONS:
            skipped["cap"] += 1; continue
        if ONE_PER_TICKER and any(p["tk"] == t["tk"] for p in open_pos):
            skipped["ticker"] += 1; continue
        sh = notional / t["entry"]
        t["shares"] = round(sh, 1)
        if t["exit"] is None:
            t["pnl"] = None
        else:
            gross = sh * (t["exit"] - t["entry"])
            t["pnl"] = round(gross - notional * eff_cost, 2)
            xd = t["xts"].date()
            daily_pnl[xd] = daily_pnl.get(xd, 0.0) + t["pnl"]
            cum += t["pnl"]; peak = max(peak, cum)
            if peak - cum >= DD_BREAKER:
                halted = True
        taken.append(t)
        open_pos.append(t)

    print(f"PAPER ENSEMBLE v2 (audited)  |  ${ACCOUNT:,.0f}  |  {NOTIONAL_PCT:.0f}%/trade  "
          f"|  slippage {SLIP_BPS} bps/side  |  ATR floor {MIN_ATR_PCT:.2%}")
    print(f"window {start.date()} .. {end.date()}  (data thru {fetch_end})\n")
    if not taken:
        print("no trades taken"); return
    print(f"  {'strat':>5} {'tk':>5} {'entry (UTC)':>16} {'bought$':>8} {'risk$':>7} {'outcome':>7} {'$ P&L':>8}")
    for t in taken:
        pl = "  open" if t["pnl"] is None else f"{t['pnl']:+.2f}"
        print(f"  {t['strat']:>5} {t['tk']:>5} {str(t['ets'])[5:16]:>16} "
              f"{t['shares']*t['entry']:>8,.0f} {t['shares']*t['atr']:>7.2f} {t['outcome']:>7} {pl:>8}")
    closed = [t for t in taken if t["pnl"] is not None]
    wins = sum(t["outcome"] == "TARGET" for t in closed)
    pnl = sum(t["pnl"] for t in closed)
    print(f"\n  RESULT: {len(closed)} closed, {wins} wins "
          f"({(wins/len(closed) if closed else 0):.0%}), "
          f"{sum(t['outcome']=='OPEN' for t in taken)} open  ->  P&L ${pnl:+,.2f} "
          f"({pnl/ACCOUNT*100:+.2f}%)   account ${ACCOUNT+pnl:,.2f}")
    print(f"  ENFORCED skips: at-cap={skipped['cap']}  same-ticker={skipped['ticker']}  "
          f"daily-limit={skipped['daily']}  dd-halt={skipped['dd']}")
    dd = 0.0
    c2 = 0.0; pk = 0.0
    for t in sorted(closed, key=lambda x: x["xts"]):
        c2 += t["pnl"]; pk = max(pk, c2); dd = min(dd, c2 - pk)
    print(f"  max drawdown ${dd:+,.2f}  (breaker -${DD_BREAKER:,.0f})  "
          f"{'HALTED' if halted else 'never tripped'}")
    Path("runs").mkdir(exist_ok=True)
    Path("runs/paper_v2_ledger.json").write_text(json.dumps(
        [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in t.items()} for t in taken], indent=1))
    print("  ledger -> runs/paper_v2_ledger.json")


if __name__ == "__main__":
    main()
