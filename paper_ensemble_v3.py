"""
paper_ensemble_v3.py — the paper-money UI, now with ALL FOUR strategies:
  v3 = 30-min, 1.5:1, ATR stop            (mean-reversion core)
  v4 = 15-min, 4:1,   ATR stop            (mean-reversion big payoff)
  v6 = 60-min, 7:1,   ATR stop, HBAR=96, +trend features   (trend hunter, experimental)
  v7 = 60-min, 10:1 on actual risk, STRUCTURE stop (0.25 ATR below 20-bar swing low),
       HBAR=96, +trend features           (the validated wide hunter)

Same audited engine as v2: embargoed training, no-bfill ATR, 0.12% ATR floor, 1 bp/side
slippage + 3 bps cost, corrected accounting, ENFORCED guardrails (position cap, one per
ticker, daily loss limit, drawdown breaker). Research/paper only — places no orders.

  python paper_ensemble_v3.py 2026-06-29 2026-07-07
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

from triple_barrier_ml import features
from triple_barrier_breadth import TICKERS
from sr_features import sr_features
from wide_hunter import atr_fixed, trend_features
from basket import ticker_cfg
from data import fetch_polygon

ACCOUNT = 10_000.0
NOTIONAL_PCT = 10.0
MAX_POSITIONS = 10
DAILY_LOSS_LIMIT = ACCOUNT * 0.02
DD_BREAKER = ACCOUNT * 0.15
ONE_PER_TICKER = True
SLIP_BPS, COST_BPS = 1.0, 3.0
MIN_ATR_PCT = 0.0012
SEL_Q = 0.93
#          name  mins hbar   mode     tp   sl  trend?
CONFIGS = [("v3", 30, 24, "atr",    1.5, 1.0, False),
           ("v4", 15, 24, "atr",    4.0, 1.0, False),
           ("v6", 60, 96, "atr",    7.0, 1.0, True),
           ("v7", 60, 96, "struct", 10.0, 1.0, True),
           ("vC", 60, 96, "atr",    30.0, 3.0, True)]
_C5 = {}


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


def candidates(strat, mins, hbar, mode, tp, sl, use_trend, start, end, fetch_end):
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
        if use_trend:
            X = pd.concat([X, trend_features(h, l, c, A).reset_index(drop=True)], axis=1)
        if mode == "struct":
            swing = (pd.Series(l).rolling(20).min().shift(1) - 0.25 * A).to_numpy()
            risk = c - swing
            valid = np.isfinite(risk) & (risk > 0.2 * A) & (risk < 4.0 * A)
            stop_px, tgt_px = swing, c + 10.0 * risk
        else:
            valid = np.isfinite(A)
            stop_px, tgt_px = c - sl * A, c + tp * A
        n = len(c)
        y = np.full(n, np.nan)                       # label with the SAME barriers
        for i in range(n - 1):
            if not valid[i]:
                continue
            for j in range(i + 1, min(i + hbar + 1, n)):
                if l[j] <= stop_px[i]:
                    y[i] = 0; break
                if h[j] >= tgt_px[i]:
                    y[i] = 1; break
        fv = (X.notna().all(axis=1) & np.isfinite(A) & valid).to_numpy()
        tr = np.where(fv & np.isfinite(y) & (ts < np.datetime64(start)))[0]
        tr = tr[:-hbar] if len(tr) > hbar else tr    # embargo
        if len(tr) < 500 or y[tr].sum() < 20:
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
            if proba.get(i, -1) < thr or A[i] / c[i] < MIN_ATR_PCT:
                i += 1; continue
            res, j = None, i + 1
            while j < min(i + hbar + 1, n):
                if l[j] <= stop_px[i]:
                    res = 0; break
                if h[j] >= tgt_px[i]:
                    res = 1; break
                j += 1
            if res is None and j >= n:
                out.append(dict(strat=strat, tk=tk, ets=pd.Timestamp(ts[i]), entry=c[i],
                                tgt=tgt_px[i], stop=stop_px[i], xts=None, exit=None,
                                outcome="OPEN"))
                i = n; continue
            ex = min(j, n - 1)
            if res is None:                          # time exit at actual close
                out.append(dict(strat=strat, tk=tk, ets=pd.Timestamp(ts[i]), entry=c[i],
                                tgt=tgt_px[i], stop=stop_px[i], xts=pd.Timestamp(ts[ex]),
                                exit=c[ex], outcome="TIME"))
            else:
                out.append(dict(strat=strat, tk=tk, ets=pd.Timestamp(ts[i]), entry=c[i],
                                tgt=tgt_px[i], stop=stop_px[i], xts=pd.Timestamp(ts[ex]),
                                exit=(tgt_px[i] if res == 1 else stop_px[i]),
                                outcome="TARGET" if res == 1 else "STOP"))
            i = j + 1
    return out


def main():
    start = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2026-06-29")
    end = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else start + pd.Timedelta(days=7)
    fetch_end = str((pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date())
    cand = []
    for cfg in CONFIGS:
        cand += candidates(*cfg, start, end, fetch_end)
        print(f"  [{cfg[0]} scanned]", flush=True)
    cand.sort(key=lambda t: t["ets"])

    notional = ACCOUNT * NOTIONAL_PCT / 100.0
    eff_cost = (COST_BPS + 2 * SLIP_BPS) / 1e4
    open_pos, taken = [], []
    skipped = {"cap": 0, "ticker": 0, "daily": 0, "dd": 0}
    daily_pnl, cum, peak, halted = {}, 0.0, 0.0, False
    for t in cand:
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
            t["pnl"] = round(sh * (t["exit"] - t["entry"]) - notional * eff_cost, 2)
            xd = t["xts"].date()
            daily_pnl[xd] = daily_pnl.get(xd, 0.0) + t["pnl"]
            cum += t["pnl"]; peak = max(peak, cum)
            if peak - cum >= DD_BREAKER:
                halted = True
        taken.append(t); open_pos.append(t)

    print(f"\nPAPER v3 — 4 strategies  |  ${ACCOUNT:,.0f}  |  {NOTIONAL_PCT:.0f}%/trade  "
          f"|  window {start.date()} .. {end.date()}  (data thru {fetch_end})\n")
    print(f"  {'strat':>5} {'tk':>5} {'entry (UTC)':>16} {'in':>8} {'tgt':>8} {'stop':>8} "
          f"{'outcome':>7} {'$ P&L':>8}")
    for t in taken:
        pl = "  open" if t["pnl"] is None else f"{t['pnl']:+.2f}"
        print(f"  {t['strat']:>5} {t['tk']:>5} {str(t['ets'])[5:16]:>16} {t['entry']:>8.2f} "
              f"{t['tgt']:>8.2f} {t['stop']:>8.2f} {t['outcome']:>7} {pl:>8}")
    print()
    tot_all = 0.0
    for name, *_ in CONFIGS:
        tt = [t for t in taken if t["strat"] == name]
        cl = [t for t in tt if t["pnl"] is not None]
        wins = sum(t["outcome"] == "TARGET" for t in cl)
        pnl = sum(t["pnl"] for t in cl)
        tot_all += pnl
        nop = sum(t["outcome"] == "OPEN" for t in tt)
        wp = f"{wins/len(cl):.0%}" if cl else "-"
        print(f"  {name}: closed={len(cl):>3}  wins={wins:>2} ({wp:>4})  "
              f"P&L=${pnl:+8.2f}  open={nop}")
    print(f"\n  COMBINED P&L ${tot_all:+,.2f} ({tot_all/ACCOUNT*100:+.2f}%)  "
          f"account ${ACCOUNT+tot_all:,.2f}")
    print(f"  skips: cap={skipped['cap']} ticker={skipped['ticker']} "
          f"daily={skipped['daily']} dd={skipped['dd']}   "
          f"{'HALTED' if halted else 'guardrails never tripped'}")
    Path("runs").mkdir(exist_ok=True)
    Path("runs/paper_v3_ledger.json").write_text(json.dumps(
        [{k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in t.items()}
         for t in taken], indent=1))
    print("  ledger -> runs/paper_v3_ledger.json")


if __name__ == "__main__":
    main()
