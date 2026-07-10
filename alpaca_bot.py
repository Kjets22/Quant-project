"""
alpaca_bot.py — the AUTONOMOUS paper-trading bot (Alpaca paper account).

Runs the five frozen strategies live:
  v3  30-min 1.5:1 ATR bracket          (mean-reversion core)
  v4  15-min 4:1 ATR bracket            (big-payoff mean reversion)
  v6  60-min 7:1 ATR bracket, trend     (experimental)
  v7  60-min 10:1-on-risk, structure stop, trend
  vC  60-min 30:3 ATR trend-holder

Each cycle:
  1. Market clock check (entries and exits only while the market is open).
  2. Refresh data: 5-year base cache + latest recent cache + fresh Polygon tail.
  3. Load (or train once per day) each strategy x ticker model; embargoed training
     on bars strictly before today.
  4. If the latest COMPLETED bar is a top-7% signal (+ ATR floor, struct validity):
     submit a native BRACKET order (market buy + sell-limit target + sell-stop).
  5. Time-exit manager: positions past their clock get legs cancelled + closed.
  6. Guardrails ENFORCED: $1,000/position (whole shares), max 10 open, one per
     ticker, daily-loss stop ($200), drawdown breaker ($1,500 from peak) -> halt.
  7. Ledger -> runs/alpaca_ledger.json, log -> runs/alpaca_log.txt.

Usage:
  python alpaca_bot.py --once      one full live cycle (what the scheduler calls)
  python alpaca_bot.py --dryrun    compute signals, place NO orders
  python alpaca_bot.py --status    account + ledger snapshot only

PAPER ONLY: alpaca_api.py hard-asserts the paper endpoint. This code never touches
real money. Scheduling it (see run_alpaca_bot.bat) is the owner's decision.
"""

from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb

import alpaca_api as broker
from triple_barrier_ml import features
from triple_barrier_breadth import TICKERS
from sr_features import sr_features
from wide_hunter import atr_fixed, trend_features
from basket import ticker_cfg
from data import fetch_polygon

# ---- sizing + guardrails (single $100k paper account, all 5 strategies) ----
NOTIONAL = 1_000.0
MAX_POSITIONS = 10
DAILY_LOSS_LIMIT = 200.0
DD_BREAKER = 1_500.0
MIN_ATR_PCT = 0.0012
SEL_Q = 0.93
#          name  mins hbar   mode     tp    sl  trend? deadline(calendar days)
CONFIGS = [("v3", 30, 24, "atr",     1.5, 1.0, False, 2),
           ("v4", 15, 24, "atr",     4.0, 1.0, False, 1),
           ("v6", 60, 96, "atr",     7.0, 1.0, True,  8),
           ("v7", 60, 96, "struct", 10.0, 1.0, True,  8),
           ("vC", 60, 96, "atr",    30.0, 3.0, True,  8)]

LEDGER = Path("runs/alpaca_ledger.json")
LOG = Path("runs/alpaca_log.txt")
MODELS = Path("models")
_DATA = {}


def log(msg):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z  {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_ledger():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"open": [], "closed": [], "acted_bars": {},
            "state": {"peak_equity": 100000.0, "halted": False}}


def save_ledger(led):
    LEDGER.parent.mkdir(exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=1, default=str))


def full_series(tk):
    """5y base cache + newest recent cache + fresh Polygon tail (always refetched)."""
    if tk in _DATA:
        return _DATA[tk]
    base = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv",
                       parse_dates=["timestamp"])
    parts = [base]
    recents = sorted(Path("data_cache").glob(f"{tk}_recent_2026-06-01_*.csv"))
    if recents:
        parts.append(pd.read_csv(recents[-1], parse_dates=["timestamp"]))
    last = max(p["timestamp"].iloc[-1] for p in parts)
    cfg = ticker_cfg(tk)
    cfg.data.start_date = str((last - pd.Timedelta(days=2)).date())
    cfg.data.end_date = str((pd.Timestamp.utcnow().tz_localize(None)
                             + pd.Timedelta(days=1)).date())
    cfg.data.multiplier, cfg.data.timespan = 5, "minute"
    try:
        parts.append(fetch_polygon(cfg))
    except Exception as e:
        log(f"  [tail fetch failed {tk}: {e} — using cached data]")
    df = (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    _DATA[tk] = df
    return df


def prep(tk, mins, use_trend, mode):
    d = full_series(tk).set_index("timestamp").resample(f"{mins}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna().reset_index()
    # drop the still-forming bar so we only ever act on COMPLETED bars
    now = pd.Timestamp.utcnow().tz_localize(None)
    d = d[d["timestamp"] + pd.Timedelta(minutes=mins) <= now].reset_index(drop=True)
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
        stop_px, tgt_px = None, None      # filled per config in caller
    return ts, h, l, c, A, X, valid, stop_px, tgt_px


def train_or_load(strat, tk, mins, hbar, mode, tp, sl, use_trend):
    """One model per strategy x ticker x day; embargoed training on bars < today."""
    MODELS.mkdir(exist_ok=True)
    tag = f"{strat}_{tk}_{datetime.now(timezone.utc):%Y%m%d}"
    pkl = MODELS / f"{tag}.pkl"
    if pkl.exists():
        return pickle.loads(pkl.read_bytes())
    ts, h, l, c, A, X, valid, stop_px, tgt_px = prep(tk, mins, use_trend, mode)
    if mode != "struct":
        stop_px, tgt_px = c - sl * A, c + tp * A
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
    today = np.datetime64(pd.Timestamp.utcnow().tz_localize(None).normalize())
    fv = (X.notna().all(axis=1) & np.isfinite(A) & valid).to_numpy()
    tr = np.where(fv & np.isfinite(y) & (ts < today))[0]
    tr = tr[:-hbar] if len(tr) > hbar else tr               # embargo
    if len(tr) < 500 or y[tr].sum() < 20:
        pkl.write_bytes(pickle.dumps(None))
        return None
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = float(np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], SEL_Q))
    obj = {"clf": clf, "thr": thr}
    pkl.write_bytes(pickle.dumps(obj))
    return obj


def manage_exits(led, dry):
    """Bracket already exited -> record it. Past deadline -> cancel legs + close."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    still = []
    for p in led["open"]:
        pos = broker.position(p["tk"])
        if pos is None or float(pos["qty"]) == 0:           # bracket leg exited
            fill = None
            try:
                for o in broker.closed_orders(p["ets"], symbols=[p["tk"]]):
                    if o["side"] == "sell" and o["status"] == "filled":
                        fill = float(o["filled_avg_price"]); break
            except Exception:
                pass
            pnl = round((fill - p["fill"]) * p["qty"], 2) if fill else None
            p.update(outcome="BRACKET", exit=fill, pnl=pnl, xts=str(now))
            led["closed"].append(p)
            log(f"  CLOSED  {p['strat']} {p['tk']} via bracket  pnl={pnl}")
        elif pd.Timestamp(p["deadline"]) <= now:
            if not dry:
                broker.cancel_symbol_orders(p["tk"])
                broker.close_position(p["tk"])
            mark = float(pos["current_price"])
            pnl = round((mark - p["fill"]) * p["qty"], 2)
            p.update(outcome="TIME", exit=mark, pnl=pnl, xts=str(now))
            led["closed"].append(p)
            log(f"  TIME-EXIT {p['strat']} {p['tk']} @~{mark}  pnl={pnl}")
        else:
            still.append(p)
    led["open"] = still


def cycle(dry=False):
    led = load_ledger()
    acct = broker.account()
    equity = float(acct["equity"])
    day_pnl = equity - float(acct["last_equity"])
    led["state"]["peak_equity"] = max(led["state"]["peak_equity"], equity)
    drawdown = led["state"]["peak_equity"] - equity
    clk = broker.clock()
    log(f"cycle start | equity=${equity:,.2f} day={day_pnl:+.2f} dd={drawdown:.2f} "
        f"| market={'OPEN' if clk['is_open'] else 'closed'}{' | DRYRUN' if dry else ''}")

    if not clk["is_open"] and not dry:
        log("market closed — nothing to do")
        save_ledger(led)
        return

    manage_exits(led, dry)

    if drawdown >= DD_BREAKER and not led["state"]["halted"]:
        led["state"]["halted"] = True
        log(f"!! DRAWDOWN BREAKER (${drawdown:.0f} >= ${DD_BREAKER:.0f}) — HALTED. "
            f"Review and delete the halted flag in {LEDGER} to resume.")
    no_new = led["state"]["halted"] or day_pnl <= -DAILY_LOSS_LIMIT
    if day_pnl <= -DAILY_LOSS_LIMIT:
        log(f"daily loss limit hit ({day_pnl:+.2f}) — no new entries today")

    held = {p["tk"] for p in led["open"]}
    n_signals = 0
    for strat, mins, hbar, mode, tp, sl, use_trend, ddl_days in CONFIGS:
        for tk in TICKERS:
            try:
                model = train_or_load(strat, tk, mins, hbar, mode, tp, sl, use_trend)
                if model is None:
                    continue
                ts, h, l, c, A, X, valid, stop_px, tgt_px = prep(tk, mins, use_trend, mode)
                if mode != "struct":
                    stop_px, tgt_px = c - sl * A, c + tp * A
                i = len(c) - 1
                if i < 1 or not valid[i] or X.iloc[i].isna().any():
                    continue
                if A[i] / c[i] < MIN_ATR_PCT:
                    continue
                proba = float(model["clf"].predict_proba(X.iloc[[i]])[0, 1])
                bar_key = f"{strat}_{tk}"
                bar_ts = str(pd.Timestamp(ts[i]))
                if proba < model["thr"] or led["acted_bars"].get(bar_key) == bar_ts:
                    continue
                n_signals += 1
                if not dry:                      # dry runs must not consume signals
                    led["acted_bars"][bar_key] = bar_ts
                if no_new or len(led["open"]) >= MAX_POSITIONS or tk in held:
                    log(f"  SIGNAL {strat} {tk} @ {bar_ts} SKIPPED "
                        f"(guardrail: halted/daily/cap/ticker)")
                    continue
                qty = int(NOTIONAL // c[i])
                if qty < 1:
                    continue
                cid = f"{strat}-{tk}-{pd.Timestamp(ts[i]):%Y%m%d%H%M}"
                log(f"  SIGNAL {strat} {tk} bar={bar_ts} p={proba:.3f} "
                    f"entry~{c[i]:.2f} tgt={tgt_px[i]:.2f} stop={stop_px[i]:.2f} qty={qty}"
                    f"{' [DRYRUN — not sent]' if dry else ''}")
                if dry:
                    continue
                o = broker.submit_bracket(tk, qty, tgt_px[i], stop_px[i], cid)
                fill = c[i]
                deadline = pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(days=ddl_days)
                led["open"].append(dict(strat=strat, tk=tk, qty=qty, fill=float(fill),
                                        tgt=float(tgt_px[i]), stop=float(stop_px[i]),
                                        ets=str(pd.Timestamp.utcnow().tz_localize(None)),
                                        deadline=str(deadline), order_id=o["id"]))
                held.add(tk)
            except Exception as e:
                log(f"  [error {strat} {tk}: {e}]")
    log(f"cycle done | signals={n_signals} open={len(led['open'])} "
        f"closed-total={len(led['closed'])}")
    save_ledger(led)


def status():
    a = broker.account()
    led = load_ledger()
    print(f"equity ${float(a['equity']):,.2f} | cash ${float(a['cash']):,.2f} | "
          f"day P&L {float(a['equity'])-float(a['last_equity']):+.2f}")
    print(f"open (ledger): {len(led['open'])}  closed: {len(led['closed'])}  "
          f"halted: {led['state']['halted']}")
    for p in led["open"]:
        print(f"  {p['strat']:>3} {p['tk']:>5} qty {p['qty']} @ {p['fill']:.2f} "
              f"tgt {p['tgt']:.2f} stop {p['stop']:.2f} deadline {p['deadline'][:10]}")
    for p in led["closed"][-10:]:
        print(f"  done: {p['strat']:>3} {p['tk']:>5} {p['outcome']} pnl {p.get('pnl')}")


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    elif "--dryrun" in sys.argv:
        cycle(dry=True)
    else:
        cycle(dry=False)
