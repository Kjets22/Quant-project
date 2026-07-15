"""
alpaca_bot2.py — LIVE A/B: MARKET entries vs LIMIT entries, same signals, real paper fills.

Every top-7% signal is placed TWICE on the paper account:
  mkt-  market BUY + bracket           (fills immediately, pays the spread)
  lmt-  limit BUY at the signal price + bracket (never fills worse than signal; may MISS)
Each arm is tracked as its own virtual account in the ledger (style-tagged), with REAL
fill prices from the broker deciding the comparison. Limit patience = one full bar of the
strategy's timeframe; unfilled -> cancelled and recorded as MISSED.

All exits are per-ORDER (bracket leg status / targeted market sell), never per-symbol,
so the two arms co-exist safely on the same tickers. Guardrails per arm: max 10 open,
one per ticker, plus the shared daily-loss stop and drawdown halt.

  python alpaca_bot2.py --once | --dryrun | --status
PAPER ONLY (endpoint hard-asserted). Replaces alpaca_bot.py as the scheduled cycle.
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
from qqq_tournament import extra_features
from basket import ticker_cfg
from data import fetch_polygon

NOTIONAL = 1_000.0
MAX_POSITIONS = 10                 # per arm
DAILY_LOSS_LIMIT = 400.0           # shared (two arms trade in parallel)
DD_BREAKER = 3_000.0
MIN_ATR_PCT = 0.0012
SEL_Q = 0.93
#          name  tickers  mins hbar  mode      tp    sl  features  selection      ddl
CONFIGS = [("v3", TICKERS, 30, 24, "atr",     1.5, 1.0, "sr",    ("q", 0.93),    2),
           ("v4", TICKERS, 15, 24, "atr",     4.0, 1.0, "sr",    ("q", 0.93),    1),
           ("v6", TICKERS, 60, 96, "atr",     7.0, 1.0, "trend", ("q", 0.93),    8),
           ("v7", TICKERS, 60, 96, "struct", 10.0, 1.0, "trend", ("q", 0.93),    8),
           ("vC", TICKERS, 60, 96, "atr",    30.0, 3.0, "trend", ("q", 0.93),    8),
           # vQ: original tournament champion — high turnover (~9/mo), thin edge.
           ("vQ",  ["QQQ"], 5, 12, "dollar",  2.0, 2.0, "full",  ("conf", 0.90), 1),
           # vQ2: EVOLVED champion (10-gen tournament, min-of-halves fitness). Long-only
           # QQQ, $2.50 target / $2 stop, 2-hour clock, HistGB, top-10% confidence gate.
           # Arena halves +1.98/+1.52; untouched final year: 68.4% win, +12.2 bps/trade.
           ("vQ2", ["QQQ"], 5, 24, "dollar",  2.5, 2.0, "full",  ("conf", 0.90), 1),
           # vA: Evolution-II ACCURACY champion. $1.50 tgt / $2 stop, 4h clock, top-5%
           # gate. Win rate replicated arena->gate->final: 65.6% -> 68.6% (172 trades/yr).
           # Thin margin (+1.2bps) — live-slippage experiment; accuracy specialist.
           ("vA",  ["QQQ"], 5, 48, "dollar",  1.5, 2.0, "full",  ("conf", 0.95), 2),
           # vP: Evolution-III P&L champion — beat vQ through all 3 stages (arena +8.67%
           # worst-of-3, gate +3.27%, final +4.18% vs vQ +2.77%). 8h clock, $2/$2, HistGB.
           ("vP",  ["QQQ"], 5, 96, "dollar",  2.0, 2.0, "full",  ("conf", 0.85), 3),
           # vR: user-spec percentage bracket +0.4%/-0.2% (true 2:1), 2h clock, top-3%
           # gate. Evolution IV FINAL WINNER: +7.00% (t=2.04) on the untouched year —
           # the best P&L in the QQQ family, at full 5 bps costs. probe_pct.py.
           ("vR",  ["QQQ"], 5, 24, "pct",  0.004, 0.002, "full", ("q", 0.97), 1),
           # vS: Evolution-IV evolved challenger — percentage bracket +0.5%/-0.4%,
           # 8h clock, HistGB, top-10% gate. Gate year +11.17% (n=328); final year
           # +6.18% (t=0.87) — LOST the final to vR but runs alongside it live
           # (user wants both; turnover-vs-precision live test).
           ("vS",  ["QQQ"], 5, 96, "pct",  0.005, 0.004, "full", ("q", 0.90), 3)]
MODEL_BY_STRAT = {"vQ2": "histgb", "vP": "histgb", "vS": "histgb"}
# NOTE: vQ and vQ2 share QQQ; the one-per-ticker guardrail means whichever signals first
# holds the slot that hour — occasional skips are expected and logged.

# ---- vCO: the OPTIONS strategy (vc_options_real.py: 1-2w ATM calls +14.9%/trade) ----
# Fires on the same signals as vC but is its OWN strategy with its OWN book — tracked
# and reported separately from vC stock so the two can be compared head-to-head.
# Calls only — the stable is long-only (short side failed the fresh holdout).
OPT_STRATS = {"vC"}                # signal sources that also trigger a vCO entry
OPT_PREMIUM = 1_000.0              # target premium per signal
OPT_DTE = (8, 16)                  # expiry window; must outlive vC's 8-day time exit
OPT_MAX_OPEN = 8
OPT_MAX_CONTRACTS = 10

LEDGER = Path("runs/alpaca2_ledger.json")
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
        led = json.loads(LEDGER.read_text())
        led.setdefault("opt_open", [])
        led.setdefault("opt_closed", [])
        return led
    led = {"open": [], "pending": [], "closed": [], "acted_bars": {},
           "opt_open": [], "opt_closed": [],
           "state": {"peak_equity": 100000.0, "halted": False}}
    old = Path("runs/alpaca_ledger.json")                 # seed from bot v1
    if old.exists():
        led["acted_bars"] = json.loads(old.read_text()).get("acted_bars", {})
    return led


def save_ledger(led):
    LEDGER.parent.mkdir(exist_ok=True)
    LEDGER.write_text(json.dumps(led, indent=1, default=str))


def full_series(tk):
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
    cfg.data.end_date = str((pd.Timestamp.utcnow().tz_localize(None) + pd.Timedelta(days=1)).date())
    cfg.data.multiplier, cfg.data.timespan = 5, "minute"
    try:
        parts.append(fetch_polygon(cfg))
    except Exception as e:
        log(f"  [tail fetch failed {tk}: {e} — using cached]")
    df = (pd.concat(parts, ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
    _DATA[tk] = df
    return df


def prep(tk, mins, featmode, mode):
    d = full_series(tk).set_index("timestamp").resample(f"{mins}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna().reset_index()
    now = pd.Timestamp.utcnow().tz_localize(None)
    d = d[d["timestamp"] + pd.Timedelta(minutes=mins) <= now].reset_index(drop=True)
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr_fixed(h, l, c)
    X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                   sr_features(d).reset_index(drop=True)], axis=1)
    if featmode in ("trend", "full"):
        X = pd.concat([X, trend_features(h, l, c, A).reset_index(drop=True)], axis=1)
    if featmode == "full":
        X = pd.concat([X, extra_features(d, h, l, c, v, A).reset_index(drop=True)], axis=1)
    if mode == "struct":
        swing = (pd.Series(l).rolling(20).min().shift(1) - 0.25 * A).to_numpy()
        risk = c - swing
        valid = np.isfinite(risk) & (risk > 0.2 * A) & (risk < 4.0 * A)
        stop_px, tgt_px = swing, c + 10.0 * risk
    elif mode == "dollar":
        valid = np.isfinite(A)
        stop_px, tgt_px = None, None                     # filled per config in caller ($)
    else:
        valid = np.isfinite(A)
        stop_px, tgt_px = None, None
    return ts, h, l, c, A, X, valid, stop_px, tgt_px


def _barriers(mode, c, A, tp, sl, stop_px, tgt_px):
    if mode == "struct":
        return stop_px, tgt_px
    if mode == "dollar":
        return c - sl, c + tp                            # fixed-dollar barriers
    if mode == "pct":
        return c * (1 - sl), c * (1 + tp)                # percentage barriers
    return c - sl * A, c + tp * A                        # ATR barriers


def train_or_load(strat, tk, mins, hbar, mode, tp, sl, featmode, sel):
    MODELS.mkdir(exist_ok=True)
    pkl = MODELS / f"{strat}_{tk}_{datetime.now(timezone.utc):%Y%m%d}.pkl"
    if pkl.exists():
        return pickle.loads(pkl.read_bytes())
    ts, h, l, c, A, X, valid, stop_px, tgt_px = prep(tk, mins, featmode, mode)
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
    today = np.datetime64(pd.Timestamp.utcnow().tz_localize(None).normalize())
    fv = (X.notna().all(axis=1) & np.isfinite(A) & valid).to_numpy()
    tr = np.where(fv & np.isfinite(y) & (ts < today))[0]
    tr = tr[:-hbar] if len(tr) > hbar else tr
    if len(tr) < 500 or y[tr].sum() < 20:
        pkl.write_bytes(pickle.dumps(None))
        return None
    if MODEL_BY_STRAT.get(strat) == "histgb":
        from qqq_tournament import MODELS as TOURN_MODELS
        clf = TOURN_MODELS["histgb"]()
    else:
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                 reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    ptr = clf.predict_proba(X.iloc[tr])[:, 1]
    if sel[0] == "conf":                                 # long-side confidence gate (vQ)
        thr = float(0.5 + np.quantile(np.abs(ptr - 0.5), sel[1]))
    else:                                                # top-quantile of P(win)
        thr = float(np.quantile(ptr, sel[1]))
    obj = {"clf": clf, "thr": thr}
    pkl.write_bytes(pickle.dumps(obj))
    return obj


def _legs(order):
    tp_leg = sl_leg = None
    for leg in (order.get("legs") or []):
        if leg["type"] == "limit":
            tp_leg = leg["id"]
        elif leg["type"] == "stop":
            sl_leg = leg["id"]
    return tp_leg, sl_leg


def poll_pending(led, dry):
    now = pd.Timestamp.utcnow().tz_localize(None)
    keep = []
    for p in led["pending"]:
        try:
            o = broker.get_order(p["order_id"])
        except Exception as e:
            log(f"  [poll error {p['style']} {p['tk']}: {e}]"); keep.append(p); continue
        if o["status"] == "filled":
            tp_leg, sl_leg = _legs(o)
            p.update(fill=float(o["filled_avg_price"]), tp_leg=tp_leg, sl_leg=sl_leg,
                     ets=str(now), deadline=str(now + pd.Timedelta(days=p["ddl_days"])))
            led["open"].append(p)
            slip = (p["fill"] - p["sig_px"]) / p["sig_px"] * 1e4
            log(f"  FILLED {p['style']} {p['strat']} {p['tk']} @ {p['fill']:.2f} "
                f"(signal {p['sig_px']:.2f}, slip {slip:+.1f} bps)")
        elif pd.Timestamp(p["expiry"]) <= now and o["status"] in ("new", "accepted", "held", "partially_filled", "pending_new"):
            if not dry:
                broker.cancel_order(p["order_id"])
            p.update(outcome="MISSED", xts=str(now), pnl=0.0)
            led["closed"].append(p)
            log(f"  MISSED {p['style']} {p['strat']} {p['tk']} (limit {p['sig_px']:.2f} never filled)")
        elif o["status"] in ("canceled", "expired", "rejected"):
            p.update(outcome="MISSED", xts=str(now), pnl=0.0)
            led["closed"].append(p)
        else:
            keep.append(p)
    led["pending"] = keep


def manage_exits(led, dry):
    now = pd.Timestamp.utcnow().tz_localize(None)
    keep = []
    for p in led["open"]:
        outcome = None
        exit_px = None
        try:
            for leg_id, name in ((p.get("tp_leg"), "TARGET"), (p.get("sl_leg"), "STOP")):
                if leg_id:
                    o = broker.get_order(leg_id)
                    if o["status"] == "filled":
                        outcome, exit_px = name, float(o["filled_avg_price"])
                        break
        except Exception as e:
            log(f"  [exit poll error {p['tk']}: {e}]")
        if outcome is None and pd.Timestamp(p["deadline"]) <= now:
            if not dry:
                for leg_id in (p.get("tp_leg"), p.get("sl_leg")):
                    if leg_id:
                        broker.cancel_order(leg_id)
                broker.market_sell(p["tk"], p["qty"],
                                   f"tx-{p['style']}-{p['tk']}-{now:%Y%m%d%H%M%S}")
            pos = broker.position(p["tk"])
            exit_px = float(pos["current_price"]) if pos else p["fill"]
            outcome = "TIME"
        if outcome:
            pnl = round((exit_px - p["fill"]) * p["qty"], 2)
            p.update(outcome=outcome, exit=exit_px, pnl=pnl, xts=str(now))
            led["closed"].append(p)
            log(f"  CLOSED {p['style']} {p['strat']} {p['tk']} {outcome} pnl={pnl:+.2f}")
        else:
            keep.append(p)
    led["open"] = keep


def pick_call(tk, spot):
    """Nearest-the-money call expiring in OPT_DTE days (covers the time exit)."""
    today = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(hours=4)).date()
    cons = broker.option_contracts(
        tk, str(today + pd.Timedelta(days=OPT_DTE[0])),
        str(today + pd.Timedelta(days=OPT_DTE[1])), spot * 0.97, spot * 1.03)
    if not cons:
        return None
    cons.sort(key=lambda c: (abs(float(c["strike_price"]) - spot),
                             c["expiration_date"]))
    return cons[0]


def manage_opts(led, dry):
    """Fill-poll the option legs; sell when the stock leg is gone / deadline / expiry."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    today = (now - pd.Timedelta(hours=4)).date()
    keep = []
    for p in led["opt_open"]:
        try:
            if p.get("closing_id"):                       # sell already submitted
                o = broker.get_order(p["closing_id"])
                if o["status"] == "filled":
                    sell = float(o["filled_avg_price"])
                    pnl = round((sell - (p.get("fill") or sell)) * 100 * p["qty"], 2)
                    p.update(exit=sell, pnl=pnl, xts=str(now))
                    led["opt_closed"].append(p)
                    log(f"  OPT CLOSED {p['occ']} pnl={pnl:+.2f}")
                    continue
                keep.append(p); continue
            o = broker.get_order(p["order_id"])
            if o["status"] in ("canceled", "expired", "rejected"):
                p.update(pnl=0.0, xts=str(now), note="entry never filled")
                led["opt_closed"].append(p); continue
            if o["status"] == "filled" and not p.get("fill"):
                p["fill"] = float(o["filled_avg_price"])
                log(f"  OPT FILLED {p['occ']} x{p['qty']} @ {p['fill']:.2f}")
            stock_alive = any(x["tk"] == p["tk"] and x["strat"] == p.get("src", "vC")
                              and x["style"] == "mkt"
                              for x in led["open"] + led["pending"])
            expiry_near = pd.Timestamp(p["expiry"]).date() <= today + pd.Timedelta(days=1)
            due = (not stock_alive) or pd.Timestamp(p["deadline"]) <= now or expiry_near
            if due and p.get("fill") and not dry:
                so = broker.market_sell(p["occ"], p["qty"],
                                        f"optx-{p['tk']}-{now:%Y%m%d%H%M%S}")
                p["closing_id"] = so["id"]
                log(f"  OPT SELL {p['occ']} x{p['qty']} "
                    f"({'stock closed' if not stock_alive else 'deadline/expiry'})")
            keep.append(p)
        except Exception as e:
            log(f"  [opt manage error {p.get('occ')}: {e}]")
            keep.append(p)
    led["opt_open"] = keep


def cycle(dry=False):
    led = load_ledger()
    acct = broker.account()
    equity = float(acct["equity"])
    day_pnl = equity - float(acct["last_equity"])
    led["state"]["peak_equity"] = max(led["state"]["peak_equity"], equity)
    drawdown = led["state"]["peak_equity"] - equity
    clk = broker.clock()
    log(f"A/B cycle | equity=${equity:,.2f} day={day_pnl:+.2f} dd={drawdown:.2f} | "
        f"market={'OPEN' if clk['is_open'] else 'closed'}{' | DRYRUN' if dry else ''}")
    if not clk["is_open"] and not dry:
        save_ledger(led)
        return
    poll_pending(led, dry)
    manage_exits(led, dry)
    manage_opts(led, dry)
    if drawdown >= DD_BREAKER and not led["state"]["halted"]:
        led["state"]["halted"] = True
        log(f"!! DRAWDOWN BREAKER — HALTED (edit {LEDGER} to resume)")
    no_new = led["state"]["halted"] or day_pnl <= -DAILY_LOSS_LIMIT

    held = {s: {x["tk"] for x in led["open"] + led["pending"] if x["style"] == s}
            for s in ("mkt", "lmt")}
    counts = {s: sum(1 for x in led["open"] + led["pending"] if x["style"] == s)
              for s in ("mkt", "lmt")}
    n_sig = 0
    for strat, tks, mins, hbar, mode, tp, sl, featmode, sel, ddl in CONFIGS:
        for tk in tks:
            try:
                model = train_or_load(strat, tk, mins, hbar, mode, tp, sl, featmode, sel)
                if model is None:
                    continue
                ts, h, l, c, A, X, valid, stop_px, tgt_px = prep(tk, mins, featmode, mode)
                stop_px, tgt_px = _barriers(mode, c, A, tp, sl, stop_px, tgt_px)
                i = len(c) - 1
                if i < 1 or not valid[i] or X.iloc[i].isna().any():
                    continue
                if mode not in ("dollar", "pct") and A[i] / c[i] < MIN_ATR_PCT:
                    continue                             # ATR floor n/a for $/% brackets
                proba = float(model["clf"].predict_proba(X.iloc[[i]])[0, 1])
                bar_key, bar_ts = f"{strat}_{tk}", str(pd.Timestamp(ts[i]))
                if proba < model["thr"] or led["acted_bars"].get(bar_key) == bar_ts:
                    continue
                n_sig += 1
                if not dry:
                    led["acted_bars"][bar_key] = bar_ts
                qty = int(NOTIONAL // c[i])
                if qty < 1 or no_new:
                    continue
                log(f"  SIGNAL {strat} {tk} bar={bar_ts} p={proba:.3f} sig_px={c[i]:.2f} "
                    f"tgt={tgt_px[i]:.2f} stop={stop_px[i]:.2f} qty={qty}"
                    f"{' [DRYRUN]' if dry else ''}")
                if dry:
                    continue
                now = pd.Timestamp.utcnow().tz_localize(None)
                base = dict(strat=strat, tk=tk, qty=qty, sig_px=float(c[i]),
                            tgt=float(tgt_px[i]), stop=float(stop_px[i]),
                            bar=bar_ts, ddl_days=ddl)
                stamp = f"{pd.Timestamp(ts[i]):%Y%m%d%H%M}"
                if counts["mkt"] < MAX_POSITIONS and tk not in held["mkt"]:
                    o = broker.submit_bracket(tk, qty, tgt_px[i], stop_px[i],
                                              f"mkt-{strat}-{tk}-{stamp}")
                    led["pending"].append(dict(base, style="mkt", order_id=o["id"],
                                               expiry=str(now + pd.Timedelta(hours=24))))
                    counts["mkt"] += 1; held["mkt"].add(tk)
                if counts["lmt"] < MAX_POSITIONS and tk not in held["lmt"]:
                    o = broker.submit_limit_bracket(tk, qty, c[i], tgt_px[i], stop_px[i],
                                                    f"lmt-{strat}-{tk}-{stamp}")
                    expiry = pd.Timestamp(ts[i]) + pd.Timedelta(minutes=2 * mins)
                    led["pending"].append(dict(base, style="lmt", order_id=o["id"],
                                               expiry=str(expiry)))
                    counts["lmt"] += 1; held["lmt"].add(tk)
                if (strat in OPT_STRATS and len(led["opt_open"]) < OPT_MAX_OPEN
                        and not any(x["tk"] == tk for x in led["opt_open"])):
                    try:
                        con = pick_call(tk, c[i])
                        if con:
                            px = float(con.get("close_price") or 0) or None
                            qo = (max(1, min(OPT_MAX_CONTRACTS,
                                             int(OPT_PREMIUM // (px * 100))))
                                  if px else 1)
                            o = broker.market_buy(con["symbol"], qo,
                                                  f"opt-{strat}-{tk}-{stamp}")
                            led["opt_open"].append(dict(
                                strat="vCO", src=strat, tk=tk, occ=con["symbol"],
                                qty=qo, order_id=o["id"],
                                expiry=con["expiration_date"],
                                sig_px=float(c[i]), bar=bar_ts, ets=str(now),
                                deadline=str(now + pd.Timedelta(days=ddl)),
                                est_px=px))
                            log(f"  OPT BUY {con['symbol']} x{qo} "
                                f"(prev close ${px if px else '?'})")
                    except Exception as e:
                        log(f"  [opt entry error {strat} {tk}: {e}]")
            except Exception as e:
                log(f"  [error {strat} {tk}: {e}]")
    log(f"A/B cycle done | signals={n_sig} open={len(led['open'])} "
        f"pending={len(led['pending'])} closed={len(led['closed'])}")
    save_ledger(led)


def status():
    a = broker.account()
    led = load_ledger()
    print(f"equity ${float(a['equity']):,.2f} | day {float(a['equity'])-float(a['last_equity']):+.2f} "
          f"| halted: {led['state']['halted']}")
    for s, name in (("mkt", "MARKET arm"), ("lmt", "LIMIT arm")):
        cl = [x for x in led["closed"] if x["style"] == s and x["outcome"] != "MISSED"]
        missed = sum(1 for x in led["closed"] if x["style"] == s and x["outcome"] == "MISSED")
        wins = sum(x["outcome"] == "TARGET" for x in cl)
        pnl = sum(x.get("pnl") or 0 for x in cl)
        op = [x for x in led["open"] if x["style"] == s]
        pend = [x for x in led["pending"] if x["style"] == s]
        slips = [(x["fill"] - x["sig_px"]) / x["sig_px"] * 1e4 for x in (cl + op) if x.get("fill")]
        avg_slip = f"{np.mean(slips):+.1f}" if slips else "n/a"
        print(f"  {name:>10}: closed={len(cl)} wins={wins} missed={missed} "
              f"P&L=${pnl:+.2f} | open={len(op)} pending={len(pend)} | avg slip {avg_slip} bps")
        for x in op:
            print(f"      open  {x['strat']} {x['tk']} {x['qty']}sh @ {x['fill']:.2f} "
                  f"tgt {x['tgt']:.2f} stop {x['stop']:.2f}")
        for x in pend:
            print(f"      pend  {x['strat']} {x['tk']} ({'limit @ '+format(x['sig_px'],'.2f') if s=='lmt' else 'market'})")
    ocl = led.get("opt_closed", [])
    opnl = sum(x.get("pnl") or 0 for x in ocl)
    owins = sum(1 for x in ocl if (x.get("pnl") or 0) > 0)
    print(f"  vCO OPTIONS strategy (own book, vC signals): "
          f"open={len(led.get('opt_open', []))} closed={len(ocl)} wins={owins} "
          f"realized=${opnl:+.2f}")
    for x in led.get("opt_open", []):
        print(f"      opt   {x['occ']} x{x['qty']} "
              f"{'@ '+format(x['fill'], '.2f') if x.get('fill') else '(pending fill)'} "
              f"exp {x['expiry']}")


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    elif "--dryrun" in sys.argv:
        cycle(dry=True)
    else:
        cycle(dry=False)
