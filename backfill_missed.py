"""
backfill_missed.py — recover trades missed for the WRONG reasons, per user policy:
credit a strategy's balance for trades it missed due to (a) infrastructure OUTAGES
(scheduler dead during market hours) or (b) the OLD cross-strategy blocking rule
(one-per-ticker across strategies, removed 2026-07-20). Genuine skips — gates not
met, capacity after the fix, MISSED limit fills — are NOT recovered.

Sources:
  A. OUTAGES: gaps > 35 min between market-hours cycles in runs/alpaca_log.txt.
     For each gap, replay the cycle cadence with the models the bot HAD (the most
     recent dated pickle before the gap) and detect signals it would have taken.
  B. RULE-BLOCKS: every "SIGNAL strat tk bar=..." line in the log with no matching
     mkt-arm ledger entry, timestamped BEFORE the independence fix — the log gives
     the exact sig_px/tgt/stop/qty the bot printed.

Each recovered signal is simulated exactly like a live mkt-arm trade: enter at the
signal close, first-touch target/stop on 5-min highs/lows (stop wins ties), time
exit at the ddl deadline, 5 bps cost. Idempotent: keyed by (strat, tk, bar); output
appended to runs/recovered_trades.json (consumed by daily_report.py).
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from alpaca_bot2 import CONFIGS, MIN_ATR_PCT, prep, _barriers, full_series, NOTIONAL

EFF_COST = 5.0 / 1e4
LOG = Path("runs/alpaca_log.txt")
LEDGER = Path("runs/alpaca2_ledger.json")
OUT = Path("runs/recovered_trades.json")
INDEPENDENCE_FIX = pd.Timestamp("2026-07-20 17:45:00")   # UTC commit time of the fix
MODELS = Path("models")


def cycle_lines():
    out = []
    for ln in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)Z\s+A/B cycle \|.*market=(\w+)", ln)
        if m:
            out.append((pd.Timestamp(m.group(1)), m.group(2) == "OPEN"))
    return out


def outage_windows():
    """Market-hours gaps > 35 min between consecutive cycles (plus never-started days)."""
    cyc = cycle_lines()
    gaps = []
    for (t0, _), (t1, _) in zip(cyc, cyc[1:]):
        if (t1 - t0) > pd.Timedelta(minutes=35):
            gaps.append((t0, t1))
    return gaps


def rth_cycle_times(lo, hi):
    """The bot's cycle timestamps (:01/:16/:31/:46, 13:31-20:16 UTC approx) in [lo,hi]."""
    times = []
    for day in pd.date_range(lo.normalize(), hi.normalize(), freq="D"):
        if day.dayofweek >= 5:
            continue
        t = day + pd.Timedelta(hours=13, minutes=31)
        end = day + pd.Timedelta(hours=20, minutes=16)
        while t <= end:
            if lo < t < hi:
                times.append(t)
            t += pd.Timedelta(minutes=15)
    return times


def held_by_real(led, strat, tk, t):
    """Was the REAL mkt book holding (strat, tk) at time t? (Self-blocks are
    legitimate under old AND new rules — never recover those.)"""
    for x in led["open"] + led["closed"] + led["pending"]:
        if x.get("style") != "mkt" or x["strat"] != strat or x["tk"] != tk:
            continue
        if x.get("outcome") == "MISSED":
            continue
        start = pd.Timestamp(x.get("ets") or x.get("bar"))
        end = pd.Timestamp(x["xts"]) if x.get("xts") else None
        if start <= t and (end is None or end >= t):
            return True
    return False


def load_model(strat, tk, before):
    """Most recent dated pickle strictly before `before` (what the bot had)."""
    best = None
    for p in MODELS.glob(f"{strat}_{tk}_*.pkl"):
        d = p.stem.split("_")[-1]
        if d < f"{before:%Y%m%d}" or d == f"{before:%Y%m%d}":
            best = p if (best is None or p.stem > best.stem) else best
    if best is None:
        return None
    try:
        return pickle.loads(best.read_bytes())
    except Exception:
        return None


def simulate(tk, entry_ts, sig_px, tgt, stop, ddl_days, qty):
    """First-touch outcome from 5-min data after entry; stop wins ties; 5bps cost."""
    d = full_series(tk)
    seg = d[d["timestamp"] > entry_ts].reset_index(drop=True)
    deadline = entry_ts + pd.Timedelta(days=ddl_days)
    cost = qty * sig_px * EFF_COST
    for r in seg.itertuples():
        if r.timestamp > deadline:
            px = float(r.close)
            return "TIME", round(qty * (px - sig_px) - cost, 2), str(r.timestamp)
        if float(r.low) <= stop:
            return "STOP", round(qty * (stop - sig_px) - cost, 2), str(r.timestamp)
        if float(r.high) >= tgt:
            return "TARGET", round(qty * (tgt - sig_px) - cost, 2), str(r.timestamp)
    px = float(seg["close"].iloc[-1]) if len(seg) else sig_px
    return "OPEN", round(qty * (px - sig_px) - cost, 2), None


def prev_holds(prev):
    """Rebuild (strat, tk) -> held-until from earlier recovered trades so reruns
    keep the one-position-at-a-time sequencing (idempotency)."""
    ddl_by = {cfg[0]: cfg[9] for cfg in CONFIGS}
    hold = {}
    for r in prev:
        start = pd.Timestamp(r["ets"])
        end = (pd.Timestamp(r["xts"]) if r.get("xts")
               else start + pd.Timedelta(days=ddl_by.get(r["strat"], 2)))
        k = (r["strat"], r["tk"])
        hold[k] = max(hold.get(k, end), end)
    return hold


def recover_outages(seen, led, prev):
    recs = []
    entries = {(x["strat"], x["tk"], x["bar"]) for x in
               led["open"] + led["pending"] + led["closed"] if x.get("style") == "mkt"}
    ph = prev_holds(prev)
    for lo, hi in outage_windows():
        print(f"outage window: {lo} .. {hi} UTC")
        for strat, tks, mins, hbar, mode, tp, sl, featmode, sel, ddl in CONFIGS:
            for tk in tks:
                model = load_model(strat, tk, lo)
                if model is None:
                    continue
                try:
                    ts, h, l, c, A, X, valid, sp, gp = prep(tk, mins, featmode, mode)
                except Exception as e:
                    print(f"  [prep warn {strat}/{tk}: {e}]"); continue
                sp, gp = _barriers(mode, c, A, tp, sl, sp, gp)
                tsx = pd.DatetimeIndex(ts)
                held_until = ph.get((strat, tk))
                for ct in rth_cycle_times(lo, hi):
                    i = int(tsx.searchsorted(ct - pd.Timedelta(minutes=mins),
                                             side="right")) - 1
                    if i < 1 or tsx[i] + pd.Timedelta(minutes=mins) > ct:
                        continue
                    bar = str(tsx[i])
                    key = f"{strat}|{tk}|{bar}"
                    if key in seen or (held_until and ct < held_until):
                        continue
                    if (strat, tk, bar) in entries:
                        continue          # the real book took this bar (late catch-up)
                    if not valid[i] or X.iloc[i].isna().any():
                        continue
                    if mode not in ("dollar", "pct") and A[i] / c[i] < MIN_ATR_PCT:
                        continue
                    if held_by_real(led, strat, tk, ct):
                        continue          # self-block: legitimate under old AND new rules
                    proba = float(model["clf"].predict_proba(X.iloc[[i]])[0, 1])
                    if proba < model["thr"]:
                        continue
                    qty = int(NOTIONAL // c[i])
                    if qty < 1:
                        continue
                    oc, pnl, xts = simulate(tk, tsx[i], float(c[i]), float(gp[i]),
                                            float(sp[i]), ddl, qty)
                    recs.append(dict(strat=strat, tk=tk, bar=bar, reason="OUTAGE",
                                     sig_px=round(float(c[i]), 2), qty=qty,
                                     tgt=round(float(gp[i]), 2),
                                     stop=round(float(sp[i]), 2),
                                     outcome=oc, pnl=pnl, xts=xts, ets=str(ct)))
                    seen.add(key)
                    held_until = (pd.Timestamp(xts) if xts
                                  else ct + pd.Timedelta(days=ddl))
                    print(f"  RECOVERED {strat} {tk} bar={bar} -> {oc} {pnl:+.2f}")
    return recs


def recover_rule_blocks(seen, led, prev):
    recs = []
    ddl_by = {cfg[0]: cfg[9] for cfg in CONFIGS}
    entries = {(x["strat"], x["tk"], x["bar"]) for x in
               led["open"] + led["pending"] + led["closed"] if x.get("style") == "mkt"}
    open_until_seed = prev_holds(prev)
    pat = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)Z\s+SIGNAL (\w+) (\w+) "
                     r"bar=(\S+ \S+) p=[\d.]+ sig_px=([\d.]+) tgt=([\d.]+) "
                     r"stop=([\d.]+) qty=(\d+)")
    open_until = dict(open_until_seed)
    for ln in LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.match(ln)
        if not m:
            continue
        t = pd.Timestamp(m.group(1))
        strat, tk, bar = m.group(2), m.group(3), m.group(4)
        if t >= INDEPENDENCE_FIX or (strat, tk, bar) in entries:
            continue
        key = f"{strat}|{tk}|{bar}"
        if key in seen:
            continue
        hu = open_until.get((strat, tk))
        if hu and t < hu:
            continue                          # our own earlier recovered trade holds it
        if held_by_real(led, strat, tk, t):
            continue                          # self-block, not the cross-strategy rule
        sig_px, tgt, stop = (float(m.group(x)) for x in (5, 6, 7))
        qty = int(m.group(8))
        oc, pnl, xts = simulate(tk, t, sig_px, tgt, stop, ddl_by.get(strat, 2), qty)
        recs.append(dict(strat=strat, tk=tk, bar=bar, reason="RULE-BLOCK",
                         sig_px=sig_px, qty=qty, tgt=tgt, stop=stop,
                         outcome=oc, pnl=pnl, xts=xts, ets=str(t)))
        seen.add(key)
        open_until[(strat, tk)] = (pd.Timestamp(xts) if xts
                                   else t + pd.Timedelta(days=ddl_by.get(strat, 2)))
        print(f"  RECOVERED {strat} {tk} bar={bar} (rule-block) -> {oc} {pnl:+.2f}")
    return recs


def main():
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else \
        {"open": [], "pending": [], "closed": []}
    prev = json.loads(OUT.read_text()) if OUT.exists() else []
    seen = {f"{r['strat']}|{r['tk']}|{r['bar']}" for r in prev}
    recs = recover_rule_blocks(seen, led, prev) + recover_outages(seen, led, prev)
    allr = prev + recs
    OUT.write_text(json.dumps(allr, indent=1))
    tot = sum(r["pnl"] for r in allr)
    print(f"\n{len(recs)} newly recovered ({len(allr)} total) | "
          f"recovered P&L ${tot:+,.2f} -> {OUT}")


if __name__ == "__main__":
    main()
