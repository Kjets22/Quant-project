"""
daily_report.py — end-of-day report: MARKET arm vs LIMIT arm, broken down by strategy
version, plus automated health checks.

Issue policy (as delegated):
  AUTO-FIX (clearly wrong): pending entry orders past their cancel window that are still
    live at the broker -> cancelled here as a safety net (and flagged).
  FLAG ONLY (judgment calls): broker/ledger position mismatches (could be a manual trade),
    time-exits due (the bot handles them next cycle), high slippage, stale scheduler
    heartbeat, missing daily models, halted state.

Writes runs/daily_reports/YYYY-MM-DD.txt (committed to the repo by the scheduled task).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

import alpaca_api as broker

LEDGER = Path("runs/alpaca2_ledger.json")
TASK_LOG = Path("runs/alpaca_task.log")
STRATS = ["v3", "v4", "v6", "v7", "vC", "vQ", "vQ2"]
SLIP_BUDGET_BPS = 5.0


def main():
    now = pd.Timestamp.utcnow().tz_localize(None)
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else None
    acct = broker.account()
    clk = broker.clock()
    bpos = {p["symbol"]: p for p in broker.positions()}
    equity = float(acct["equity"])
    day = equity - float(acct["last_equity"])

    L = []
    def w(s=""):
        L.append(s)

    w(f"DAILY REPORT — {datetime.now():%Y-%m-%d %H:%M} local  "
      f"(market {'OPEN' if clk['is_open'] else 'closed'})")
    w(f"account equity ${equity:,.2f}   day P&L {day:+.2f}   cash ${float(acct['cash']):,.2f}")
    w("")
    flags = []
    if led is None:
        flags.append("CRITICAL: ledger missing — bot has never run?")
        led = {"open": [], "pending": [], "closed": [], "state": {}}

    arm_tot = {}
    for style, name in (("mkt", "MARKET arm"), ("lmt", "LIMIT arm")):
        w(f"===== {name} =====")
        w(f"  {'strat':>5} {'closed':>7} {'win%':>6} {'missed':>7} {'realized$':>10} "
          f"{'open':>5} {'unreal$':>9} {'avg slip':>9}")
        tot_r = tot_u = 0.0
        tot_cl = tot_w = tot_ms = tot_op = 0
        for st in STRATS:
            cl = [x for x in led["closed"]
                  if x["style"] == style and x["strat"] == st and x["outcome"] != "MISSED"]
            ms = sum(1 for x in led["closed"]
                     if x["style"] == style and x["strat"] == st and x["outcome"] == "MISSED")
            op = [x for x in led["open"] if x["style"] == style and x["strat"] == st]
            wins = sum(x["outcome"] == "TARGET" for x in cl)
            rp = sum(x.get("pnl") or 0 for x in cl)
            up = 0.0
            for x in op:
                cur = float(bpos[x["tk"]]["current_price"]) if x["tk"] in bpos else x["fill"]
                up += (cur - x["fill"]) * x["qty"]
            slips = [(x["fill"] - x["sig_px"]) / x["sig_px"] * 1e4
                     for x in (cl + op) if x.get("fill")]
            sl = f"{np.mean(slips):+.1f}bp" if slips else "-"
            wp = f"{wins/len(cl):.0%}" if cl else "-"
            w(f"  {st:>5} {len(cl):>7} {wp:>6} {ms:>7} {rp:>+10.2f} "
              f"{len(op):>5} {up:>+9.2f} {sl:>9}")
            tot_r += rp; tot_u += up
            tot_cl += len(cl); tot_w += wins; tot_ms += ms; tot_op += len(op)
        wp = f"{tot_w/tot_cl:.0%}" if tot_cl else "-"
        w(f"  {'TOTAL':>5} {tot_cl:>7} {wp:>6} {tot_ms:>7} {tot_r:>+10.2f} "
          f"{tot_op:>5} {tot_u:>+9.2f}")
        w("")
        arm_tot[style] = tot_r + tot_u
        # arm-level slippage flag
        aslips = [(x["fill"] - x["sig_px"]) / x["sig_px"] * 1e4
                  for x in led["closed"] + led["open"]
                  if x["style"] == style and x.get("fill")]
        if style == "mkt" and aslips and np.mean(aslips) > SLIP_BUDGET_BPS:
            flags.append(f"WARN: market-arm avg slippage {np.mean(aslips):+.1f} bps exceeds "
                         f"the {SLIP_BUDGET_BPS:.0f} bps budget — edge margins at risk.")

    diff = arm_tot.get("mkt", 0) - arm_tot.get("lmt", 0)
    lead = "MARKET" if diff > 0 else "LIMIT" if diff < 0 else "tied"
    w(f"A/B VERDICT SO FAR: {lead} arm leads by ${abs(diff):,.2f} "
      f"(mkt ${arm_tot.get('mkt',0):+,.2f} vs lmt ${arm_tot.get('lmt',0):+,.2f}, "
      f"incl. unrealized)")
    w("")

    # ---------- health checks ----------
    # 1. AUTO-FIX: pending orders past expiry that are still live at the broker
    for p in list(led["pending"]):
        if pd.Timestamp(p["expiry"]) <= now:
            try:
                o = broker.get_order(p["order_id"])
                if o["status"] in ("new", "accepted", "held", "pending_new"):
                    broker.cancel_order(p["order_id"])
                    flags.append(f"FIXED: stale pending {p['style']} {p['strat']} {p['tk']} "
                                 f"order past expiry was still live -> cancelled.")
            except Exception as e:
                flags.append(f"WARN: could not verify pending {p['tk']}: {e}")
    # 2. ledger vs broker reconciliation. Pending entries may or may not be filled yet,
    #    so the broker qty must lie in [open_qty, open_qty + pending_qty] per symbol.
    open_q, pend_q = {}, {}
    for x in led["open"]:
        open_q[x["tk"]] = open_q.get(x["tk"], 0) + x["qty"]
    for x in led["pending"]:
        pend_q[x["tk"]] = pend_q.get(x["tk"], 0) + x["qty"]
    syms = set(bpos) | set(open_q) | set(pend_q)
    for sym in syms:
        bq = int(float(bpos[sym]["qty"])) if sym in bpos else 0
        lo, hi = open_q.get(sym, 0), open_q.get(sym, 0) + pend_q.get(sym, 0)
        if not (lo <= bq <= hi):
            flags.append(f"FLAG: {sym} broker qty {bq} outside ledger range [{lo},{hi}] "
                         f"(manual trade? partial fill? review before trusting stats)")
    # 3. time exits due
    for x in led["open"]:
        if pd.Timestamp(x["deadline"]) <= now:
            flags.append(f"NOTE: {x['style']} {x['strat']} {x['tk']} past time-exit deadline — "
                         f"bot closes it next open cycle.")
    # 4. scheduler heartbeat
    if TASK_LOG.exists():
        lines = [ln for ln in TASK_LOG.read_text(encoding="utf-8", errors="ignore").splitlines()
                 if "cycle" in ln]
        if lines:
            last = lines[-1][:20].strip()
            try:
                last_ts = pd.Timestamp(last.replace("Z", ""))
                if clk["is_open"] and (now - last_ts) > pd.Timedelta(minutes=45):
                    flags.append(f"CRITICAL: last bot cycle was {last_ts} UTC (>45 min ago "
                                 f"during market hours) — is the scheduled task running?")
            except Exception:
                pass
    # 5. model freshness (newest pickle within 36h — avoids UTC-date false alarms at night)
    pkls = list(Path("models").glob("*.pkl"))
    if pkls:
        newest = max(p.stat().st_mtime for p in pkls)
        age_h = (datetime.now().timestamp() - newest) / 3600
        if age_h > 36:
            flags.append(f"WARN: newest model is {age_h:.0f}h old — daily retraining may have stopped.")
    else:
        flags.append("WARN: no models exist — the bot has never trained.")
    # 6. halted
    if led.get("state", {}).get("halted"):
        flags.append("CRITICAL: drawdown breaker is HALTED — no new entries until reviewed.")

    w("===== FLAGS =====")
    if flags:
        for f in flags:
            w("  " + f)
    else:
        w("  none — all systems normal")

    report = "\n".join(L)
    print(report)
    out = Path("runs/daily_reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{datetime.now():%Y-%m-%d}.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
