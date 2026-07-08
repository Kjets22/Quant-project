"""
paper_ensemble_v4.py — 5 strategies, each with its OWN $10,000 paper account.
No cross-strategy competition: every strategy takes its own trades ($1,000 each,
max 10 open at a time, guardrails per account). Reports per-account blotters,
END-OF-DAY realized P&L, and unrealized P&L on open positions (marked at last close).

  python paper_ensemble_v4.py 2026-06-29 2026-07-07
Research/paper only — places no orders.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from paper_ensemble_v3 import candidates, CONFIGS, _C5

ACCOUNT = 10_000.0
NOTIONAL = 1_000.0                 # $1,000 per trade = 10 concurrent max fills the account
MAX_POSITIONS = 10
DAILY_LOSS_LIMIT = ACCOUNT * 0.02
DD_BREAKER = ACCOUNT * 0.15
SLIP_BPS, COST_BPS = 1.0, 3.0


def simulate(cand):
    """Independent single-account portfolio pass. Returns (taken, skipped)."""
    eff = (COST_BPS + 2 * SLIP_BPS) / 1e4
    open_pos, taken = [], []
    skipped = {"cap": 0, "daily": 0, "dd": 0}
    daily, cum, peak, halted = {}, 0.0, 0.0, False
    for t in sorted(cand, key=lambda x: x["ets"]):
        for p in [p for p in open_pos if p["xts"] is not None and p["xts"] <= t["ets"]]:
            open_pos.remove(p)
        if halted:
            skipped["dd"] += 1; continue
        if daily.get(t["ets"].date(), 0.0) <= -DAILY_LOSS_LIMIT:
            skipped["daily"] += 1; continue
        if len(open_pos) >= MAX_POSITIONS:
            skipped["cap"] += 1; continue
        sh = NOTIONAL / t["entry"]
        t["shares"] = round(sh, 2)
        if t["exit"] is None:
            t["pnl"] = None
            mark = float(_C5[t["tk"]]["close"].iloc[-1])
            t["mark"] = mark
            t["unreal"] = round(sh * (mark - t["entry"]), 2)
        else:
            t["pnl"] = round(sh * (t["exit"] - t["entry"]) - NOTIONAL * eff, 2)
            xd = t["xts"].date()
            daily[xd] = daily.get(xd, 0.0) + t["pnl"]
            cum += t["pnl"]; peak = max(peak, cum)
            if peak - cum >= DD_BREAKER:
                halted = True
        taken.append(t); open_pos.append(t)
    return taken, skipped, daily


def main():
    start = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp("2026-06-29")
    end = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else start + pd.Timedelta(days=7)
    fetch_end = str((pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).date())
    print(f"PAPER v4 — 5 INDEPENDENT $10,000 accounts | $1,000/trade, max 10 open each")
    print(f"window {start.date()} .. {end.date()}  (data thru {fetch_end})\n")

    all_daily = {}
    ledger = {}
    summary = []
    for cfg in CONFIGS:
        name = cfg[0]
        cand = candidates(*cfg, start, end, fetch_end)
        taken, skipped, daily = simulate(cand)
        ledger[name] = taken
        closed = [t for t in taken if t["pnl"] is not None]
        wins = sum(t["outcome"] == "TARGET" for t in closed)
        pnl = sum(t["pnl"] for t in closed)
        opens = [t for t in taken if t["pnl"] is None]
        unreal = sum(t["unreal"] for t in opens)
        for d, v in daily.items():
            all_daily.setdefault(d, {})[name] = v
        summary.append((name, len(closed), wins, pnl, len(opens), unreal, skipped))
        print(f"===== {name}: account ${ACCOUNT + pnl:,.2f}  "
              f"(realized ${pnl:+,.2f}, unrealized ${unreal:+,.2f}) =====")
        for t in taken:
            if t["pnl"] is None:
                tail = f"OPEN @mark {t['mark']:.2f}  unreal {t['unreal']:+.2f}"
            else:
                tail = f"{t['outcome']:>6}  {t['pnl']:+.2f}"
            print(f"   {t['tk']:>5} {str(t['ets'])[5:16]}  in {t['entry']:>8.2f}  "
                  f"tgt {t['tgt']:>8.2f}  stop {t['stop']:>8.2f}  {tail}")
        wp = f"{wins/len(closed):.0%}" if closed else "-"
        print(f"   closed={len(closed)} wins={wins} ({wp})  skips: cap={skipped['cap']} "
              f"daily={skipped['daily']} dd={skipped['dd']}\n")

    print("=== END-OF-DAY realized P&L ($) ===")
    names = [c[0] for c in CONFIGS]
    print("  " + f"{'date':>10} " + " ".join(f"{n:>8}" for n in names) + f" {'ALL':>9}")
    for d in sorted(all_daily):
        row = all_daily[d]
        tot = sum(row.values())
        print("  " + f"{str(d):>10} " + " ".join(f"{row.get(n, 0):>+8.2f}" for n in names)
              + f" {tot:>+9.2f}")

    print("\n=== SCOREBOARD (each started with $10,000) ===")
    print(f"  {'strat':>5} {'closed':>7} {'win%':>6} {'realized':>10} {'open':>5} "
          f"{'unrealized':>11} {'account (incl unreal)':>22}")
    for name, ncl, wins, pnl, nop, unreal, _ in summary:
        wp = f"{wins/ncl:.0%}" if ncl else "-"
        print(f"  {name:>5} {ncl:>7} {wp:>6} {pnl:>+10.2f} {nop:>5} {unreal:>+11.2f} "
              f"{ACCOUNT + pnl + unreal:>22,.2f}")
    Path("runs").mkdir(exist_ok=True)
    Path("runs/paper_v4_ledger.json").write_text(json.dumps(
        {k: [{kk: (str(vv) if isinstance(vv, pd.Timestamp) else vv) for kk, vv in t.items()}
             for t in v] for k, v in ledger.items()}, indent=1))
    print("  ledger -> runs/paper_v4_ledger.json")


if __name__ == "__main__":
    main()
