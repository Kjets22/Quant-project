"""
trade_log.py — what actually traded on given day(s): entries, exits, skips.

  python trade_log.py                 today
  python trade_log.py 2026-07-31      one day
  python trade_log.py 2026-07-31 today   several

Reads the ledger (authoritative fills/exits) and the bot log (signals/skips).
Read-only.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    today = str(pd.Timestamp.now(tz="America/New_York").date())
    days = [today if a == "today" else a for a in args] or [today]

    led = json.loads(Path("runs/alpaca2_ledger.json").read_text())
    log = Path("runs/alpaca_log.txt").read_text(encoding="utf-8", errors="ignore")
    allpos = (led["open"] + led["closed"] + led.get("opt_open", [])
              + led.get("opt_closed", []))

    for day in days:
        print("=" * 72)
        print(f"  {day}")
        print("=" * 72)

        ents = [x for x in allpos if (x.get("ets") or "").startswith(day)]
        exts = [x for x in led["closed"] + led.get("opt_closed", [])
                if (x.get("xts") or "").startswith(day)]

        print(f"\nENTRIES ({len(ents)})")
        if not ents:
            print("  none")
        for x in sorted(ents, key=lambda z: z.get("ets") or ""):
            t = (x.get("ets") or "")[11:19]
            if "occ" in x:
                print(f"  {t}  {x.get('strat', 'vCO'):>4} OPTION {x['occ']} "
                      f"x{x['qty']} @ {x.get('fill')}")
            else:
                side = "SHORT" if x.get("side") == "short" else "long"
                slip = ((x["fill"] - x["sig_px"]) / x["sig_px"] * 1e4
                        if x.get("fill") else 0)
                print(f"  {t}  {x['strat']:>4} {x['style']} {side:>5} {x['tk']:>5} "
                      f"{x['qty']:>3}sh @ {x['fill']:>8.2f}  slip {slip:>+6.1f}bp")

        print(f"\nEXITS ({len(exts)})")
        if not exts:
            print("  none")
        tot = 0.0
        for x in sorted(exts, key=lambda z: z.get("xts") or ""):
            t = (x.get("xts") or "")[11:19]
            pnl = x.get("pnl") or 0
            tot += pnl
            if "occ" in x:
                print(f"  {t}  {x.get('strat', 'vCO'):>4} OPTION {x['occ']} "
                      f"{x.get('exit_reason', ''):>6}  {pnl:>+9.2f}")
            else:
                print(f"  {t}  {x['strat']:>4} {x['style']} {x['tk']:>5} "
                      f"{x['outcome']:>7}  {pnl:>+9.2f}")
        print(f"  {'':>10}{'REALIZED TOTAL':>32}  {tot:>+9.2f}")

        sk = re.findall(rf"^{day} (\d\d:\d\d:\d\d)Z\s+\[skip ([^\]]+)\]", log, re.M)
        mi = re.findall(rf"^{day} (\d\d:\d\d:\d\d)Z\s+MISSED (\w+) (\w+) (\w+)",
                        log, re.M)
        print(f"\nSKIPPED by the stale-entry guard ({len(sk)})")
        for t, s in sk:
            print(f"  {t}  {s[:95]}")
        if not sk:
            print("  none")
        print(f"\nUNFILLED limit orders ({len(mi)})")
        for t, arm, st, tk in mi:
            print(f"  {t}  {st:>4} {arm} {tk}")
        if not mi:
            print("  none")
        print()


if __name__ == "__main__":
    main()
