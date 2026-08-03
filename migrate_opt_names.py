"""
migrate_opt_names.py — one-shot rename of the options books in the ledger.

  vCO -> vC-OPT-2W      (vC signals, 1-2 week ATM calls)
  vMO -> vM-OPT-0DTE    (vM signals, same-day calls/puts)

Backs the ledger up first. Idempotent: re-running finds nothing to do.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from alpaca_bot2 import OPT_ALIASES

LEDGER = Path("runs/alpaca2_ledger.json")


def main():
    led = json.loads(LEDGER.read_text())
    n = 0
    for bucket in ("opt_open", "opt_closed", "opt_queue"):
        for x in led.get(bucket, []):
            old = x.get("strat", "vCO")
            if old in OPT_ALIASES:
                x["strat"] = OPT_ALIASES[old]
                n += 1
    if not n:
        print("nothing to migrate (already renamed)")
        return
    bak = LEDGER.with_suffix(".json.bak-optnames")
    shutil.copy2(LEDGER, bak)
    LEDGER.write_text(json.dumps(led, indent=1, default=str))
    print(f"renamed {n} option records -> {OPT_ALIASES}")
    print(f"backup: {bak.name}")


if __name__ == "__main__":
    main()
