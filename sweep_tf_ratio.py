"""
sweep_tf_ratio.py — grid sweep of candle timeframe x target:stop ratio (incl 4:1),
using the SAME engine as the working strategy: base+S/R features, top-7% selectivity,
3 bps cost, H=24 bars, walk-forward, pooled over the 8-name basket. Reuses
timeframe_test.run so the methodology is byte-identical to the validated 30-min/1.5:1.

Standalone study. Touches no frozen snapshot (v1/v2/v3) or working file.

  python sweep_tf_ratio.py 60,30        # coarse candles (fast)
  python sweep_tf_ratio.py 15,5         # fine candles (slow; 5-min is the cache floor)
  python sweep_tf_ratio.py 60,30,15,5   # everything
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from timeframe_test import run   # run(minutes, tp, sl) -> (n, win, z, total%, mean bps, avgMove%)

RATIOS = [(1, 1), (1.5, 1), (2, 1), (3, 1), (4, 1)]


def emit(line, fh):
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def main():
    tfs = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "60,30,15,5").split(",")]
    out = Path("runs/sweep_results.txt")
    out.parent.mkdir(exist_ok=True)
    with out.open("a") as fh:
        for mins in tfs:
            emit(f"\n=== {mins}-min candles  (top-7%, 3 bps, base+S/R, 8 names) ===", fh)
            emit(f"  {'tgt:stop':>8} {'breakeven':>9} {'trades':>7} {'win%':>6} "
                 f"{'avgMove%':>9} {'mean bps':>9} {'total%':>8}", fh)
            for tp, sl in RATIOS:
                n, wr, z, tot, bps, mv = run(mins, tp, sl)
                be = sl / (sl + tp)
                emit(f"  {tp:g}:{sl:g}     {be:>9.0%} {n:>7} {wr:>6.1%} {mv:>9.2f} "
                     f"{bps:>+9.1f} {tot:>+8.0f}", fh)
    print("\nREAD: wider target (3:1, 4:1) needs a much lower win% to break even, but the edge")
    print("is short-horizon so accuracy fades as the target moves out. Best cell = highest total%")
    print("with win% comfortably above its break-even and mean bps clearly positive.")


if __name__ == "__main__":
    main()
