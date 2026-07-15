"""
sizing_sim.py — does 10%-of-a-$10k-account sizing (compounding) beat flat $1k/trade?

Replays each strategy's final-year trade sequence two ways:
  FLAT:     $1,000 notional on every trade (what the backtests report)
  COMPOUND: $10,000 account, every trade sized at 10% of CURRENT equity
Reports both totals plus the compounded account's max drawdown. Also runs vCO
(vC's option legs, real fills, 1% side cost) where per-trade returns are large
enough for compounding to actually matter.

Note: trades are applied in entry order; overlapping multi-ticker holds (vC) are
approximated as sequential — fine at 10% allocation.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from qqq_options_real import gen_trades as qqq_gen
import vc_options_real as vco

QQQ = [("vQ",  "dollar", 2.0,   2.0,  12, ("conf", 0.90), "lgbm"),
       ("vQ2", "dollar", 2.5,   2.0,  24, ("conf", 0.90), "histgb"),
       ("vA",  "dollar", 1.5,   2.0,  48, ("conf", 0.95), "lgbm"),
       ("vP",  "dollar", 2.0,   2.0,  96, ("conf", 0.85), "histgb"),
       ("vR",  "pct",    0.004, 0.002, 24, ("q", 0.97),   "lgbm"),
       ("vS",  "pct",    0.005, 0.004, 96, ("q", 0.90),   "histgb")]
ACCT, FRAC = 10_000.0, 0.10


def compound(rets):
    eq, peak, mdd = ACCT, ACCT, 0.0
    for r in rets:
        eq *= (1 + FRAC * r)
        peak = max(peak, eq)
        mdd = max(mdd, (peak - eq) / peak)
    return eq - ACCT, mdd


def report(name, rets):
    r = np.array(rets)
    flat = 1000.0 * r.sum()
    comp, mdd = compound(rets)
    print(f"  {name:>4} {len(r):>4} {r.sum() * 100:>+7.2f}% {flat:>+9.2f} "
          f"{comp:>+9.2f} {comp - flat:>+7.2f} {mdd * 100:>5.1f}%")


def main():
    print("SIZING: flat $1k/trade vs $10k account @ 10%/trade (final year)")
    print(f"  {'strat':>4} {'n':>4} {'sum(r)':>8} {'flat $':>9} {'10%comp $':>9} "
          f"{'diff':>7} {'maxDD':>6}")
    for name, mode, tp, sl, H, sel, mname in QQQ:
        trades, _, _ = qqq_gen(name, mode, tp, sl, H, sel, mname)
        trades.sort(key=lambda t: t["t0"])
        report(name, [t["stock_ret"] for t in trades])
        sys.stdout.flush()
    vc_all, tsmap = [], {}
    for tk in vco.TICKERS:
        r = vco.gen_trades(tk)
        if not r:
            continue
        trades, ts_all, c_all = r
        tsmap[tk] = (ts_all, c_all)
        vc_all += trades
    vc_all.sort(key=lambda t: t["t0"])
    report("vC", [t["stock_ret"] for t in vc_all])
    # vCO: real option fills (1-2w ATM), 1% per-side cost, entry order per ticker
    fills = []
    for tk in tsmap:
        tt = [t for t in vc_all if t["tk"] == tk]
        f, _, _ = vco.replay(tt, 5, 14, "ATM", *tsmap[tk])
        fills += f
    hc = 0.01
    report("vCO", [(x * (1 - hc) - e * (1 + hc)) / (e * (1 + hc)) for e, x, _ in fills])
    print("\n  diff = compounding effect. maxDD = worst peak-to-trough of the $10k")
    print("  account. vCO per-trade returns are on PREMIUM (10% of acct = the premium).")


if __name__ == "__main__":
    main()
