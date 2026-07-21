"""
morning_opt_book.py — REAL-OPTIONS REPLAY of the morning BOOK config on its
options-viable legs (QQQ + SPY only; DIA/XLK/MDY have no daily expiries / thin
prints — established in today's coverage probes).

Book champion status: NONE (0 of 6 book configs passed the arena). Per the
orchestrator's instruction the replay therefore runs on the BEST NEAR-MISS
config: "QQQ+SPY nr<=0.3" (arena worst -1.48%, freq 0.37 — it passed nothing;
this is a diagnostic look, not a validated strategy).

For each ticker: stock legs via morning_opt.orb2s_trade(day, 5, 2.0,
nr=BOOK_NR, last_entry=690, sides='both') over 2024-07-22..now (usable Polygon
option-minute window), replayed on REAL option 5-min bars in cells
{0dte/atm, 0dte/otm1, 1-3d/atm}. Costs 1%/side of premium; 2%/side sensitivity
recomputed from raw bar closes. Era split at 2025-07-14 (gate-era vs final-era).

KEY ANALYSIS — marginal vs core: trades split into CORE (nr_rank <= 0.3, the
already-validated vMO bucket) and MARGINAL (added by a wider book bucket,
0.3 < nr_rank <= BOOK_NR). With BOOK_NR = 0.3 the marginal set is EMPTY by
construction (nr_rank is quantized in 1/6 steps: only ranks 0 and 1/6 fire) —
the script still performs the split generically so it stays repeatable if a
wider bucket (0.4 / 0.55) ever becomes the book champion.

Research only — no orders. Usage: python morning_opt_book.py
Output: runs/morning_options_book.txt
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from morning_qqq import days
from morning_qqq3 import add_nr
from morning_validate import load_ticker
from morning_opt import orb2s_trade, replay

BOOK_NAME = "QQQ+SPY nr<=0.3 (best near-miss; book champion = NONE)"
BOOK_NR = 0.3                    # book bucket replayed
CORE_NR = 0.3                    # validated vMO bucket -> CORE/MARGINAL split
TICKERS = ("QQQ", "SPY")         # options-viable legs only
START = dt.date(2024, 7, 22)     # rolling 2y Polygon option-minute floor
ERA_CUT = dt.date(2025, 7, 14)   # gate-era / final-era boundary
CELLS = [("0dte", "atm"), ("0dte", "otm1"), ("1-3d", "atm")]

RUNS = Path(__file__).with_name("runs")
RUNS.mkdir(exist_ok=True)
OUT = RUNS / "morning_options_book.txt"


def book_trades(tk):
    """Stock legs of the book config for one ticker, nr_rank attached."""
    dd = days(load_ticker(tk))
    add_nr(dd)
    out = []
    for d in dd:
        if d["date"] < START:
            continue
        t = orb2s_trade(d, 5, 2.0, nr=BOOK_NR, last_entry=690, sides="both")
        if t is not None:
            t["nr_rank"] = d["nr_rank"]
            out.append(t)
    return out


def opt_rets(rows, cost=0.01):
    """Per-trade option returns of FILLED rows at `cost` per side, recomputed
    from raw bar closes (matches morning_opt convention at cost=0.01)."""
    return np.array([(r["opt_exit"] * (1 - cost) - r["opt_entry"] * (1 + cost))
                     / (r["opt_entry"] * (1 + cost))
                     for r in rows if r["status"] == "filled"], dtype=float)


def stats(rows, cost=0.01):
    n = len(rows)
    filled = [r for r in rows if r["status"] == "filled"]
    rr = opt_rets(rows, cost)
    s = {"n": n, "filled": len(filled),
         "fill_rate": len(filled) / n if n else None,
         "win": float((rr > 0).mean()) if len(rr) else None,
         "avg": float(rr.mean()) if len(rr) else None,
         "med": float(np.median(rr)) if len(rr) else None,
         "tot": float(rr.sum()) if len(rr) else None}
    for tag, keep in (("era1", lambda d: d < ERA_CUT),
                      ("era2", lambda d: d >= ERA_CUT)):
        sub = [r for r in filled if keep(r["date"])]
        er = opt_rets(sub, cost)
        s[tag] = {"n": len(sub), "tot": float(er.sum()) if len(er) else None}
    return s


def pct(x, d=1):
    return "None" if x is None else f"{x * 100:+.{d}f}%"


def fmt(s, prefix=""):
    line = (f"{prefix}n={s['n']} filled={s['filled']}"
            f" (fill {s['fill_rate']:.0%})" if s["n"] else f"{prefix}n=0")
    if s.get("filled"):
        line += (f" | win {s['win']:.0%} avg {pct(s['avg'])} med {pct(s['med'])}"
                 f" tot {pct(s['tot'])}"
                 f" | era1(gate) {pct(s['era1']['tot'])} (n={s['era1']['n']})"
                 f" / era2(final) {pct(s['era2']['tot'])} (n={s['era2']['n']})")
    return line


def main():
    lines = []

    def emit(s=""):
        print(s, flush=True)
        lines.append(s)

    emit(f"MORNING OPTIONS BOOK REPLAY — {BOOK_NAME}")
    emit(f"run at {dt.datetime.now():%Y-%m-%d %H:%M} | window {START}..now "
         f"(Polygon option-minute floor) | era cut {ERA_CUT}")
    emit("legs: QQQ + SPY only (DIA/XLK/MDY not options-viable) | costs "
         "1%/side of premium, 2%/side sensitivity on best cell | stock 2bps")
    emit(f"CORE = nr_rank <= {CORE_NR} (validated vMO bucket); MARGINAL = "
         f"{CORE_NR} < nr_rank <= {BOOK_NR} (book bucket == core bucket -> "
         "marginal EMPTY by construction)")
    emit()

    table = []
    for tk in TICKERS:
        trades = book_trades(tk)
        core_t = [t for t in trades if t["nr_rank"] <= CORE_NR]
        marg_t = [t for t in trades if t["nr_rank"] > CORE_NR]
        ranks = sorted({round(t["nr_rank"], 4) for t in trades})
        stock = np.array([t["stock_ret_net"] for t in trades])
        emit("=" * 100)
        emit(f"TICKER {tk} — {len(trades)} trades "
             f"{trades[0]['date']}..{trades[-1]['date']} | distinct nr_ranks "
             f"fired: {ranks} | core n={len(core_t)} marginal n={len(marg_t)}")
        emit(f"  STOCK (2bps): tot {pct(stock.sum(), 2)} "
             f"avg {stock.mean() * 1e4:+.1f}bp win {(stock > 0).mean():.0%}")
        for bucket, mode in CELLS:
            res = replay(trades, bucket, mode, underlying=tk)
            for r, t in zip(res, trades):
                r["nr_rank"] = t["nr_rank"]
            s_all = stats(res)
            s2 = stats(res, cost=0.02)
            core_r = [r for r in res if r["nr_rank"] <= CORE_NR]
            marg_r = [r for r in res if r["nr_rank"] > CORE_NR]
            emit(f"  --- {bucket}/{mode} ---")
            emit(fmt(s_all, "    ALL      "))
            emit(f"    2%/side  avg {pct(s2['avg'])} tot {pct(s2['tot'])}"
                 f" win {s2['win']:.0%}" if s2["filled"] else "    2%/side  n/a")
            emit(fmt(stats(core_r), "    CORE     "))
            emit(fmt(stats(marg_r), "    MARGINAL "))
            table.append({"ticker": tk, "dte": bucket, "strike": mode,
                          "all": s_all, "all_2pct": s2,
                          "core": stats(core_r), "marginal": stats(marg_r)})
        emit()

    emit("=" * 100)
    emit("BOOK VIEW (equal-weight sum of the two option legs, per cell)")
    for bucket, mode in CELLS:
        rows = [r for r in table if r["dte"] == bucket and r["strike"] == mode]
        tot = sum(r["all"]["tot"] for r in rows)
        e1 = sum(r["all"]["era1"]["tot"] for r in rows)
        e2 = sum(r["all"]["era2"]["tot"] for r in rows)
        n = sum(r["all"]["n"] for r in rows)
        emit(f"  {bucket}/{mode}: n={n} tot {pct(tot)} | era1(gate) {pct(e1)}"
             f" / era2(final) {pct(e2)}")
    best = max(table, key=lambda r: r["all"]["tot"] or -9e9)
    emit()
    emit(f"BEST CELL: {best['ticker']} {best['dte']}/{best['strike']} — "
         + fmt(best["all"]).strip())
    emit(f"  2%/side sensitivity: avg {pct(best['all_2pct']['avg'])} "
         f"med {pct(best['all_2pct']['med'])} tot {pct(best['all_2pct']['tot'])} "
         f"win {best['all_2pct']['win']:.0%}")
    emit()
    emit("HONESTY: this config passed NOTHING (arena worst -1.48%); the replay "
         "window (2024-07-22..now) covers only gate+final eras and its gate/"
         "final stock performance was never examined in the selection ladder. "
         "Marginal set is empty because the near-miss bucket IS the core "
         "bucket — the marginal-vs-core decision is untestable on this config. "
         "Unfillable/no-contract trades never enter P&L.")

    OUT.write_text("\n".join(lines) + "\n\nJSON_TABLE = " +
                   json.dumps(table, default=str, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
