"""
morning_book.py — BOOK-LEVEL LADDER for vM across ticker baskets.

The tradeable unit is the equal-weight BOOK, not a single ticker: each day every
member ticker runs vM = orb2s(or_bars=5, rr=2.0, nr=X, last_entry=690, sides='both')
independently; the day's book return = mean of the ACTIVE members' trade returns
(days with no trades = flat, excluded from the trade-day count).

Pre-registered grid (6 configs, no other tuning):
    members in { [QQQ,SPY], [QQQ,SPY,DIA] }  x  nr in { 0.3, 0.4, 0.55 }

Ladder (same honesty discipline as all morning rounds):
    ARENA  = worst of 5 half-year book totals 2022-01-14..2024-07-14
             (each half needs >= 8 book trade-days) — selection on arena worst ONLY
    GATE   = 2024-07-14..2025-07-14, ONE look, arena survivors only
    FINAL  = 2025-07-14..now, ONE look, gate survivors only, 0/2/5bps recomputed
             per leg, portfolio monthly Sharpe, maxDD, per-leg contribution
CHAMPION = full-ladder passer with FINAL-year union trade-day freq >= 0.48,
best final total, tie-break arena worst. Research only — no orders.

Usage:  python morning_book.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from morning_qqq import COST, SUBS, GATE, FINAL, days
from morning_qqq3 import add_nr
from morning_qqq4 import orb2s
from morning_validate import load_ticker

TICKERS = ["QQQ", "SPY", "DIA"]
NRS = [0.3, 0.4, 0.55]
MEMBER_SETS = [("QQQ", "SPY"), ("QQQ", "SPY", "DIA")]
VM = dict(or_bars=5, rr=2.0, last_entry=690, sides="both")
BPS = 1e-4


def in_win(dt, lo, hi):
    return pd.Timestamp(lo).date() <= dt < pd.Timestamp(hi).date()


def build_all():
    """Per-ticker day records once; per-nr RAW trade returns (cost added back)."""
    day_recs, trades = {}, {}
    for tk in TICKERS:
        dd = days(load_ticker(tk))
        add_nr(dd)
        day_recs[tk] = dd
        trades[tk] = {}
        for nr in NRS:
            t = {}
            for d in dd:
                r = orb2s(d, nr=nr, **VM)
                if r is not None:
                    t[d["date"]] = r + COST          # raw, before costs
            trades[tk][nr] = t
    return day_recs, trades


def fire_rate_report(day_recs, trades):
    print("=== FIRE-RATE VERIFICATION of nr buckets (nr_rank quantized in 1/6 steps) ===")
    for tk in TICKERS:
        dd = day_recs[tk]
        ranks = [d["nr_rank"] for d in dd if d["nr_rank"] is not None]
        n = len(ranks)
        vals = sorted(set(round(r, 4) for r in ranks))
        dist = {v: sum(1 for r in ranks if round(r, 4) == v) for v in vals}
        print(f"  {tk}: {n} ranked days | distinct nr_rank values: "
              f"{ {v: dist[v] for v in vals} }")
        for nr in NRS:
            elig = sum(1 for r in ranks if r <= nr)
            fired = len(trades[tk][nr])
            print(f"    nr<={nr:<4}  eligible {elig}/{n} = {elig/n:.1%}   "
                  f"fired {fired}/{n} = {fired/n:.1%}")
    print()


def book_days(members, nr, trades, cal):
    """[(date, {tk: raw_ret})] for calendar days with >= 1 active member."""
    out = []
    for dt in cal:
        legs = {tk: trades[tk][nr][dt] for tk in members if dt in trades[tk][nr]}
        if legs:
            out.append((dt, legs))
    return out


def book_rets(rows, cost):
    """Book day returns at a given per-leg round-trip cost."""
    return np.array([np.mean([r - cost for r in legs.values()]) for _, legs in rows])


def main():
    day_recs, trades = build_all()
    cals = {}
    for ms in MEMBER_SETS:
        common = set(d["date"] for d in day_recs[ms[0]])
        for tk in ms[1:]:
            common &= set(d["date"] for d in day_recs[tk])
        cals[ms] = sorted(common)
    for tk in TICKERS:
        dd = day_recs[tk]
        print(f"loaded {tk}: {len(dd)} days {dd[0]['date']}..{dd[-1]['date']}")
    for ms in MEMBER_SETS:
        print(f"aligned calendar {'+'.join(ms)}: {len(cals[ms])} days")
    print()

    fire_rate_report(day_recs, trades)

    configs = [(ms, nr) for ms in MEMBER_SETS for nr in NRS]
    print(f"BOOK LADDER: {len(configs)} pre-registered configs | vM "
          f"or_bars=5 rr=2.0 last_entry=690 both | book = mean of active legs | "
          f"headline {COST*1e4:.0f}bps\n")

    # ---------------------------------------------------------------- arena
    print(f"  {'config':<22} {'worst':>7}  {'subs 2022H1..2024H1':<42} n/half   "
          f"freq ovr/gate/fin")
    arena_pass, stats = [], {}
    for ms, nr in configs:
        cal = cals[ms]
        rows = book_days(ms, nr, trades, cal)
        rd = dict((dt, legs) for dt, legs in rows)
        subs, ns = [], []
        for lo, hi in SUBS:
            w = [(dt, legs) for dt, legs in rows if in_win(dt, lo, hi)]
            ns.append(len(w))
            subs.append(float(book_rets(w, COST).sum() * 100) if len(w) >= 8
                        else -99.0)
        worst = min(subs)
        freqs = {}
        for tag, (lo, hi) in (("ovr", (str(cal[0]), "2099-01-01")),
                              ("gate", GATE), ("fin", FINAL)):
            den = sum(1 for dt in cal if in_win(dt, lo, hi))
            num = sum(1 for dt in rd if in_win(dt, lo, hi))
            freqs[tag] = num / den if den else float("nan")
        name = f"{'+'.join(ms)} nr{nr}"
        stats[(ms, nr)] = dict(rows=rows, subs=subs, ns=ns, worst=worst,
                               freqs=freqs, name=name)
        flag = "  <-- ARENA PASS" if worst > 0 else ""
        print(f"  {name:<22} {worst:>+6.2f}%  {str([round(s, 1) for s in subs]):<42} "
              f"{ns}  {freqs['ovr']:.2f}/{freqs['gate']:.2f}/{freqs['fin']:.2f}"
              f"{flag}")
        if worst > 0:
            arena_pass.append((ms, nr))

    # ----------------------------------------------------------------- gate
    print(f"\n=== GATE {GATE[0]}..{GATE[1]} ({len(arena_pass)} arena survivors, "
          f"ONE look) ===")
    gate_pass = []
    for ms, nr in arena_pass:
        st = stats[(ms, nr)]
        w = [(dt, legs) for dt, legs in st["rows"] if in_win(dt, *GATE)]
        r = book_rets(w, COST)
        tot = r.sum() * 100 if len(r) >= 10 else -99.0
        t = r.mean() / r.std() * np.sqrt(len(r)) if len(r) > 5 and r.std() > 0 else 0
        st["gate"] = dict(n=len(r), tot=tot, t=t,
                          win=float((r > 0).mean()) if len(r) else 0.0,
                          t0=float(book_rets(w, 0.0).sum() * 100),
                          t5=float(book_rets(w, 5 * BPS).sum() * 100))
        print(f"  {st['name']:<22} n={len(r):>3} win={st['gate']['win']:.0%} "
              f"tot={tot:+.2f}% t={t:+.2f} | 0bps {st['gate']['t0']:+.2f}% / "
              f"5bps {st['gate']['t5']:+.2f}%"
              f"{'  <-- GATE PASS' if tot > 0 else ''}")
        if tot > 0:
            gate_pass.append((ms, nr))

    # ---------------------------------------------------------------- final
    print(f"\n=== FINAL one-shot {FINAL[0]}..now ({len(gate_pass)} gate survivors, "
          f"ONE look) ===")
    for ms, nr in gate_pass:
        st = stats[(ms, nr)]
        w = [(dt, legs) for dt, legs in st["rows"] if in_win(dt, *FINAL)]
        r = book_rets(w, COST)
        n = len(r)
        t = r.mean() / r.std() * np.sqrt(n) if n > 5 and r.std() > 0 else 0
        ser = pd.Series(r, index=pd.to_datetime([dt for dt, _ in w]))
        eq = (1 + ser).cumprod()
        maxdd = float((eq / eq.cummax() - 1).min()) if n else float("nan")
        monthly = ser.resample("ME").sum()
        sharpe = (monthly.mean() / monthly.std() * np.sqrt(12)
                  if len(monthly) > 3 and monthly.std() > 0 else float("nan"))
        contrib = {tk: 0.0 for tk in ms}
        legs_n = {tk: 0 for tk in ms}
        for dt, legs in w:
            for tk, raw in legs.items():
                contrib[tk] += (raw - COST) / len(legs)
                legs_n[tk] += 1
        st["final"] = dict(
            n=n, t=t, win=float((r > 0).mean()) if n else 0.0,
            tot=float(r.sum() * 100),
            t0=float(book_rets(w, 0.0).sum() * 100),
            t5=float(book_rets(w, 5 * BPS).sum() * 100),
            sharpe=float(sharpe), maxdd=maxdd,
            contrib={tk: float(c * 100) for tk, c in contrib.items()},
            legs_n=legs_n)
        f = st["final"]
        cs = " ".join(f"{tk}{f['contrib'][tk]:+.2f}%({legs_n[tk]})" for tk in ms)
        print(f"  {st['name']:<22} n={n:>3} win={f['win']:.0%} tot={f['tot']:+.2f}% "
              f"t={t:+.2f} | 0bps {f['t0']:+.2f}% / 2bps {f['tot']:+.2f}% / "
              f"5bps {f['t5']:+.2f}% | mSharpe={sharpe:.2f} maxDD={maxdd:.2%} | "
              f"legs: {cs} | freq_fin={st['freqs']['fin']:.2f}")

    # ------------------------------------------------------------- champion
    print("\n=== CHAMPION (full-ladder pass + final freq >= 0.48, best final total, "
          "tie-break arena worst) ===")
    full = [(ms, nr) for ms, nr in gate_pass if stats[(ms, nr)]["final"]["tot"] > 0]
    elig = [(ms, nr) for ms, nr in full if stats[(ms, nr)]["freqs"]["fin"] >= 0.48]
    if elig:
        best = max(elig, key=lambda k: (stats[k]["final"]["tot"], stats[k]["worst"]))
        st = stats[best]
        f = st["final"]
        print(f"  CHAMPION: {st['name']}  final={f['tot']:+.2f}% (n={f['n']}, "
              f"t={f['t']:+.2f}, win={f['win']:.0%})  freq_fin={st['freqs']['fin']:.2f}  "
              f"arena_worst={st['worst']:+.2f}%  0/5bps {f['t0']:+.2f}%/{f['t5']:+.2f}%  "
              f"mSharpe={f['sharpe']:.2f} maxDD={f['maxdd']:.2%}")
    else:
        print("  NONE passes (full ladder + freq >= 0.48). Near-misses:")
        # best full-ladder passer regardless of freq
        if full:
            best = max(full, key=lambda k: (stats[k]["final"]["tot"],
                                            stats[k]["worst"]))
            st = stats[best]
            print(f"    best full-ladder passer: {st['name']} "
                  f"final={st['final']['tot']:+.2f}% but freq_fin="
                  f"{st['freqs']['fin']:.2f} < 0.48")
        # best near-miss = closest to passing the arena (no gate/final peeking)
        nm = max(configs, key=lambda k: stats[k]["worst"])
        st = stats[nm]
        bad = [f"{SUBS[i][0][:7]}:{st['subs'][i]:+.1f}%"
               for i in range(5) if st["subs"][i] <= 0]
        print(f"    best near-miss (arena): {st['name']} worst={st['worst']:+.2f}% "
              f"freq_fin={st['freqs']['fin']:.2f} | negative halves: {bad} "
              f"(gate/final NOT examined — arena selection only)")
        # highest-freq config and where it died
        hf = max(configs, key=lambda k: stats[k]["freqs"]["fin"])
        st = stats[hf]
        died = ("arena" if st["worst"] <= 0 else
                "gate" if (hf not in gate_pass) else "final total <= 0")
        print(f"    highest-freq config: {st['name']} freq_fin="
              f"{st['freqs']['fin']:.2f}, failed at {died} "
              f"(arena worst {st['worst']:+.2f}%)")


if __name__ == "__main__":
    main()
