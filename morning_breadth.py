"""
morning_breadth.py — INDEX BREADTH TEST (pre-registered, minimal multiplicity).

Thesis: vM's compression->morning-breakout edge is a DIVERSIFIED-INDEX phenomenon
(QQQ, SPY work; single stocks and GLD/JPM/XLE fail). Test EXACTLY three more index
products chosen for the thesis, with vM params UNCHANGED:
    or_bars=5, rr=2.0, nr=0.3, last_entry=690, sides='both'
Tickers: DIA (Dow 30), XLK (tech sector), MDY (S&P 400 midcap).
No per-ticker tuning of any kind.

Pre-registered PASS bar (same as SPY's transfer): gate>0 AND final>0 with final t>0.5.

Book stats: for QQQ+SPY (baseline) plus each passer and all passers combined:
  * union trade-day fraction (a day counts if ANY member trades)
    overall / gate-year / final-year
  * equal-weight portfolio (daily average of ACTIVE members' returns):
    total, monthly Sharpe, maxDD.

Honesty: this is ONE pre-registered look per ticker at gate and final; params frozen;
selection happened long ago on QQQ's arena. 2bps headline, 5bps sensitivity shown.
Research only — no orders.
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

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

NEW = ["DIA", "XLK", "MDY"]
BOOK_BASE = ["QQQ", "SPY"]
START, END = "2021-06-01", "2026-06-01"
CHAMP = dict(or_bars=5, rr=2.0, nr=0.3, last_entry=690, sides="both")


def champ(day):
    return orb2s(day, **CHAMP)


# ------------------------------------------------------------------- data fetch
def fetch_all():
    from basket import ticker_cfg
    from data import fetch_polygon
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    for tk in NEW:
        jobs = [
            (Path(f"data_cache/{tk}_5minute_{START}_{END}.csv"), START, END),
            (Path(f"data_cache/{tk}_recent_2026-06-01_{tomorrow}.csv"),
             "2026-06-01", tomorrow),
        ]
        for out, s, e in jobs:
            if out.exists():
                print(f"  {tk}: already cached ({out.name})", flush=True)
                continue
            cfg = ticker_cfg(tk)
            cfg.data.start_date, cfg.data.end_date = s, e
            cfg.data.multiplier, cfg.data.timespan = 5, "minute"
            for attempt in range(4):
                try:
                    df = fetch_polygon(cfg)
                    df.to_csv(out, index=False)
                    print(f"  {tk}: saved {len(df)} rows -> {out.name}  "
                          f"[{df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}]",
                          flush=True)
                    break
                except Exception as exc:
                    print(f"  {tk}: attempt {attempt+1} failed ({exc}); "
                          f"retrying in 10s", flush=True)
                    time.sleep(10)
            else:
                raise SystemExit(f"{tk}: fetch GAVE UP after retries")


# --------------------------------------------------------------------- ladder
def in_win(dt, win):
    lo, hi = pd.Timestamp(win[0]).date(), pd.Timestamp(win[1]).date()
    return lo <= dt < hi


def ladder(tk, dd):
    """Exact vM ladder on one ticker. Returns a stats dict (one gate/final look)."""
    trades = [(d["date"], champ(d)) for d in dd]
    trades = [(dt, r) for dt, r in trades if r is not None]
    rets = pd.Series([r for _, r in trades],
                     index=pd.to_datetime([dt for dt, _ in trades]), dtype=float)

    subs = []
    for lo, hi in SUBS:
        r = np.array([r for dt, r in trades if in_win(dt, (lo, hi))])
        subs.append(float(r.sum() * 100) if len(r) >= 8 else float("nan"))
    g = np.array([r for dt, r in trades if in_win(dt, GATE)])
    f = np.array([r for dt, r in trades if in_win(dt, FINAL)])
    gt = float(g.sum() * 100) if len(g) else float("nan")
    ft = float(f.sum() * 100) if len(f) else float("nan")
    t = (float(f.mean() / f.std() * np.sqrt(len(f)))
         if len(f) > 5 and f.std() > 0 else float("nan"))

    monthly = rets.resample("ME").sum()
    sharpe = (float(monthly.mean() / monthly.std() * np.sqrt(12))
              if len(monthly) > 3 and monthly.std() > 0 else float("nan"))
    eq = (1 + rets).cumprod()
    maxdd = float((eq / eq.cummax() - 1).min()) if len(eq) else float("nan")

    verdict = ("PASS" if (not np.isnan(gt) and gt > 0
                          and not np.isnan(ft) and ft > 0
                          and not np.isnan(t) and t > 0.5) else "FAIL")

    f5 = f - 3e-4          # 2bps headline already applied; +3bps -> 5bps total
    print(f"  {tk:<5} arena_worst={np.nanmin(subs):+6.2f}%  "
          f"subs={[None if np.isnan(s) else round(s, 1) for s in subs]}")
    print(f"        gate={gt:+6.2f}% (n={len(g)}, win={(g > 0).mean() if len(g) else 0:.0%})  "
          f"final={ft:+6.2f}% (n={len(f)}, win={(f > 0).mean() if len(f) else 0:.0%}, "
          f"t={t:+.2f})  final@5bps={f5.sum()*100:+.2f}%")
    print(f"        full-history: trades={len(rets)}  win={float((rets > 0).mean()):.1%}  "
          f"total={rets.sum()*100:+.2f}%  monthlySharpe={sharpe:.2f}  maxDD={maxdd:.2%}")
    print(f"        VERDICT (pre-registered: gate>0 AND final>0 AND t>0.5): {verdict}")
    return dict(ticker=tk, n_trades=int(len(rets)),
                win=round(float((rets > 0).mean()), 3),
                total_pct=round(float(rets.sum() * 100), 2),
                arena_worst_pct=round(float(np.nanmin(subs)), 2),
                gate_pct=round(gt, 2), gate_n=int(len(g)),
                final_pct=round(ft, 2), final_n=int(len(f)),
                final_t=round(t, 2) if not np.isnan(t) else None,
                final_5bps_pct=round(float(f5.sum() * 100), 2),
                monthly_sharpe=round(sharpe, 2) if not np.isnan(sharpe) else None,
                max_dd_pct=round(maxdd * 100, 2), verdict=verdict)


# ----------------------------------------------------------------------- book
def book(name, members, all_days, trades):
    """Union trade-day fraction + equal-weight (active members) portfolio stats."""
    dates = sorted(set().union(*[all_days[tk] for tk in members]))
    port, active = [], []
    for dt in dates:
        rs = [trades[tk][dt] for tk in members if dt in trades[tk]]
        active.append(bool(rs))
        port.append(float(np.mean(rs)) if rs else 0.0)
    ser = pd.Series(port, index=pd.to_datetime(dates))
    act = pd.Series(active, index=pd.to_datetime(dates))

    def frac(win):
        m = [in_win(dt, win) for dt in dates]
        sub = act[m]
        return float(sub.mean()) if len(sub) else float("nan")

    overall = float(act.mean())
    f_gate, f_final = frac(GATE), frac(FINAL)
    total = float(ser.sum() * 100)
    gate_tot = float(ser[[in_win(dt, GATE) for dt in dates]].sum() * 100)
    final_tot = float(ser[[in_win(dt, FINAL) for dt in dates]].sum() * 100)
    monthly = ser.resample("ME").sum()
    sharpe = (float(monthly.mean() / monthly.std() * np.sqrt(12))
              if monthly.std() > 0 else float("nan"))
    eq = (1 + ser).cumprod()
    maxdd = float((eq / eq.cummax() - 1).min())

    print(f"  {name:<22} days={len(dates)}  trade-day frac: "
          f"overall={overall:.1%}  gate={f_gate:.1%}  final={f_final:.1%}")
    print(f"  {'':<22} EW portfolio: total={total:+.2f}%  gate={gate_tot:+.2f}%  "
          f"final={final_tot:+.2f}%  monthlySharpe={sharpe:.2f}  maxDD={maxdd:.2%}")
    return dict(members=members, n_days=len(dates),
                trade_day_frac_overall=round(overall, 3),
                trade_day_frac_gate=round(f_gate, 3),
                trade_day_frac_final=round(f_final, 3),
                ew_total_pct=round(total, 2),
                ew_gate_pct=round(gate_tot, 2), ew_final_pct=round(final_tot, 2),
                monthly_sharpe=round(sharpe, 2), max_dd_pct=round(maxdd * 100, 2))


# ----------------------------------------------------------------------- main
def main():
    print("=== 0) FETCH (skip if cached) ===")
    fetch_all()

    print(f"\n=== 1) EXACT vM LADDER per ticker "
          f"(params frozen: {CHAMP}, cost {COST*1e4:.0f}bps) ===")
    stats, all_days, trades = {}, {}, {}
    for tk in BOOK_BASE + NEW:
        dd = days(load_ticker(tk))
        add_nr(dd)
        all_days[tk] = {d["date"] for d in dd}
        tr = {}
        for d in dd:
            r = champ(d)
            if r is not None:
                tr[d["date"]] = r
        trades[tk] = tr
        label = "(baseline, already validated)" if tk in BOOK_BASE else "(NEW, one look)"
        print(f"\n  --- {tk} {label} ---")
        stats[tk] = ladder(tk, dd)

    passers = [tk for tk in NEW if stats[tk]["verdict"] == "PASS"]
    print(f"\n  NEW-TICKER PASSERS: {passers if passers else 'none'}")

    print(f"\n=== 2) BOOKS (union trade-day fraction + equal-weight portfolio) ===")
    books = {}
    books["QQQ+SPY (baseline)"] = book("QQQ+SPY (baseline)", BOOK_BASE,
                                       all_days, trades)
    for tk in passers:
        nm = f"QQQ+SPY+{tk}"
        books[nm] = book(nm, BOOK_BASE + [tk], all_days, trades)
    if len(passers) > 1:
        nm = "QQQ+SPY+" + "+".join(passers)
        books[nm] = book(nm, BOOK_BASE + passers, all_days, trades)
    return stats, books, passers


if __name__ == "__main__":
    main()
