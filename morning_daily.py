"""
morning_daily.py — DAILY simulation + improvement engine for the morning strategies.

Tracked strategies (all forward-simulated daily on data they have never seen):
  vM  stock  on QQQ and SPY — two-sided 25-min opening-range breakout, target 2 x risk,
             stop = far side of the opening range, NR<=0.3 compression filter,
             entries 9:55-11:30 ET, hard exit 12:00, 2 bps. (QQQ ladder-validated
             2026-07-21; SPY confirmed as the one transfer ticker.)
  vMO options on QQQ and SPY — the SAME signals bought as 0DTE ATM options (call on
             long, put on short), REAL Polygon option 5-min bar fills, 1%/side on
             premium, sold at the stock leg's exit time. Fixed premium per trade.

What one daily run does (scheduled weekdays after the close):
  1. REFRESH  — pull recent QQQ+SPY 5-min bars from Polygon into data_cache.
  2. FORWARD  — simulate all four tracks on every new session since the freeze;
                options fills use the day's real option bars (fetched EOD).
  3. SEARCH   — evaluate ~20 new QQQ configs from the morning ORB space on the
                ladder; a config that beats the champion's ladder becomes CHALLENGER.
  4. PROMOTE  — only on FORWARD evidence (>= 20 forward trades AND higher forward
                total than the champion over the same span).
  5. REPORT   — runs/morning_reports/YYYY-MM-DD.txt (+ git commit/push).

Paper research only — this never places orders anywhere.
Usage:  python morning_daily.py [--no-fetch] [--no-push]
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from morning_qqq import COST, SUBS, GATE, FINAL, days, window, run
from morning_qqq3 import add_nr
from morning_qqq4 import orb2s
from morning_validate import load_ticker

STATE = Path("runs/morning_state.json")
REPORTS = Path("runs/morning_reports")
FREEZE = "2026-07-21"                      # forward track starts here
CHAMP = dict(or_bars=5, rr=2.0, nr=0.3, last_entry=690, sides="both", volx=None)
TICKERS = ["QQQ", "SPY"]                   # vM + vMO run on both
VMO = ("0dte", "atm")                      # options overlay: bucket, strike mode

SPACE = {"or_bars": [2, 3, 4, 5, 6, 8],
         "rr": [1.5, 2.0, 3.0, None],
         "nr": [0.2, 0.3, 0.4, 0.5, None],
         "last_entry": [630, 660, 690, 715],
         "sides": ["both", "short"],
         "volx": [None, 1.2]}
N_SAMPLES = 20
PROMOTE_MIN_TRADES = 20


def ckey(cfg):
    return json.dumps(cfg, sort_keys=True)


def load_state():
    if STATE.exists():
        st = json.loads(STATE.read_text())
        # migrate v1 (QQQ-only) forward keys -> "QQQ|<cfg>"
        if st.get("layout") != 2:
            st["forward"] = {f"QQQ|{k}": v for k, v in st.get("forward", {}).items()}
            st["layout"] = 2
        return st
    return {"layout": 2, "champion": CHAMP, "champion_since": FREEZE,
            "challenger": None, "challenger_since": None,
            "forward": {},              # "<TK>|<cfg-or-vMO>" -> {date: [rets]}
            "tried": {}, "events": []}


def save_state(st):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def refresh():
    """Pull recent bars for all tracked tickers; supersede older recent files."""
    from basket import ticker_cfg
    from data import fetch_polygon
    end = (date.today() + timedelta(days=1)).isoformat()
    for tk in TICKERS:
        out = Path(f"data_cache/{tk}_recent_2026-06-01_{end}.csv")
        cfg = ticker_cfg(tk)
        cfg.data.start_date, cfg.data.end_date = "2026-06-01", end
        cfg.data.multiplier, cfg.data.timespan = 5, "minute"
        df = fetch_polygon(cfg)
        df.to_csv(out, index=False)
        print(f"  refreshed {tk}: {len(df)} rows -> {out.name}")
        for p in Path("data_cache").glob(f"{tk}_recent_2026-06-01_*.csv"):
            if p != out:
                p.unlink()


def day_data():
    """{ticker: day-records with nr_rank} for all tracked tickers."""
    out = {}
    for tk in TICKERS:
        dd = days(load_ticker(tk))
        add_nr(dd)
        out[tk] = dd
    return out


def ladder(dd, cfg):
    fn = lambda d: orb2s(d, **cfg)
    subs = []
    for lo, hi in SUBS:
        r = run(window(dd, lo, hi), fn)
        subs.append(float(r.sum() * 100) if len(r) >= 8 else -99.0)
    g = run(window(dd, *GATE), fn)
    f = run(window(dd, *FINAL), fn)
    return {"arena_worst": min(subs),
            "gate": float(g.sum() * 100) if len(g) >= 10 else -99.0,
            "final": float(f.sum() * 100) if len(f) >= 10 else -99.0,
            "final_n": int(len(f))}


def forward_sim(dd, st, key, day_fn, since):
    """Append per-day results (list of rets, [] = no trade) for new sessions."""
    fwd = st["forward"].setdefault(key, {})
    lo = max(date.fromisoformat(since), date.fromisoformat(FREEZE))
    new = []
    for d in dd:
        ds = d["date"].isoformat() if hasattr(d["date"], "isoformat") else str(d["date"])
        if date.fromisoformat(ds) < lo or ds in fwd:
            continue
        r = day_fn(d)
        fwd[ds] = [] if r is None else [round(float(r), 6)]
        if fwd[ds]:
            new.append((ds, r))
    return new


def opt_day_fn(tk):
    """Day -> vMO option return (real bars; None if no signal; NaN-skip if data
    missing so the day can retry tomorrow)."""
    from morning_opt import orb2s_trade, replay

    def fn(d):
        t = orb2s_trade(d, CHAMP["or_bars"], CHAMP["rr"], nr=CHAMP["nr"],
                        last_entry=CHAMP["last_entry"], sides=CHAMP["sides"])
        if t is None:
            return None
        res = replay([t], VMO[0], VMO[1], underlying=tk)[0]
        if res["status"] != "filled":
            raise LookupError(f"option leg {res['status']} for {tk} {d['date']}")
        return res["opt_ret"]
    return fn


def opt_forward(dd, st, tk, since):
    """Options forward track; unfilled days are NOT recorded (retried next run,
    logged) so missing option data never silently becomes a zero."""
    key = f"{tk}|vMO|{VMO[0]}-{VMO[1]}"
    fwd = st["forward"].setdefault(key, {})
    lo = max(date.fromisoformat(since), date.fromisoformat(FREEZE))
    fn = opt_day_fn(tk)
    new, misses = [], []
    for d in dd:
        ds = d["date"].isoformat() if hasattr(d["date"], "isoformat") else str(d["date"])
        if date.fromisoformat(ds) < lo or ds in fwd:
            continue
        try:
            r = fn(d)
        except LookupError as e:
            misses.append(str(e))
            continue
        fwd[ds] = [] if r is None else [round(float(r), 6)]
        if fwd[ds]:
            new.append((ds, r))
    return new, misses


def fwd_stats(st, key, since):
    rets = [r for ds, rs in st["forward"].get(key, {}).items()
            if date.fromisoformat(ds) >= date.fromisoformat(since) for r in rs]
    a = np.array(rets)
    return {"n": len(a), "total": float(a.sum() * 100) if len(a) else 0.0,
            "win": float((a > 0).mean()) if len(a) else 0.0}


def sample_configs(st, rng):
    out, guard = [], 0
    while len(out) < N_SAMPLES and guard < 4000:
        guard += 1
        cfg = {k: rng.choice(v) for k, v in SPACE.items()}
        k = ckey(cfg)
        if k in st["tried"] or k == ckey(st["champion"]) or any(ckey(c) == k for c in out):
            continue
        out.append(cfg)
    return out


def main():
    no_fetch = "--no-fetch" in sys.argv
    no_push = "--no-push" in sys.argv
    today = date.today().isoformat()
    lines = [f"MORNING daily run  {today}  (vM stock + vMO 0DTE-ATM options, QQQ+SPY)",
             "=" * 66]

    if not no_fetch:
        try:
            refresh()
            lines.append("data refresh: OK")
        except Exception as e:
            lines.append(f"data refresh FAILED ({e}) — using cached data")

    dds = day_data()
    st = load_state()
    champ_key = ckey(st["champion"])

    # ---- champion ladder health on QQQ (does the edge persist?) ----
    lad = ladder(dds["QQQ"], st["champion"])
    lines.append(f"\nCHAMPION {champ_key}")
    lines.append(f"  QQQ ladder: arena_worst={lad['arena_worst']:+.2f}%  "
                 f"gate={lad['gate']:+.2f}%  final={lad['final']:+.2f}% "
                 f"(n={lad['final_n']})")

    # ---- stock forward tracks (champion on each ticker) ----
    for tk in TICKERS:
        key = f"{tk}|{champ_key}"
        new = forward_sim(dds[tk], st, key,
                          lambda d: orb2s(d, **st["champion"]), st["champion_since"])
        fs = fwd_stats(st, key, st["champion_since"])
        lines.append(f"  {tk} stock forward since {st['champion_since']}: n={fs['n']} "
                     f"total={fs['total']:+.2f}% win={fs['win']:.0%}")
        for ds, r in new:
            lines.append(f"    new fill {ds}: {r * 1e4:+.1f}bp")

    # ---- options forward tracks (vMO on each ticker, real bars) ----
    lines.append(f"\nvMO OPTIONS ({VMO[0]} {VMO[1]}, 1%/side, fixed premium/trade)")
    for tk in TICKERS:
        try:
            new, misses = opt_forward(dds[tk], st, tk, FREEZE)
            fs = fwd_stats(st, f"{tk}|vMO|{VMO[0]}-{VMO[1]}", FREEZE)
            lines.append(f"  {tk} vMO forward: n={fs['n']} "
                         f"total={fs['total']:+.1f}% of premium win={fs['win']:.0%}")
            for ds, r in new:
                lines.append(f"    new option fill {ds}: {r * 100:+.1f}% of premium")
            for msg in misses:
                lines.append(f"    RETRY LATER: {msg}")
        except Exception as e:
            lines.append(f"  {tk} vMO forward FAILED ({e}) — will retry next run")

    # ---- challenger (stock, QQQ) ----
    if st["challenger"]:
        ch_key = f"QQQ|{ckey(st['challenger'])}"
        forward_sim(dds["QQQ"], st, ch_key,
                    lambda d: orb2s(d, **st["challenger"]), st["challenger_since"])
        cs = fwd_stats(st, ch_key, st["challenger_since"])
        cmp_rets = [r for ds, rs in st["forward"].get(f"QQQ|{champ_key}", {}).items()
                    if date.fromisoformat(ds) >= date.fromisoformat(st["challenger_since"])
                    for r in rs]
        cmp_tot = float(np.sum(cmp_rets) * 100) if cmp_rets else 0.0
        lines.append(f"\nCHALLENGER {ckey(st['challenger'])}")
        lines.append(f"  forward since {st['challenger_since']}: n={cs['n']} "
                     f"total={cs['total']:+.2f}% vs champion {cmp_tot:+.2f}% same span")
        if cs["n"] >= PROMOTE_MIN_TRADES and cs["total"] > cmp_tot:
            st["events"].append({"date": today, "event": "PROMOTION",
                                 "old": st["champion"], "new": st["challenger"]})
            lines.append("  *** PROMOTED to champion on forward evidence ***")
            st["champion"], st["champion_since"] = st["challenger"], today
            st["challenger"], st["challenger_since"] = None, None

    # ---- daily improvement search (QQQ stock configs) ----
    rng = random.Random(today)
    cands = sample_configs(st, rng)
    best = None
    for cfg in cands:
        res = ladder(dds["QQQ"], cfg)
        st["tried"][ckey(cfg)] = {"date": today, **res}
        beats = (res["arena_worst"] > max(lad["arena_worst"], 0) and
                 res["gate"] > 0 and res["final"] > lad["final"])
        if beats and (best is None or res["final"] > best[1]["final"]):
            best = (cfg, res)
    lines.append(f"\nsearch: {len(cands)} new configs evaluated "
                 f"({len(st['tried'])} lifetime)")
    if best:
        cfg, res = best
        cur = st["challenger"]
        cur_final = st["tried"].get(ckey(cur), {}).get("final", -99) if cur else -99
        if cur is None or res["final"] > cur_final:
            st["challenger"], st["challenger_since"] = cfg, today
            st["events"].append({"date": today, "event": "NEW CHALLENGER",
                                 "cfg": cfg, **res})
            lines.append(f"  NEW CHALLENGER: {ckey(cfg)}")
            lines.append(f"    arena_worst={res['arena_worst']:+.2f}% "
                         f"gate={res['gate']:+.2f}% final={res['final']:+.2f}%")
    else:
        lines.append("  no config beat the champion's ladder today")

    save_state(st)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rep = REPORTS / f"{today}.txt"
    rep.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if not no_push:
        try:
            subprocess.run(["git", "add", "runs/morning_state.json",
                            str(rep), "morning_daily.py"], cwd=".", check=True)
            subprocess.run(["git", "commit", "-m",
                            f"morning daily run {today}"], cwd=".", check=True)
            subprocess.run(["git", "push"], cwd=".", check=True)
            print("pushed to GitHub")
        except Exception as e:
            print(f"git push skipped ({e})")


if __name__ == "__main__":
    main()
