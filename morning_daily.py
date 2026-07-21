"""
morning_daily.py — DAILY simulation + improvement engine for the QQQ morning strategy.

Champion "vM" (frozen 2026-07-21, passed the full honesty ladder + robustness map):
  QQQ two-sided 25-min opening-range breakout, target 2 x risk, stop = far side of the
  opening range, NR filter (prior-day range in bottom 30% of trailing 7 days),
  entries 9:55-11:30 ET, hard exit 12:00, 2 bps cost assumption.

What one daily run does (scheduled weekdays after the close):
  1. REFRESH  — pull recent QQQ 5-min bars from Polygon into data_cache.
  2. FORWARD  — simulate champion (and challenger, if any) on every new session since
                the freeze; append fills to runs/morning_state.json. This is the honest
                forward track — data the config has never seen.
  3. SEARCH   — evaluate ~20 new configs from the morning ORB space on the ladder
                (arena = worst of 5 half-years 22-24, gate = 24-25, final = 25-now).
                A config that beats the champion's ladder everywhere becomes the
                CHALLENGER (tracked forward from that day).
  4. PROMOTE  — only on FORWARD evidence: challenger needs >= 20 forward trades AND a
                higher forward total than the champion over the same span. Backtest
                numbers alone can never dethrone the champion.
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

from morning_qqq import COST, SUBS, GATE, FINAL, load, days, window, run
from morning_qqq3 import add_nr
from morning_qqq4 import orb2s

STATE = Path("runs/morning_state.json")
REPORTS = Path("runs/morning_reports")
FREEZE = "2026-07-21"                      # forward track starts here
CHAMP = dict(or_bars=5, rr=2.0, nr=0.3, last_entry=690, sides="both", volx=None)

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
        return json.loads(STATE.read_text())
    return {"champion": CHAMP, "champion_since": FREEZE,
            "challenger": None, "challenger_since": None,
            "forward": {},              # ckey -> {date: [trade rets]}
            "tried": {}, "events": []}


def save_state(st):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(st, indent=1))


def refresh():
    """Pull recent QQQ bars; supersede older *_recent_2026-06-01_* files."""
    from basket import ticker_cfg
    from data import fetch_polygon
    today = date.today()
    end = (today + timedelta(days=1)).isoformat()
    out = Path(f"data_cache/QQQ_recent_2026-06-01_{end}.csv")
    cfg = ticker_cfg("QQQ")
    cfg.data.start_date, cfg.data.end_date = "2026-06-01", end
    cfg.data.multiplier, cfg.data.timespan = 5, "minute"
    df = fetch_polygon(cfg)
    df.to_csv(out, index=False)
    print(f"  refreshed QQQ: {len(df)} rows -> {out.name}")
    for p in Path("data_cache").glob("QQQ_recent_2026-06-01_*.csv"):
        if p != out:
            p.unlink()


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


def forward_sim(dd, st, cfg, since):
    """Append per-day results for sessions >= since not yet recorded."""
    key = ckey(cfg)
    fwd = st["forward"].setdefault(key, {})
    lo = max(date.fromisoformat(since), date.fromisoformat(FREEZE))
    new = []
    for d in dd:
        ds = d["date"].isoformat() if hasattr(d["date"], "isoformat") else str(d["date"])
        if date.fromisoformat(ds) < lo or ds in fwd:
            continue
        r = orb2s(d, **cfg)
        fwd[ds] = [] if r is None else [round(float(r), 6)]
        if fwd[ds]:
            new.append((ds, r))
    return new


def fwd_stats(st, cfg, since):
    key = ckey(cfg)
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
    lines = [f"MORNING-QQQ daily run  {today}", "=" * 46]

    if not no_fetch:
        try:
            refresh()
            lines.append("data refresh: OK")
        except Exception as e:
            lines.append(f"data refresh FAILED ({e}) — using cached data")

    df = load()
    dd = days(df)
    add_nr(dd)
    st = load_state()

    # ---- champion ladder health (does the edge persist as 'final' grows?) ----
    lad = ladder(dd, st["champion"])
    lines.append(f"\nCHAMPION {ckey(st['champion'])}")
    lines.append(f"  ladder: arena_worst={lad['arena_worst']:+.2f}%  "
                 f"gate={lad['gate']:+.2f}%  final={lad['final']:+.2f}% "
                 f"(n={lad['final_n']})")

    # ---- forward tracks ----
    new_c = forward_sim(dd, st, st["champion"], st["champion_since"])
    fs = fwd_stats(st, st["champion"], st["champion_since"])
    lines.append(f"  forward since {st['champion_since']}: n={fs['n']} "
                 f"total={fs['total']:+.2f}% win={fs['win']:.0%}")
    for ds, r in new_c:
        lines.append(f"    new fill {ds}: {r * 1e4:+.1f}bp")

    if st["challenger"]:
        forward_sim(dd, st, st["challenger"], st["challenger_since"])
        cs = fwd_stats(st, st["challenger"], st["challenger_since"])
        ch_since = st["challenger_since"]
        cmp_fs = {"n": 0, "total": 0.0}
        key = ckey(st["champion"])
        cmp_rets = [r for ds, rs in st["forward"].get(key, {}).items()
                    if date.fromisoformat(ds) >= date.fromisoformat(ch_since)
                    for r in rs]
        cmp_tot = float(np.sum(cmp_rets) * 100) if cmp_rets else 0.0
        lines.append(f"CHALLENGER {ckey(st['challenger'])}")
        lines.append(f"  forward since {ch_since}: n={cs['n']} total={cs['total']:+.2f}% "
                     f"vs champion {cmp_tot:+.2f}% over same span")
        if cs["n"] >= PROMOTE_MIN_TRADES and cs["total"] > cmp_tot:
            st["events"].append({"date": today, "event": "PROMOTION",
                                 "old": st["champion"], "new": st["challenger"]})
            lines.append("  *** PROMOTED to champion on forward evidence ***")
            st["champion"], st["champion_since"] = st["challenger"], today
            st["challenger"], st["challenger_since"] = None, None

    # ---- daily improvement search ----
    rng = random.Random(today)             # reproducible per-day sample
    cands = sample_configs(st, rng)
    best = None
    for cfg in cands:
        res = ladder(dd, cfg)
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
                            f"morning-qqq daily run {today}"], cwd=".", check=True)
            subprocess.run(["git", "push"], cwd=".", check=True)
            print("pushed to GitHub")
        except Exception as e:
            print(f"git push skipped ({e})")


if __name__ == "__main__":
    main()
