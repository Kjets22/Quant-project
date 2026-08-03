"""
audit_v6_opt_812w.py — is v6's "8-12w ATM" options cell a real edge or an artifact?

READ-ONLY audit of v6_options_real.py. Runs entirely off the existing caches:
Polygon is hard-disabled (odp._get raises), contract listings and 5-min option
bars are read from data_cache/options only, and NOTHING is written anywhere.

Replicates v6_options_real.replay() bar-for-bar, but instruments every trade so
we can see which signals filled, which were skipped and WHY, what DTE actually
got traded, and how the P&L is distributed.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

import options_data_polygon as odp

# ---- hard-disable the network so this audit can only read what already exists
def _no_net(*a, **k):
    raise RuntimeError("network disabled in audit (cache-only)")


odp._get = _no_net
odp._session_key = _no_net

import v6_options_real as V6
from v6_options_real import gen_trades, BUCKETS, STRIKES, HAIRCUT, TRAIN_END, SIM_END
from triple_barrier_breadth import TICKERS

CACHE = Path("data_cache/options")
AGGS = CACHE / "aggs5"

_CONS_MEM: dict = {}
_BARS_MEM: dict = {}
CONS_MISS = Counter()
BARS_MISS = Counter()


def cons_cached(ul, day, klo, khi):
    """Exactly v6's cache key; returns [] (and counts a miss) if not on disk."""
    key = f"{ul}_{day}_{int(klo)}_{int(khi)}"
    if key in _CONS_MEM:
        return _CONS_MEM[key]
    cf = CACHE / f"cons_{key}.json"
    if not cf.exists():
        CONS_MISS[ul] += 1
        _CONS_MEM[key] = []
        return []
    rows = json.loads(cf.read_text())
    _CONS_MEM[key] = rows
    return rows


def bars_cached(ticker, d0, d1):
    cf = AGGS / f"{ticker.replace(':', '_')}_{d0}_{d1}.json"
    k = str(cf)
    if k in _BARS_MEM:
        return _BARS_MEM[k]
    if not cf.exists():
        BARS_MISS[ticker.split(":")[-1][:4]] += 1
        _BARS_MEM[k] = []
        return []
    b = json.loads(cf.read_text())
    _BARS_MEM[k] = b
    return b


from qqq_options_real import et_date
from vc_options_real import day_close


def replay_inst(trades, lo, hi, mult, ts_all, c_all):
    """v6_options_real.replay(), instrumented. Returns (fills, skips).

    fills: dicts with the trade, the contract chosen and the option return.
    skips: dicts with a reason code.
    """
    fills, skips = [], []
    for t in trades:
        d0, d1 = et_date(t["t0"]), et_date(t["t1"])
        tgt_k = t["S0"] * mult
        raw = cons_cached(t["tk"], d0, t["S0"] * 0.88, t["S0"] * 1.05)
        dtes = sorted({(dt.date.fromisoformat(x["exp"]) - d0).days for x in raw})
        cons = [x for x in raw
                if lo <= (dt.date.fromisoformat(x["exp"]) - d0).days <= hi]
        if not cons:
            skips.append(dict(t=t, d0=d0, reason="no_contract_in_dte_window",
                              n_listed=len(raw),
                              max_dte=(max(dtes) if dtes else None)))
            continue
        cons.sort(key=lambda x: (abs(x["K"] - tgt_k),
                                 dt.date.fromisoformat(x["exp"])))
        got, why = None, "no_usable_bars"
        for con in cons[:3]:
            exp = dt.date.fromisoformat(con["exp"])
            bars = bars_cached(con["ticker"], str(d0), str(max(d1, exp)))
            if not bars:
                continue
            bt = np.array([b["t"] for b in bars], dtype=np.int64)
            t0ms = int(t["t0"].value // 1_000_000)
            t1ms = int(t["t1"].value // 1_000_000)
            i0 = int(np.searchsorted(bt, t0ms))
            if i0 >= len(bars) or bt[i0] - t0ms > 24 * 3600_000:
                why = "no_print_within_24h"
                continue
            entry = bars[i0]["c"] if bt[i0] == t0ms else bars[i0]["o"]
            if entry <= 0.05:
                why = "entry_le_5c"
                continue
            if d1 > exp:
                Sx = day_close(ts_all, c_all, exp)
                if Sx is None:
                    why = "no_expiry_close"
                    continue
                ex_px = max(Sx - con["K"], 0.0)
                settle = "intrinsic_at_exp"
            else:
                i1 = int(np.searchsorted(bt, t1ms))
                ex_px = (bars[i1]["c"] if i1 < len(bars) and bt[i1] == t1ms
                         else bars[min(i1, len(bars) - 1)]["o"])
                settle = "bar_at_stock_exit"
            got = (con, exp, entry, ex_px, settle)
            break
        if got is None:
            skips.append(dict(t=t, d0=d0, reason=why, n_listed=len(raw),
                              max_dte=(max(dtes) if dtes else None)))
            continue
        con, exp, e, x, settle = got
        r = (x * (1 - HAIRCUT) - e * (1 + HAIRCUT)) / (e * (1 + HAIRCUT))
        fills.append(dict(tk=t["tk"], t0=t["t0"], t1=t["t1"], d0=d0, d1=d1,
                          S0=t["S0"], S1=t["S1"], stock_ret=t["stock_ret"],
                          con=con["ticker"], K=con["K"], exp=exp,
                          dte=(exp - d0).days, entry=e, exit=x, ret=r,
                          settle=settle, moneyness=con["K"] / t["S0"],
                          hold_days=(d1 - d0).days))
    return fills, skips


def desc(r):
    r = np.asarray(r, float)
    if len(r) == 0:
        return "n=0"
    return (f"n={len(r):>3} win {(r > 0).mean():>4.0%} avg {r.mean() * 100:>+7.1f}% "
            f"med {np.median(r) * 100:>+7.1f}% tot {r.sum() * 100:>+8.1f}%")


def tstat(r):
    r = np.asarray(r, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))


def boot_ci(r, n=20000, seed=7):
    r = np.asarray(r, float)
    if len(r) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(r, size=(n, len(r)), replace=True).mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def hr(c="="):
    print(c * 96)


def main():
    print("V6 OPTIONS AUDIT — is '8-12w ATM' a real edge? (cache-only, no network, "
          "no writes)")
    print(f"window {TRAIN_END}..{SIM_END} | v6 buckets {BUCKETS} | strikes {STRIKES} "
          f"| haircut {HAIRCUT:.0%}/side")
    print()

    # ---------------------------------------------------------------- signals
    allt, tsmap = [], {}
    for tk in TICKERS:
        r = gen_trades(tk)
        if not r:
            continue
        trades, ts_all, c_all = r
        tsmap[tk] = (ts_all, c_all)
        allt += trades
        print(f"  {tk}: {len(trades)} v6 stock signals")
        sys.stdout.flush()
    sr = np.array([t["stock_ret"] for t in allt])
    print(f"\nSTOCK leg: {len(allt)} trades, win {(sr > 0).mean():.0%}, "
          f"avg {sr.mean() * 1e4:+.0f}bps, total {sr.sum() * 100:+.1f}%")
    print()

    # -------------------------------------------------- 1. rebuild 9-cell grid
    hr()
    print("1. BASELINE — v6's 9-cell grid rebuilt from cache (should match the run)")
    hr("-")
    print(f"  {'expiry':>7} {'strike':>6} {'n':>4} {'skip':>5} {'win%':>6} "
          f"{'avg/trade':>10} {'median':>8} {'total':>9}")
    grid = {}
    for bname, lo, hi in BUCKETS:
        for sname, mult in STRIKES:
            F, S = [], []
            for tk in tsmap:
                f, s = replay_inst([t for t in allt if t["tk"] == tk],
                                   lo, hi, mult, *tsmap[tk])
                F += f
                S += s
            grid[(bname, sname)] = (F, S)
            if len(F) < 10:
                print(f"  {bname:>7} {sname:>6}  too few fills ({len(F)})")
                continue
            r = np.array([x["ret"] for x in F])
            print(f"  {bname:>7} {sname:>6} {len(r):>4} {len(S):>5} "
                  f"{(r > 0).mean():>6.0%} {r.mean() * 100:>+9.1f}% "
                  f"{np.median(r) * 100:>+7.1f}% {r.sum() * 100:>+8.0f}%")
    if CONS_MISS or BARS_MISS:
        print(f"  [cache misses — contracts {sum(CONS_MISS.values())}, "
              f"bar files {sum(BARS_MISS.values())}]")

    F, S = grid[("8-12w", "ATM")]
    R = np.array([x["ret"] for x in F])

    # ------------------------------------------------ 2. why 422 were skipped
    hr()
    print("2. WHY 422/480 WERE SKIPPED — the DTE window is unreachable by construction")
    hr("-")
    print("  skip reasons in 8-12w/ATM:")
    for k, v in Counter(s["reason"] for s in S).most_common():
        print(f"    {k:<28} {v:>4}")
    mx = [s["max_dte"] for s in S if s["max_dte"] is not None]
    print(f"\n  vc_options_real.contracts_near() queries Polygon with "
          f"expiration_date.lte = signal_day + 50 days,")
    print(f"  but the 8-12w bucket asks for DTE in [50, 90]. So the ONLY expiry that "
          f"can ever match is DTE == 50.")
    print(f"  longest DTE present in any cached contract list: "
          f"{max(mx) if mx else 'n/a'} (over {len(mx)} skipped signals)")
    dte_hist = Counter(x["dte"] for x in F)
    print(f"  DTE actually traded by the 58 '8-12w' fills: {dict(dte_hist)}")
    print(f"  -> the cell labelled '8-12 weeks' is really a single point: "
          f"{list(dte_hist)[0] if len(dte_hist) == 1 else '?'} calendar days "
          f"= 7 weeks + 1 day.")
    print()
    print("  Which signals CAN reach DTE 50? only those whose day+50 is a listed expiry:")
    wd_f = Counter(x["d0"].strftime("%a") for x in F)
    wd_s = Counter(s["d0"].strftime("%a") for s in S)
    print(f"    weekday of FILLED signals : "
          f"{ {k: wd_f.get(k, 0) for k in ['Mon','Tue','Wed','Thu','Fri']} }")
    print(f"    weekday of SKIPPED signals: "
          f"{ {k: wd_s.get(k, 0) for k in ['Mon','Tue','Wed','Thu','Fri']} }")
    print(f"    weekday of the EXPIRY chosen: "
          f"{dict(Counter(x['exp'].strftime('%a') for x in F))}")
    print("    (day+50 is a Friday iff the signal fires on a Thursday — 49 days = 7 weeks)")

    # ---------------------------------------------- 3. concentration of the 58
    hr()
    print("3. CONCENTRATION OF THE 58 FILLS")
    hr("-")
    print("  by ticker:")
    by_tk = defaultdict(list)
    for x in F:
        by_tk[x["tk"]].append(x["ret"])
    for tk, v in sorted(by_tk.items(), key=lambda kv: -np.sum(kv[1])):
        v = np.array(v)
        print(f"    {tk:<5} {desc(v)}  share of total P&L "
              f"{v.sum() / R.sum() * 100:>6.1f}%")
    print("\n  by month:")
    by_m = defaultdict(list)
    for x in F:
        by_m[x["d0"].strftime("%Y-%m")].append(x["ret"])
    for m in sorted(by_m):
        v = np.array(by_m[m])
        print(f"    {m}  {desc(v)}  share {v.sum() / R.sum() * 100:>6.1f}%")
    print("\n  candidate-vs-fill coverage by ticker (fills / candidates):")
    cand = Counter(t["tk"] for t in allt)
    for tk in sorted(cand):
        print(f"    {tk:<5} {len(by_tk.get(tk, [])):>3} / {cand[tk]:>3} "
              f"= {len(by_tk.get(tk, [])) / cand[tk]:>5.0%}")

    # ------------------------------------------------- 4. outlier sensitivity
    hr()
    print("4. IS IT A HANDFUL OF OUTLIERS?")
    hr("-")
    srt = np.sort(R)[::-1]
    print(f"  mean {R.mean() * 100:+.2f}%  median {np.median(R) * 100:+.2f}%  "
          f"std {R.std(ddof=1) * 100:.1f}%  win {(R > 0).mean():.0%}")
    print(f"  t-stat {tstat(R):+.2f}   bootstrap 95% CI on the mean "
          f"[{boot_ci(R)[0] * 100:+.2f}%, {boot_ci(R)[1] * 100:+.2f}%]")
    print(f"  total {R.sum() * 100:+.1f}%")
    print(f"  best 3 returns: {[f'{x * 100:+.0f}%' for x in srt[:3]]}  "
          f"sum {srt[:3].sum() * 100:+.1f}%  = {srt[:3].sum() / R.sum() * 100:.0f}% "
          f"of all P&L")
    print(f"  best 5 returns: {[f'{x * 100:+.0f}%' for x in srt[:5]]}  "
          f"= {srt[:5].sum() / R.sum() * 100:.0f}% of all P&L")
    print(f"  worst 3:        {[f'{x * 100:+.0f}%' for x in srt[-3:]]}")
    for k in (1, 2, 3, 5):
        rest = srt[k:]
        print(f"  drop top {k}: {desc(rest)}  t={tstat(rest):+.2f}")
    print(f"  25% trimmed mean: {np.mean(np.sort(R)[len(R) // 4: -len(R) // 4]) * 100:+.2f}%")
    q = np.percentile(R, [10, 25, 50, 75, 90])
    print(f"  deciles/quartiles p10 {q[0] * 100:+.0f}% p25 {q[1] * 100:+.0f}% "
          f"p50 {q[2] * 100:+.0f}% p75 {q[3] * 100:+.0f}% p90 {q[4] * 100:+.0f}%")
    # leave-one-ticker-out
    print("\n  leave-one-ticker-out (mean/trade of what remains):")
    for tk in sorted(by_tk):
        rest = np.array([x["ret"] for x in F if x["tk"] != tk])
        print(f"    without {tk:<5} {desc(rest)} t={tstat(rest):+.2f}")

    # ----------------------------------- 5. same-signal cross-bucket (the test)
    hr()
    print("5. STRUCTURE OR SIGNAL SELECTION? — same 58 signals, other expiry buckets")
    hr("-")
    keys58 = {(x["tk"], x["t0"]) for x in F}
    print("  If long DTE is what helps, 2-3w/4-6w should be WORSE on these same signals.")
    print("  If the 58 signals were simply the good ones, every bucket is positive on them.")
    for bname, lo, hi in BUCKETS:
        for sname, mult in STRIKES:
            FF, _ = grid[(bname, sname)]
            sub = np.array([x["ret"] for x in FF if (x["tk"], x["t0"]) in keys58])
            oth = np.array([x["ret"] for x in FF if (x["tk"], x["t0"]) not in keys58])
            print(f"    {bname:>6}/{sname:<5} on the 58: {desc(sub)}   | "
                  f"on everything else: {desc(oth)}")

    hr("-")
    print("  STOCK leg of the same split (options aside — were these just better trades?):")
    s_in = np.array([t["stock_ret"] for t in allt if (t["tk"], t["t0"]) in keys58])
    s_out = np.array([t["stock_ret"] for t in allt if (t["tk"], t["t0"]) not in keys58])
    print(f"    filled-58 stock : n={len(s_in):>3} win {(s_in > 0).mean():.0%} "
          f"avg {s_in.mean() * 1e4:+.0f}bps  med {np.median(s_in) * 1e4:+.0f}bps")
    print(f"    skipped   stock : n={len(s_out):>3} win {(s_out > 0).mean():.0%} "
          f"avg {s_out.mean() * 1e4:+.0f}bps  med {np.median(s_out) * 1e4:+.0f}bps")
    print(f"    hold length (calendar days): filled avg "
          f"{np.mean([x['hold_days'] for x in F]):.1f}")

    # -------------------------------------------------- 6. DTE window sweep
    hr()
    print("6. DTE-WINDOW STABILITY (ATM)")
    hr("-")
    print("  NOTE: the contract cache physically stops at signal_day+50, so any window")
    print("  reaching past 50 is silently truncated. Reachable ranges only:")
    print(f"  {'window':>10} {'reach':>7} {'n':>4} {'skip':>5} {'win%':>6} "
          f"{'avg':>8} {'med':>8} {'tot':>9} {'t':>6}")
    windows = [(12, 25), (26, 45), (30, 50), (35, 50), (40, 50), (45, 50),
               (45, 85), (48, 52), (50, 90), (55, 95), (46, 50), (42, 48)]
    for lo, hi in windows:
        FF, SS = [], []
        for tk in tsmap:
            f, s = replay_inst([t for t in allt if t["tk"] == tk], lo, hi, 1.00,
                               *tsmap[tk])
            FF += f
            SS += s
        reach = f"{lo}-{min(hi, 50)}"
        if not FF:
            print(f"  {f'{lo}-{hi}':>10} {reach:>7} {0:>4} {len(SS):>5}   "
                  f"NO FILLS (window unreachable in cache)")
            continue
        r = np.array([x["ret"] for x in FF])
        dts = sorted(Counter(x["dte"] for x in FF).items())
        print(f"  {f'{lo}-{hi}':>10} {reach:>7} {len(r):>4} {len(SS):>5} "
              f"{(r > 0).mean():>6.0%} {r.mean() * 100:>+7.1f}% "
              f"{np.median(r) * 100:>+7.1f}% {r.sum() * 100:>+8.0f}% "
              f"{tstat(r):>+6.2f}   dte={dts[:4]}{'...' if len(dts) > 4 else ''}")

    # ------------------------------------------- 7. random subsets / splits
    hr()
    print("7. SUBSAMPLE STABILITY OF THE 58")
    hr("-")
    rng = np.random.default_rng(11)
    means = []
    for _ in range(2000):
        idx = rng.choice(len(R), size=len(R) // 2, replace=False)
        means.append(R[idx].mean())
    means = np.array(means)
    print(f"  2000 random half-samples (n=29): mean of means "
          f"{means.mean() * 100:+.2f}%, {(means < 0).mean():.0%} of halves NEGATIVE, "
          f"p5..p95 [{np.percentile(means, 5) * 100:+.1f}%, "
          f"{np.percentile(means, 95) * 100:+.1f}%]")
    order = np.argsort([x["t0"].value for x in F])
    Ro = R[order]
    h1, h2 = Ro[:len(Ro) // 2], Ro[len(Ro) // 2:]
    print(f"  chronological 1st half: {desc(h1)} t={tstat(h1):+.2f}")
    print(f"  chronological 2nd half: {desc(h2)} t={tstat(h2):+.2f}")
    print(f"  sign test: {(R > 0).sum()}/{len(R)} positive "
          f"(coin-flip would give {len(R) / 2:.0f})")
    print(f"  P(mean>0 | resampling): {(np.array(boot_ci(R, 20000)) > 0).all()}")

    hr()
    print("VERDICT INPUTS SUMMARY")
    hr("-")
    print(f"  fills {len(R)} / candidates {len(R) + len(S)} = "
          f"{len(R) / (len(R) + len(S)):.0%} coverage")
    print(f"  every fill has DTE {sorted(dte_hist)} (not 56-84 as '8-12w' implies)")
    print(f"  mean {R.mean() * 100:+.2f}% median {np.median(R) * 100:+.2f}% "
          f"t={tstat(R):+.2f} top-3 = {srt[:3].sum() / R.sum() * 100:.0f}% of P&L")


if __name__ == "__main__":
    main()
