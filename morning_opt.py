"""
morning_opt.py — REAL-OPTIONS REPLAY HARNESS for QQQ morning strategies.

Replays QQQ morning STOCK trades as OPTION trades using REAL Polygon option
5-minute bars (same fetch/cache/contract-selection patterns proven in
qqq_options_real.py / vc_options_real.py; cache in data_cache/options/).

Public surface:
  replay(trades, dte_bucket, strike_mode) -> list of per-trade result dicts.
      trades: list of dicts {date, entry_ts (bar timestamp ET), entry_px
      (underlying), exit_ts, exit_px, side ('long'/'short'), exit_kind
      ('target'/'stop'/'noon')}.
      side long -> CALL, short -> PUT. Contract picked at entry:
        dte_bucket in {'0dte' (same-day if listed else nearest), '1-3d', '4-10d'}
        strike_mode in {'atm', 'otm1'} (one strike step OTM).
      Entry = close of the option's real 5-min bar at/just after entry_ts,
      +1% cost; exit = close of the option bar NEAREST exit_ts, -1% cost.
      If no bar within 15 min of a needed timestamp the trade is UNFILLABLE —
      kept in the output with status set, never silently dropped into P&L.
  summarize(results) -> dict with n / fill counts / unfillable rate / P&L
      (filled trades only — unfillable & no-contract reported separately).
  orb2s_trade(day, ...) -> trade dict; faithful re-implementation of
      morning_qqq4.orb2s exposing entry/exit bar timestamps (existing
      morning_*.py files are NOT modified).
  trade_list(fn, start=None, end=None) -> run a day->trade-dict strategy over
      morning_qqq.days() (with nr_rank attached) and emit the trades list.
  vm_trade(day) -> current champion vM = orb2s(5, 2.0, nr=0.3, 690, 'both').

Research only — no orders anywhere.
Usage:  python morning_opt.py            (smoke test + coverage probe)
        python morning_opt.py probe      (coverage probe only)
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

import data  # truststore SSL + .env — MUST precede `import requests`
import requests
from config import default_config

from morning_qqq import COST, load, days
from morning_qqq3 import add_nr

_ET = ZoneInfo("America/New_York")
CACHE = Path(__file__).with_name("data_cache") / "options"
AGGS = CACHE / "aggs5"
AGGS.mkdir(parents=True, exist_ok=True)

OPT_COST = 0.01                      # 1% per side on the option premium
MAX_GAP_MS = 15 * 60_000             # no bar within 15 min -> unfillable
DTE_BUCKETS = {"0dte": (0, 0), "1-3d": (1, 3), "4-10d": (4, 10)}
STRIKE_MODES = ("atm", "otm1")


# ------------------------------------------------------------------ HTTP layer
def _session_key():
    return data._make_session(), default_config().api_key


def _get(session, key, url, params, retries=6):
    params = {**params, "apiKey": key}
    for i in range(retries):
        try:
            r = session.get(url, params=params, timeout=45)
        except requests.exceptions.RequestException:
            time.sleep(5 * (i + 1))          # transient drop -> retry
            continue
        if r.status_code == 429:             # rate limit -> backoff
            time.sleep(10 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"request failed after {retries} retries: {url}")


# ------------------------------------------------- contracts + option minute bars
_CONS: dict = {}


def contracts_for(day, klo, khi, ctype, max_days=12):
    """QQQ contracts of `ctype` expiring day..day+max_days, strikes in [klo,khi]."""
    key = f"QQQ_{ctype}_{day}_{int(klo)}_{int(khi)}_{max_days}"
    cf = CACHE / f"mo_cons_{key}.json"
    if key in _CONS:
        return _CONS[key]
    if cf.exists():
        _CONS[key] = json.loads(cf.read_text())
        return _CONS[key]
    s, k = _session_key()
    out = []
    for expired in ("true", "false"):
        url = "https://api.polygon.io/v3/reference/options/contracts"
        params = {"underlying_ticker": "QQQ", "contract_type": ctype,
                  "expiration_date.gte": str(day),
                  "expiration_date.lte": str(day + dt.timedelta(days=max_days)),
                  "strike_price.gte": klo, "strike_price.lte": khi,
                  "expired": expired, "limit": 1000}
        while True:
            j = _get(s, k, url, params)
            out += [{"ticker": r["ticker"], "K": float(r["strike_price"]),
                     "exp": r["expiration_date"]} for r in j.get("results", [])]
            nxt = j.get("next_url")
            if not nxt:
                break
            url, params = nxt, {}
    # dedupe (a contract can appear in both expired sweeps around the flip date)
    seen, ded = set(), []
    for c in out:
        if c["ticker"] not in seen:
            seen.add(c["ticker"])
            ded.append(c)
    cf.write_text(json.dumps(ded))
    _CONS[key] = ded
    return ded


def bars_for(ticker, d0, d1):
    """Real 5-min option bars (Polygon aggs), cached — same layout qqq_options_real
    uses. Returns a list ([] = entitled but the contract printed no trades) or
    None = HTTP 403, i.e. the date is OUTSIDE the plan's history window."""
    cf = AGGS / f"{ticker.replace(':', '_')}_{d0}_{d1}.json"
    if cf.exists():
        return json.loads(cf.read_text())
    s, k = _session_key()
    try:
        j = _get(s, k,
                 f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{d0}/{d1}",
                 {"limit": 50000, "adjusted": "true"})
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            return None                       # no entitlement for this date range
        return []                             # do not cache other failures
    except RuntimeError:
        return []                             # do not cache transport failures
    bars = j.get("results", []) or []
    cf.write_text(json.dumps(bars))
    return bars


# ------------------------------------------------------------------ time helpers
def to_utc_ms(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:                      # project CSVs carry naive-UTC stamps
        t = t.tz_localize("UTC")
    return int(t.tz_convert("UTC").value // 1_000_000)


def _et_ts(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert(_ET)


# ------------------------------------------------------------- contract selection
def pick_contracts(day, spot, side, dte_bucket, strike_mode):
    """Choose the expiry per dte_bucket and strike per strike_mode at entry.

    Returns (candidates, dte) — candidates is the target contract first plus up
    to two nearest-strike fallbacks in the SAME expiry (liquidity insurance,
    same pattern as qqq_options_real), or ([], None) if nothing is listed."""
    ctype = "call" if side == "long" else "put"
    cons = contracts_for(day, spot * 0.97, spot * 1.03, ctype)
    by_dte: dict = {}
    for c in cons:
        d = (dt.date.fromisoformat(c["exp"]) - day).days
        if d >= 0:
            by_dte.setdefault(d, []).append(c)
    if not by_dte:
        return [], None
    dtes = sorted(by_dte)
    if dte_bucket == "0dte":
        use = 0 if 0 in by_dte else dtes[0]   # same-day if listed, else nearest
    else:
        lo, hi = DTE_BUCKETS[dte_bucket]
        in_b = [d for d in dtes if lo <= d <= hi]
        if not in_b:
            return [], None
        use = in_b[0]
    pool = by_dte[use]
    ks = sorted({c["K"] for c in pool})
    i_atm = int(np.argmin([abs(k - spot) for k in ks]))
    if strike_mode == "atm":
        i_tgt = i_atm
    elif strike_mode == "otm1":               # one strike step OTM of ATM
        i_tgt = i_atm + 1 if side == "long" else i_atm - 1
        if not (0 <= i_tgt < len(ks)):
            i_tgt = i_atm                     # edge of listed grid -> stay ATM
    else:
        raise ValueError(f"strike_mode must be in {STRIKE_MODES}, got {strike_mode!r}")
    k_tgt = ks[i_tgt]
    order = sorted(pool, key=lambda c: abs(c["K"] - k_tgt))
    return order[:3], use


# ------------------------------------------------------------------ fill matching
def _entry_fill(bt, bars, t_ms):
    """Close of the bar at/just after t_ms (within 15 min); None if absent."""
    i = int(np.searchsorted(bt, t_ms, side="left"))
    if i >= len(bars) or bt[i] - t_ms > MAX_GAP_MS:
        return None, None
    return float(bars[i]["c"]), int(bt[i])


def _exit_fill(bt, bars, t_ms):
    """Close of the bar NEAREST t_ms, either side (within 15 min)."""
    i = int(np.searchsorted(bt, t_ms, side="left"))
    best, bts = None, None
    for j in (i - 1, i):
        if 0 <= j < len(bars):
            gap = abs(int(bt[j]) - t_ms)
            if gap <= MAX_GAP_MS and (best is None or gap < best[0]):
                best, bts = (gap, float(bars[j]["c"])), int(bt[j])
    return (best[1], bts) if best else (None, None)


# ------------------------------------------------------------------------ replay
def replay(trades, dte_bucket, strike_mode, verbose=False):
    """Replay stock trades as REAL option trades. Returns one result dict per
    input trade with status in {'filled','unfillable','no_contract'}.
    opt_ret is set only on 'filled' rows (costs 1% per side already applied)."""
    if dte_bucket not in DTE_BUCKETS:
        raise ValueError(f"dte_bucket must be in {list(DTE_BUCKETS)}, got {dte_bucket!r}")
    if strike_mode not in STRIKE_MODES:
        raise ValueError(f"strike_mode must be in {STRIKE_MODES}, got {strike_mode!r}")
    out = []
    for t in trades:
        d0 = t["date"] if isinstance(t["date"], dt.date) else \
            dt.date.fromisoformat(str(t["date"])[:10])
        side = t["side"]
        spot = float(t["entry_px"])
        res = {"date": d0, "side": side, "exit_kind": t.get("exit_kind"),
               "entry_ts": t["entry_ts"], "exit_ts": t["exit_ts"],
               "entry_px": spot, "exit_px": float(t["exit_px"]),
               "stock_ret_net": t.get("stock_ret_net"),
               "dte_bucket": dte_bucket, "strike_mode": strike_mode,
               "status": "no_contract", "contract": None, "K": None, "exp": None,
               "dte": None, "opt_entry": None, "opt_exit": None, "opt_ret": None}
        cands, dte = pick_contracts(d0, spot, side, dte_bucket, strike_mode)
        if cands:
            res["status"] = "unfillable"      # contract exists; bars must confirm
            res["dte"] = dte
            t0 = to_utc_ms(t["entry_ts"])
            t1 = to_utc_ms(t["exit_ts"])
            d1 = _et_ts(t["exit_ts"]).date()
            for con in cands:
                bars = bars_for(con["ticker"], str(d0), str(max(d0, d1)))
                if not bars:
                    continue
                bt = np.array([b["t"] for b in bars], dtype=np.int64)
                e_px, _ = _entry_fill(bt, bars, t0)
                x_px, _ = _exit_fill(bt, bars, t1)
                if e_px is None or x_px is None or e_px <= 0:
                    continue
                e_eff = e_px * (1 + OPT_COST)
                x_eff = x_px * (1 - OPT_COST)
                res.update(status="filled", contract=con["ticker"], K=con["K"],
                           exp=con["exp"], opt_entry=e_px, opt_exit=x_px,
                           opt_ret=(x_eff - e_eff) / e_eff)
                break
        if verbose:
            print(f"  {d0} {side:>5} {res['exit_kind'] or '?':>6} "
                  f"S {res['entry_px']:.2f}->{res['exit_px']:.2f} | "
                  f"{res['status']:<11} {res['contract'] or '-':<24} "
                  f"opt {res['opt_entry'] if res['opt_entry'] is not None else float('nan'):.2f}"
                  f"->{res['opt_exit'] if res['opt_exit'] is not None else float('nan'):.2f} "
                  f"ret {res['opt_ret'] * 100 if res['opt_ret'] is not None else float('nan'):+.1f}%")
        out.append(res)
    return out


def summarize(results):
    """Aggregate replay() output. Unfillable/no-contract NEVER enter the P&L."""
    n = len(results)
    filled = [r for r in results if r["status"] == "filled"]
    unf = sum(1 for r in results if r["status"] == "unfillable")
    noc = sum(1 for r in results if r["status"] == "no_contract")
    rr = np.array([r["opt_ret"] for r in filled], dtype=float)
    return {"n": n, "filled": len(filled), "unfillable": unf, "no_contract": noc,
            "unfillable_rate": (unf + noc) / n if n else 0.0,
            "win": float((rr > 0).mean()) if len(rr) else None,
            "avg_ret": float(rr.mean()) if len(rr) else None,
            "med_ret": float(np.median(rr)) if len(rr) else None,
            "tot_ret": float(rr.sum()) if len(rr) else None}


# --------------------------------------------------- strategy -> trade-dict plumbing
def orb2s_trade(day, or_bars, rr, nr=None, last_entry=690, sides="both", volx=None):
    """morning_qqq4.orb2s re-implemented with entry/exit bar timestamps exposed
    (existing morning_*.py untouched). Same walk, same stop-before-target order,
    same risk cap; returns a trade dict (stock_ret gross, stock_ret_net = -COST)
    or None. stock_ret_net equals orb2s()'s return exactly (verified in smoke)."""
    am = day["am"]
    if len(am) < or_bars + 3:
        return None
    if nr is not None and (day.get("nr_rank") is None or day["nr_rank"] > nr):
        return None
    if volx is not None and (day.get("pm_vol_x") is None or day["pm_vol_x"] < volx):
        return None
    hi = float(am["high"].iloc[:or_bars].max())
    lo = float(am["low"].iloc[:or_bars].min())
    c = am["close"].to_numpy(); m = am["min"].to_numpy()
    h = am["high"].to_numpy(); l = am["low"].to_numpy()
    tsv = am["timestamp"].to_numpy()

    def mk(i, j, exit_px, side, kind):
        e = float(c[i])
        gross = (exit_px - e) / e if side == "long" else (e - exit_px) / e
        return {"date": day["date"], "side": side,
                "entry_ts": _et_ts(tsv[i]), "entry_px": e,
                "exit_ts": _et_ts(tsv[j]), "exit_px": float(exit_px),
                "exit_kind": kind, "stock_ret": gross,
                "stock_ret_net": gross - COST}

    for i in range(or_bars, len(am)):
        if m[i] > last_entry:
            return None
        if c[i] > hi and sides in ("both", "long"):
            e = c[i]; risk = e - lo
            if risk <= 0 or risk / e > 0.012:
                return None
            tgt = e + rr * risk if rr else None
            for j in range(i + 1, len(am)):     # stop before target, like bracket_to_noon
                if l[j] <= lo:
                    return mk(i, j, lo, "long", "stop")
                if tgt is not None and h[j] >= tgt:
                    return mk(i, j, tgt, "long", "target")
            return mk(i, len(am) - 1, float(c[-1]), "long", "noon")
        if c[i] < lo and sides in ("both", "short"):
            e = c[i]; risk = hi - e
            if risk <= 0 or risk / e > 0.012:
                return None
            tgt = e - rr * risk if rr else None
            for j in range(i + 1, len(am)):
                if h[j] >= hi:
                    return mk(i, j, hi, "short", "stop")
                if tgt is not None and l[j] <= tgt:
                    return mk(i, j, tgt, "short", "target")
            return mk(i, len(am) - 1, float(c[-1]), "short", "noon")
    return None


def vm_trade(day):
    """Current champion vM: orb2s(or_bars=5, rr=2.0, nr=0.3, 690, 'both')."""
    return orb2s_trade(day, 5, 2.0, nr=0.3, last_entry=690, sides="both")


_DD = None


def get_days():
    """Per-day records (morning_qqq.days) with nr_rank attached, cached in-process."""
    global _DD
    if _DD is None:
        dd = days(load())
        add_nr(dd)
        _DD = dd
    return _DD


def trade_list(fn, start=None, end=None):
    """Run a day -> trade-dict strategy over days() and emit the trades list.
    `fn` must return a dict like orb2s_trade's (a plain float has no timestamps
    — wrap the strategy with orb2s_trade-style plumbing first)."""
    dd = get_days()
    if start is not None:
        s = pd.Timestamp(start).date()
        dd = [d for d in dd if d["date"] >= s]
    if end is not None:
        e = pd.Timestamp(end).date()
        dd = [d for d in dd if d["date"] < e]
    out = []
    for d in dd:
        t = fn(d)
        if t is None:
            continue
        if not isinstance(t, dict):
            raise TypeError("trade_list needs a day->trade-DICT strategy "
                            "(e.g. orb2s_trade / vm_trade); a bare float return "
                            "carries no timestamps")
        out.append(t)
    return out


# ------------------------------------------------------------ coverage probe
def probe_coverage(months=("2021-07", "2022-06", "2023-06", "2024-06", "2024-09",
                           "2024-12", "2025-01", "2025-04", "2025-10", "2026-01",
                           "2026-06")):
    """How far back does Polygon QQQ option MINUTE data go? For each month pick a
    mid-month trading day, list the near-ATM call expiring soonest, pull its
    5-min bars for that day, report the bar count (None bars = HTTP 403, i.e.
    the date is outside the plan's history window)."""
    dd = get_days()
    by_month = {}
    for d in dd:
        by_month.setdefault(str(d["date"])[:7], []).append(d)
    print("\nCOVERAGE PROBE — QQQ option 5-min bars by month")
    print(f"  {'month':<8} {'day':<11} {'contract':<24} {'dte':>3} {'bars':>5}  note")
    report = {}
    for mo in months:
        recs = by_month.get(mo)
        if not recs:
            print(f"  {mo:<8} no underlying data in cache")
            report[mo] = None
            continue
        day = recs[min(9, len(recs) - 1)]     # ~mid-month trading day
        d0, S = day["date"], day["open"]
        cands, dte = pick_contracts(d0, S, "long", "0dte", "atm")
        if not cands:
            print(f"  {mo:<8} {d0} no contracts listed near ATM")
            report[mo] = 0
            continue
        con = cands[0]
        bars = bars_for(con["ticker"], str(d0), str(d0))
        if bars is None:
            print(f"  {mo:<8} {d0} {con['ticker']:<24} {dte:>3} {'403':>5}  "
                  f"FORBIDDEN — outside plan history window")
            report[mo] = None
            continue
        note = "OK" if len(bars) >= 30 else ("thin" if bars else "NO MINUTE DATA")
        print(f"  {mo:<8} {d0} {con['ticker']:<24} {dte:>3} {len(bars):>5}  {note}")
        report[mo] = len(bars)
    ok = [m for m, v in report.items() if v]
    if ok:
        print(f"  -> usable QQQ option minute coverage: {min(ok)} .. {max(ok)} "
              f"(earliest probed month with bars)")
    return report


# ------------------------------------------------------------------- smoke test
def smoke():
    import morning_qqq4

    print("SMOKE 1/3 — orb2s_trade must reproduce morning_qqq4.orb2s exactly")
    dd = get_days()
    n_cmp = n_tr = 0
    for d in dd:
        r_ref = morning_qqq4.orb2s(d, 5, 2.0, nr=0.3, last_entry=690, sides="both")
        t_new = vm_trade(d)
        assert (r_ref is None) == (t_new is None), f"presence mismatch {d['date']}"
        if r_ref is not None:
            assert abs(r_ref - t_new["stock_ret_net"]) < 1e-12, \
                f"return mismatch {d['date']}: {r_ref} vs {t_new['stock_ret_net']}"
            n_tr += 1
        n_cmp += 1
    print(f"  {n_cmp} days compared, {n_tr} trades — all identical (net of 2bps)\n")

    print("SMOKE 2/3 — vM breakouts 2026-06-01..2026-07-22 replayed on REAL option bars")
    trades = trade_list(vm_trade, start="2026-06-01", end="2026-07-23")
    trades = trades[:10]
    print(f"  {len(trades)} vM trades in window")
    for t in trades:
        print(f"    {t['date']} {t['side']:>5} entry {t['entry_ts'].strftime('%H:%M')} ET "
              f"@{t['entry_px']:.2f} -> exit {t['exit_ts'].strftime('%H:%M')} "
              f"@{t['exit_px']:.2f} ({t['exit_kind']}) stock {t['stock_ret_net']*1e4:+.0f}bp")
    combos = [("0dte", "atm"), ("0dte", "otm1"), ("1-3d", "atm"), ("4-10d", "atm")]
    for bucket, mode in combos:
        print(f"\n  --- {bucket} / {mode} ---")
        res = replay(trades, bucket, mode, verbose=True)
        s = summarize(res)
        print(f"  SUMMARY {bucket}/{mode}: n={s['n']} filled={s['filled']} "
              f"unfillable={s['unfillable']} no_contract={s['no_contract']} "
              f"(unfillable rate {s['unfillable_rate']:.0%})"
              + (f" | win {s['win']:.0%} avg {s['avg_ret']*100:+.1f}% "
                 f"med {s['med_ret']*100:+.1f}% tot {s['tot_ret']*100:+.1f}%"
                 if s['filled'] else ""))

    print("\nSMOKE 3/3 — coverage probe")
    probe_coverage()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    if cmd == "probe":
        probe_coverage()
    else:
        smoke()
