"""
morning_ml.py — ML META-LABEL FILTER on the LOOSE-NR morning breakout.

Idea (the untried lane that directly targets ~50% of days): the two-sided 25-min ORB
with a LOOSE or NO narrow-range filter fires on ~62-90% of days but is final-year
NEGATIVE unfiltered.  Train a classifier to predict which breakouts WIN and trade only
those (meta-labeling — same concept as this project's original validated v3/v4 family).

BASE SIGNAL (fixed, from the validated champion family — not tuned here):
    orb2s entries with or_bars=5, rr=2.0, last_entry=690 (11:30 ET), sides='both',
    nr in {0.7, None}.  Trades come from morning_opt.orb2s_trade (entry/exit
    timestamps exposed; verified vs morning_qqq4.orb2s in this file's smoke check).

FEATURES — entry-time knowable ONLY:
    gap, pm_ret, pm_rng, pm_vol_x, nr_rank, atr_pct (atr/prev_close),
    or_width_pct (OR range / open), or_width_vs_atr, side (+1 long / -1 short),
    entry_minute, bars_waited (entry bar index - or_bars), day_of_week,
    prior-day return, dist to prev_high / prev_low (from entry price),
    above_ma50 / above_ma200 flags, and cross-asset premarket returns
    spy_pm / tlt_pm via morning_qqq3.load_aux (QQQ base -> no qqq_pm self-feature).
LABEL: trade net return > 0 at 2bps.

WALK-FORWARD (no lookahead): expanding window, retrain at each half-year boundary
(Jan-14 / Jul-14) from 2022-01-14 on; each model predicts ONLY the following half and
saw ONLY earlier trades.  Min 120 training trades else that half is SKIPPED (no
trades taken there).  Models: lightgbm small (n_estimators=200, num_leaves=15,
min_child_samples=20, lr=0.05) and sklearn HistGradientBoosting (same-size).
ACCEPTANCE: keep a breakout iff predicted p(win) > the q-quantile of the TRAIN-set
predictions, q in {0.2, 0.3, 0.4, 0.5} (keep-rate ~= 1-q).
GRID: 2 base-nr x 2 models x 2 feature-sets (full / no-cross-asset) x 4 q = 32.

FREQ FLOOR (the point of this round): kept-trade-day fraction >= 0.45 overall AND in
gate AND in final — a config that filters down to vM-like sparsity fails even if
profitable.

HONESTY LADDER (unchanged): selection on ARENA (worst of 5 half-years 2022-01..
2024-07, min 8 trades/half, must be > 0) plus the arena-period freq floor; ONE gate
look (2024-07-14..2025-07-14) for arena survivors; ONE final look (2025-07-14..now)
for gate survivors with 0/2/5bps sensitivity.  Champion designation among full-ladder
passers is PRE-COMMITTED to best arena-worst (tie: higher overall freq) — never
chosen on gate/final numbers.  Unfiltered base ladders shown as reference (their
final-year failure is the premise of this round).  Research only — no orders.

Usage:  python morning_ml.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from morning_qqq import COST, SUBS, GATE, FINAL, load, days, window
from morning_qqq3 import add_nr, load_aux, attach_aux
from morning_qqq4 import orb2s
from morning_opt import orb2s_trade

OR_BARS, RR, LAST_ENTRY, SIDES = 5, 2.0, 690, "both"
NRS = [0.7, None]
MODELS = ["lgbm", "hgb"]
QS = [0.2, 0.3, 0.4, 0.5]
MIN_TRAIN = 120
FREQ_FLOOR = 0.45
ARENA = (SUBS[0][0], SUBS[-1][1])            # 2022-01-14 .. 2024-07-14
OVERALL = (SUBS[0][0], "2099-01-01")         # everything the walk-forward can trade

FEATS_FULL = ["gap", "pm_ret", "pm_rng", "pm_vol_x", "nr_rank", "atr_pct",
              "or_width_pct", "or_width_vs_atr", "side", "entry_minute",
              "bars_waited", "day_of_week", "prior_ret", "dist_prev_high",
              "dist_prev_low", "above_ma50", "above_ma200", "spy_pm", "tlt_pm"]
FEATS_NOX = [f for f in FEATS_FULL if f not in ("spy_pm", "tlt_pm")]
FSETS = {"full": FEATS_FULL, "nox": FEATS_NOX}


def _nn(x):
    return np.nan if x is None else float(x)


# ------------------------------------------------------------- trade + feature rows
def build_trades(dd, nr):
    """One row per base-signal trade: entry-time features + gross/net return + label."""
    rows, prev_pc = [], None
    for d in dd:
        pc = d["prev_close"]
        prior_ret = (pc / prev_pc - 1) if (pc and prev_pc) else np.nan
        t = orb2s_trade(d, OR_BARS, RR, nr=nr, last_entry=LAST_ENTRY, sides=SIDES)
        if t is not None:
            am = d["am"]
            m = am["min"].to_numpy()
            ts = t["entry_ts"]
            entry_minute = ts.hour * 60 + ts.minute
            i = int(np.searchsorted(m, entry_minute))
            or_hi = float(am["high"].iloc[:OR_BARS].max())
            or_lo = float(am["low"].iloc[:OR_BARS].min())
            orw = or_hi - or_lo
            e = float(t["entry_px"])
            atr = d["atr"]
            rows.append({
                "date": d["date"],
                "gap": _nn(d["gap"]),
                "pm_ret": _nn(d["pm_ret"]),
                "pm_rng": ((d["pm_hi"] - d["pm_lo"]) / pc
                           if d["pm_hi"] is not None and d["pm_lo"] is not None and pc
                           else np.nan),
                "pm_vol_x": _nn(d["pm_vol_x"]),
                "nr_rank": _nn(d.get("nr_rank")),
                "atr_pct": (atr / pc if atr and pc else np.nan),
                "or_width_pct": orw / d["open"],
                "or_width_vs_atr": (orw / atr if atr else np.nan),
                "side": 1.0 if t["side"] == "long" else -1.0,
                "entry_minute": float(entry_minute),
                "bars_waited": float(i - OR_BARS),
                "day_of_week": float(pd.Timestamp(d["date"]).weekday()),
                "prior_ret": prior_ret,
                "dist_prev_high": ((e - d["prev_high"]) / d["prev_high"]
                                   if d["prev_high"] else np.nan),
                "dist_prev_low": ((e - d["prev_low"]) / d["prev_low"]
                                  if d["prev_low"] else np.nan),
                "above_ma50": (float(pc > d["ma50"]) if pc and d["ma50"] else np.nan),
                "above_ma200": (float(pc > d["ma200"]) if pc and d["ma200"] else np.nan),
                "spy_pm": _nn(d.get("spy_pm")),
                "tlt_pm": _nn(d.get("tlt_pm")),
                "ret_gross": t["stock_ret"],
                "ret_net": t["stock_ret_net"],
                "label": int(t["stock_ret_net"] > 0),
            })
        prev_pc = pc
    return pd.DataFrame(rows)


def verify_vs_orb2s(dd, nr, tr):
    """orb2s_trade must reproduce morning_qqq4.orb2s exactly for this base config."""
    refs = [(d["date"], orb2s(d, OR_BARS, RR, nr=nr, last_entry=LAST_ENTRY, sides=SIDES))
            for d in dd]
    refs = [(dt, r) for dt, r in refs if r is not None]
    assert len(refs) == len(tr), f"nr={nr}: {len(refs)} ref vs {len(tr)} trade rows"
    for (dt, r), rn in zip(refs, tr["ret_net"].to_numpy()):
        assert abs(r - rn) < 1e-12, f"nr={nr} {dt}: {r} vs {rn}"


# ------------------------------------------------------------------- walk-forward
def half_boundaries(max_date):
    bs, y = [], 2022
    while True:
        for md in ("01-14", "07-14"):
            b = pd.Timestamp(f"{y}-{md}").date()
            if b > max_date:
                return bs
            bs.append(b)
        y += 1


def make_model(kind):
    if kind == "lgbm":
        return lgb.LGBMClassifier(n_estimators=200, num_leaves=15,
                                  min_child_samples=20, learning_rate=0.05,
                                  random_state=0, n_jobs=1, verbosity=-1)
    return HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15,
                                          min_samples_leaf=20, learning_rate=0.05,
                                          random_state=0)


def walk_forward(tr, feats, kind):
    """Expanding-window walk-forward.  Returns (keep{q: bool mask}, scored mask,
    fits [(boundary, model, n_train)], n_skipped)."""
    dates = tr["date"].to_numpy()
    X = tr[feats].to_numpy(dtype=float)
    y = tr["label"].to_numpy()
    keep = {q: np.zeros(len(tr), dtype=bool) for q in QS}
    scored = np.zeros(len(tr), dtype=bool)
    bs = half_boundaries(dates.max())
    fits, skipped = [], 0
    for k, b in enumerate(bs):
        b_end = bs[k + 1] if k + 1 < len(bs) else pd.Timestamp("2099-01-01").date()
        tr_mask = dates < b
        te_mask = (dates >= b) & (dates < b_end)
        if not te_mask.any():
            continue
        if tr_mask.sum() < MIN_TRAIN:
            skipped += 1
            continue
        mdl = make_model(kind)
        mdl.fit(X[tr_mask], y[tr_mask])
        p_tr = mdl.predict_proba(X[tr_mask])[:, 1]
        p_te = mdl.predict_proba(X[te_mask])[:, 1]
        scored |= te_mask
        for q in QS:
            thr = float(np.quantile(p_tr, q))
            m2 = te_mask.copy()
            m2[te_mask] = p_te > thr
            keep[q] |= m2
        fits.append((b, mdl, int(tr_mask.sum())))
    return keep, scored, fits, skipped


# ------------------------------------------------------------------ stats helpers
def in_win(dates, lo, hi):
    lo, hi = pd.Timestamp(lo).date(), pd.Timestamp(hi).date()
    return np.array([lo <= d < hi for d in dates])


def seg(tr, mask, lo, hi, col="ret_net"):
    m = mask & in_win(tr["date"].to_numpy(), lo, hi)
    return tr.loc[m, col].to_numpy()


def freq(tr, mask, n_days_map, key):
    lo, hi = key
    n = int((mask & in_win(tr["date"].to_numpy(), lo, hi)).sum())
    nd = n_days_map[key]
    return n / nd if nd else 0.0


def ladder_stats(tr, mask, n_days_map):
    """Arena subs/worst + freqs for a kept-trade mask (counts only outside arena)."""
    subs, ns = [], []
    for lo, hi in SUBS:
        r = seg(tr, mask, lo, hi)
        ns.append(len(r))
        subs.append(float(r.sum() * 100) if len(r) >= 8 else -99.0)
    return {
        "subs": subs, "ns": ns, "worst": min(subs),
        "f_arena": freq(tr, mask, n_days_map, ARENA),
        "f_gate": freq(tr, mask, n_days_map, GATE),
        "f_final": freq(tr, mask, n_days_map, FINAL),
        "f_all": freq(tr, mask, n_days_map, OVERALL),
    }


def final_line(r_net, r_gross):
    t = r_net.mean() / r_net.std() * np.sqrt(len(r_net)) if r_net.std() > 0 else 0.0
    return {"n": len(r_net), "win": float((r_net > 0).mean()),
            "avg_bp": float(r_net.mean() * 1e4), "tot2": float(r_net.sum() * 100),
            "t": float(t), "tot0": float(r_gross.sum() * 100),
            "tot5": float((r_gross - 5e-4).sum() * 100)}


def base_reference(tr, nr_lab, n_days_map):
    all_mask = np.ones(len(tr), dtype=bool)
    st = ladder_stats(tr, all_mask, n_days_map)
    g = seg(tr, all_mask, *GATE)
    fn = seg(tr, all_mask, *FINAL)
    fg = seg(tr, all_mask, *FINAL, col="ret_gross")
    fl = final_line(fn, fg)
    print(f"  BASE nr={nr_lab:<4} worst={st['worst']:+7.2f}%  "
          f"subs={[round(s, 1) for s in st['subs']]} n/half={st['ns']}")
    print(f"       gate n={len(g)} tot={g.sum()*100:+.2f}% | "
          f"final n={fl['n']} win={fl['win']:.0%} avg={fl['avg_bp']:+.1f}bp "
          f"tot={fl['tot2']:+.2f}% t={fl['t']:+.2f} "
          f"(0bps {fl['tot0']:+.2f}% / 5bps {fl['tot5']:+.2f}%)")
    print(f"       trade-day freq: overall {st['f_all']:.2f} arena {st['f_arena']:.2f} "
          f"gate {st['f_gate']:.2f} final {st['f_final']:.2f}")


# ------------------------------------------------------------- champion reporting
def champion_report(cfg, tr, mask, scored, fits, n_days_map, feats):
    nr_lab, kind, fs, q = cfg
    kept = tr[mask]
    print(f"\n=== CHAMPION DETAIL: nr={nr_lab} model={kind} fset={fs} q={q} ===")
    n_sc = int(scored.sum())
    print(f"  base trades scored={n_sc}  kept={len(kept)}  "
          f"keep-rate={len(kept)/n_sc:.0%} (target ~{1-q:.0%})")
    print(f"  trade-day freq: overall={freq(tr, mask, n_days_map, OVERALL):.2f} "
          f"arena={freq(tr, mask, n_days_map, ARENA):.2f} "
          f"gate={freq(tr, mask, n_days_map, GATE):.2f} "
          f"final={freq(tr, mask, n_days_map, FINAL):.2f}")
    rets = pd.Series(kept["ret_net"].to_numpy(),
                     index=pd.to_datetime(kept["date"].to_numpy()))
    print(f"  all kept: n={len(rets)} win={float((rets > 0).mean()):.1%} "
          f"avg={rets.mean()*1e4:+.1f}bp total={rets.sum()*100:+.2f}%")
    print("  per-year:")
    for y, r in rets.groupby(rets.index.year):
        print(f"    {y}: {r.sum()*100:+7.2f}%  (n={len(r)}, win={(r > 0).mean():.0%}, "
              f"avg={r.mean()*1e4:+.1f}bp)")
    print(f"  walk-forward fits={len(fits)} "
          f"(train sizes {fits[0][2]}..{fits[-1][2]})" if fits else "  no fits")
    # what the model learned
    if kind == "lgbm":
        imp = np.zeros(len(feats))
        for _, mdl, _ in fits:
            gi = mdl.booster_.feature_importance(importance_type="gain")
            imp += gi / gi.sum() if gi.sum() else 0
        imp /= max(len(fits), 1)
        src = "lgbm gain, avg over walk-forward fits"
    else:
        b, mdl, _ = fits[-1]
        dts = tr["date"].to_numpy()
        trm = dts < b
        r = permutation_importance(mdl, tr.loc[trm, feats].to_numpy(dtype=float),
                                   tr.loc[trm, "label"].to_numpy(),
                                   n_repeats=5, random_state=0)
        imp = np.clip(r.importances_mean, 0, None)
        imp = imp / imp.sum() if imp.sum() else imp
        src = "permutation importance on last train set"
    order = np.argsort(imp)[::-1]
    print(f"  top features ({src}):")
    for j in order[:8]:
        print(f"    {feats[j]:<18} {imp[j]:.3f}")


# --------------------------------------------------------------------------- main
def main():
    df = load()
    dd = days(df)
    add_nr(dd)
    for tk, key in (("SPY", "spy_pm"), ("TLT", "tlt_pm")):
        attach_aux(dd, load_aux(tk), key)
    n_days_map = {k: len(window(dd, *k)) for k in
                  (ARENA, GATE, FINAL, OVERALL)}
    print(f"MORNING-ML meta-label filter | base=orb2s({OR_BARS} bars, rr={RR}, "
          f"last_entry={LAST_ENTRY}, {SIDES}) nr in {{0.7, None}} | "
          f"{len(dd)} days {dd[0]['date']}..{dd[-1]['date']} | cost {COST*1e4:.0f}bps")
    print(f"days: arena={n_days_map[ARENA]} gate={n_days_map[GATE]} "
          f"final={n_days_map[FINAL]} overall={n_days_map[OVERALL]}\n")

    trades, results = {}, {}
    print("=== UNFILTERED BASE LADDERS (reference — the thing being fixed) ===")
    for nr in NRS:
        tr = build_trades(dd, nr)
        verify_vs_orb2s(dd, nr, tr)
        trades[nr] = tr
        nr_lab = "none" if nr is None else str(nr)
        n_pre = int((tr["date"] < pd.Timestamp(SUBS[0][0]).date()).sum())
        base_reference(tr, nr_lab, n_days_map)
        print(f"       trades total={len(tr)}  before first boundary "
              f"{SUBS[0][0]}: {n_pre} (min_train={MIN_TRAIN})\n")

    print("=== WALK-FORWARD GRID: 2 nr x 2 models x 2 fsets x 4 q = 32 configs ===")
    for nr in NRS:
        for kind in MODELS:
            for fs, feats in FSETS.items():
                keep, scored, fits, skipped = walk_forward(trades[nr], feats, kind)
                for q in QS:
                    results[(nr, kind, fs, q)] = (keep[q], scored, fits, skipped)

    print(f"\n--- ARENA (worst of 5 halves > 0 AND arena freq >= {FREQ_FLOOR}) ---")
    print(f"  {'config':<22} {'worst':>8} {'fA':>5} {'fAll':>5} skip  subs / n-half")
    arena_pass = []
    for (nr, kind, fs, q), (mask, scored, fits, skipped) in results.items():
        tr = trades[nr]
        st = ladder_stats(tr, mask, n_days_map)
        nr_lab = "none" if nr is None else str(nr)
        name = f"nr{nr_lab}|{kind}|{fs}|q{q}"
        ok = st["worst"] > 0 and st["f_arena"] >= FREQ_FLOOR
        flag = "  <-- ARENA PASS" if ok else ""
        print(f"  {name:<22} {st['worst']:>+7.2f}% {st['f_arena']:>5.2f} "
              f"{st['f_all']:>5.2f} {skipped:>4}  "
              f"{[round(s, 1) for s in st['subs']]} {st['ns']}{flag}")
        if ok:
            arena_pass.append(((nr, kind, fs, q), st))

    print(f"\n=== GATE {GATE[0]}..{GATE[1]} — one look for {len(arena_pass)} arena "
          f"survivors (tot > 0 AND gate freq >= {FREQ_FLOOR}) ===")
    gate_pass = []
    for cfg, st in arena_pass:
        nr, kind, fs, q = cfg
        mask = results[cfg][0]
        r = seg(trades[nr], mask, *GATE)
        tot = r.sum() * 100 if len(r) >= 10 else -99.0
        nr_lab = "none" if nr is None else str(nr)
        name = f"nr{nr_lab}|{kind}|{fs}|q{q}"
        ok = tot > 0 and st["f_gate"] >= FREQ_FLOOR
        flag = "  <-- GATE PASS" if ok else ""
        print(f"  {name:<22} n={len(r):>3} win={(r > 0).mean() if len(r) else 0:.0%} "
              f"tot={tot:+.2f}% freq={st['f_gate']:.2f}{flag}")
        if ok:
            gate_pass.append((cfg, st))

    print(f"\n=== FINAL one-shot {FINAL[0]}..now — {len(gate_pass)} gate survivors "
          f"(freq floor {FREQ_FLOOR} in final AND overall) ===")
    finalists = []
    for cfg, st in gate_pass:
        nr, kind, fs, q = cfg
        mask = results[cfg][0]
        rn = seg(trades[nr], mask, *FINAL)
        rg = seg(trades[nr], mask, *FINAL, col="ret_gross")
        nr_lab = "none" if nr is None else str(nr)
        name = f"nr{nr_lab}|{kind}|{fs}|q{q}"
        if len(rn) < 10:
            print(f"  {name:<22} too few trades ({len(rn)})")
            continue
        fl = final_line(rn, rg)
        ok = (fl["tot2"] > 0 and st["f_final"] >= FREQ_FLOOR
              and st["f_all"] >= FREQ_FLOOR)
        flag = "  <-- FULL-LADDER PASS" if ok else ""
        print(f"  {name:<22} n={fl['n']:>3} win={fl['win']:.0%} "
              f"avg={fl['avg_bp']:+.1f}bp tot={fl['tot2']:+.2f}% t={fl['t']:+.2f} | "
              f"0bps {fl['tot0']:+.2f}% / 5bps {fl['tot5']:+.2f}% | "
              f"freq fin={st['f_final']:.2f} all={st['f_all']:.2f}{flag}")
        if ok:
            finalists.append((cfg, st, fl))

    if finalists:
        # pre-committed champion rule: best arena worst, tie -> higher overall freq
        finalists.sort(key=lambda x: (x[1]["worst"], x[1]["f_all"]), reverse=True)
        cfg, st, fl = finalists[0]
        nr, kind, fs, q = cfg
        mask, scored, fits, _ = results[cfg]
        nr_lab = "none" if nr is None else str(nr)
        champion_report((nr_lab, kind, fs, q), trades[nr], mask, scored, fits,
                        n_days_map, FSETS[fs])
    else:
        print("\nNO full-ladder passer met the frequency floor — see near-misses above.")


if __name__ == "__main__":
    main()
