"""
evolve6.py — TOURNAMENT VI: the DEEP RTH search, armed with session features.

Evolution V's ten islands all failed the RTH day-trading game (5/10 couldn't even find
positive ARENA fitness) — evidence the signal wasn't in F2-F5. This run goes deeper AND
adds the missing information: extended-hours data as INPUTS (trading stays RTH-only):
  F6 = F4 + session features   |   F7 = F2 + session features
  session features (no lookahead): overnight gap, premarket return/range, prior-day
  high/low/close distances, position vs today's open, running opening-range position,
  minutes-since-open, day return so far.

Same island rules as Evo V (10 groups, SAME data, populations cannot interact), but
deeper: 14 agents x 15 generations per island (2,100 nominal evals). Same trade rules:
entries 9:30-16:00 ET only, 2-4h clocks, forced flat by the close.

Honesty ladder: fitness = worst-of-3 half-year arena P&Ls; ALL champions run the gate
(2024-07..2025-07); ONLY gate-positive champions get a FINAL one-shot (2025-07..now) —
the final year has already been looked at by Evo I-V, so every additional peek weakens
it; any survivor still needs live paper confirmation before entering the bot.

  python evolve6.py step   (repeat until complete)   |   python evolve6.py final
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd

from evolve3 import data, _rnn_proba, SUBS, TRAIN_END, GATE0, GATE1, CACHE
from evolve5 import (MODELS5, MODEL_GENE, WINDOWS, info, glabel, sim, gkey,
                     logline as _log5)

GROUPS, POP, GENS, ELITE, IMMI, MUT = 10, 14, 15, 4, 2, 0.40
GENES = {"feat": ["F2", "F3", "F4", "F5", "F6", "F7"],
         "model": MODEL_GENE,
         "tgt": [0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008],
         "stop": [0.001, 0.0015, 0.002, 0.003, 0.004, 0.005],
         "H": [24, 36, 48],
         "gate": ["q", "conf"],
         "qv": [0.85, 0.90, 0.93, 0.95, 0.97],
         "win": ["full", "am", "pm"]}
STATE = Path("runs/evolve6_state.json")
LOGF = Path("runs/evolve6_log.txt")
_D6 = {}


def logline(s):
    print(s, flush=True)
    LOGF.parent.mkdir(exist_ok=True)
    with LOGF.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def data6():
    """feats + F6/F7 with session features (extended-hours info, RTH trading)."""
    if _D6:
        return _D6["v"]
    ts, feats, h, l, c, hours = data()
    rth, mins, last_rth = info()
    idx = pd.DatetimeIndex(ts).tz_localize("UTC").tz_convert("America/New_York")
    dates = np.asarray(idx.date)
    n = len(c)
    gap = np.full(n, np.nan); pm_ret = np.full(n, np.nan); pm_rng = np.full(n, np.nan)
    d_pc = np.full(n, np.nan); d_ph = np.full(n, np.nan); d_pl = np.full(n, np.nan)
    d_open = np.full(n, np.nan); or_pos = np.full(n, np.nan)
    tso = np.full(n, np.nan); day_ret = np.full(n, np.nan)
    prev_close = prev_high = prev_low = None
    i = 0
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            j += 1
        day = slice(i, j)
        r_idx = np.where(rth[day])[0] + i
        p_idx = np.where(mins[day] < 570)[0] + i
        if prev_close is not None and len(r_idx):
            o_px = c[r_idx[0]]
            g = (o_px - prev_close) / prev_close
            if len(p_idx):
                pr = (c[p_idx[-1]] - prev_close) / prev_close
                rng = (h[p_idx].max() - l[p_idx].min()) / prev_close
            else:
                pr, rng = 0.0, 0.0
            run_hi, run_lo = -np.inf, np.inf
            for k in r_idx:
                run_hi, run_lo = max(run_hi, h[k]), min(run_lo, l[k])
                gap[k], pm_ret[k], pm_rng[k] = g, pr, rng
                d_pc[k] = (c[k] - prev_close) / prev_close
                d_ph[k] = (c[k] - prev_high) / prev_close
                d_pl[k] = (c[k] - prev_low) / prev_close
                d_open[k] = (c[k] - o_px) / o_px
                w = run_hi - run_lo
                or_pos[k] = (c[k] - run_lo) / w if w > 0 else 0.5
                tso[k] = (mins[k] - 570) / 390.0
                day_ret[k] = (c[k] - o_px) / o_px
        if len(r_idx):
            prev_close = c[r_idx[-1]]
            prev_high, prev_low = h[r_idx].max(), l[r_idx].min()
        i = j
    sess = pd.DataFrame({"gap": gap, "pm_ret": pm_ret, "pm_rng": pm_rng,
                         "d_pc": d_pc, "d_ph": d_ph, "d_pl": d_pl,
                         "d_open": d_open, "or_pos": or_pos, "tso": tso,
                         "day_ret": day_ret})
    f6 = pd.concat([feats["F4"].reset_index(drop=True),
                    sess.reset_index(drop=True)], axis=1)
    f7 = pd.concat([feats["F2"].reset_index(drop=True),
                    sess.reset_index(drop=True)], axis=1)
    allf = dict(feats)
    allf["F6"], allf["F7"] = f6, f7
    _D6["v"] = (ts, allf, h, l, c)
    return _D6["v"]


def gproba6(g, train_before):
    key = f"p6_{g['feat']}_{g['model']}_{g['tgt']}_{g['stop']}_{g['H']}_{train_before}"
    f = CACHE / f"{key}.npy"
    if f.exists():
        return np.load(f)
    ts, allf, h, l, c = data6()
    X = allf[g["feat"]]
    y = glabel(g["tgt"], g["stop"], g["H"])
    fin = X.notna().all(axis=1).to_numpy()
    tr = np.where(fin & np.isfinite(y) & (ts < np.datetime64(train_before)))[0]
    tr = tr[:-g["H"]] if len(tr) > g["H"] else tr
    if len(tr) < 2000 or np.nansum(y[tr]) < 50:
        return None
    if g["model"] in ("lstm", "gru"):
        y2 = np.where(np.isfinite(y), y, 0.0)
        proba = _rnn_proba(g, X, y2, fin, tr, ts)
        if proba is None:
            return None
    else:
        clf = MODELS5[g["model"]]()
        clf.fit(X.iloc[tr], y[tr].astype(int))
        proba = np.full(len(X), np.nan)
        ok = np.where(fin)[0]
        proba[ok] = clf.predict_proba(X.iloc[ok])[:, 1]
    np.save(f, proba)
    return proba


def fitness(g):
    proba = gproba6(g, TRAIN_END)
    if proba is None:
        return -99.0, []
    subs = []
    for lo, hi in SUBS:
        r = sim(g, proba, lo, hi, TRAIN_END)
        if len(r) < 8:
            return -99.0, []
        subs.append(float(np.sum(r) * 100))
    return min(subs), subs


def rand_genome(rng):
    return {k: rng.choice(v) for k, v in GENES.items()}


def mutate(g, rng):
    c = dict(g)
    for k in GENES:
        if rng.random() < MUT:
            c[k] = rng.choice(GENES[k])
    return c


def crossover(a, b, rng):
    return {k: (a if rng.random() < 0.5 else b)[k] for k in GENES}


def init_state():
    groups = []
    for k in range(GROUPS):
        rng = random.Random(7000 + k)
        pop = []
        for a in range(POP):
            g = rand_genome(rng)
            if a < 5:                          # make sure session features get explored
                g["feat"] = "F6" if a % 2 == 0 else "F7"
            if a < 3:
                g["model"] = MODEL_GENE[(k + a) % len(MODEL_GENE)]
            pop.append(g)
        groups.append({"gen": 1, "pop": pop, "seen": [], "hist": []})
    return {"groups": groups, "scores": {}, "subs": {}}


def step():
    st = json.loads(STATE.read_text()) if STATE.exists() else init_state()
    if not STATE.exists():
        logline(f"== EVOLUTION-VI INIT: {GROUPS} islands x {POP} agents x {GENS} gens | "
                f"RTH-only + SESSION features (gap/premarket/prior-day/opening-range) ==")
        STATE.write_text(json.dumps(st))
    t0 = time.time()
    done = 0
    for k, grp in enumerate(st["groups"]):
        rng = random.Random(9000 + k * 131 + grp["gen"])
        while grp["gen"] <= GENS:
            for g in grp["pop"]:
                key = gkey(g)
                if key not in st["scores"]:
                    fit, subs = fitness(g)
                    st["scores"][key] = fit
                    st["subs"][key] = subs
                    logline(f"  G{k:02d} g{grp['gen']:02d} fit={fit:+7.2f}%  "
                            f"subs={[round(s, 2) for s in subs]}  {key}")
                    STATE.write_text(json.dumps(st))
                    if time.time() - t0 > 440:
                        logline("  [checkpoint — run `step` again]")
                        return
                if key not in grp["seen"]:
                    grp["seen"].append(key)
            ranked = sorted(grp["pop"], key=lambda g: -st["scores"][gkey(g)])
            best = ranked[0]
            grp["hist"].append({"gen": grp["gen"], "fit": st["scores"][gkey(best)],
                                "best": best})
            logline(f"== VI ISLAND {k:02d} GEN {grp['gen']} BEST "
                    f"fit={st['scores'][gkey(best)]:+.2f}% {gkey(best)} ==")
            elites = ranked[:ELITE]
            children, seen = [], {gkey(x) for x in elites}
            while len(children) < POP - ELITE - IMMI:
                ch = mutate(crossover(rng.choice(elites), rng.choice(elites), rng), rng)
                if gkey(ch) in seen:
                    ch = rand_genome(rng)
                seen.add(gkey(ch)); children.append(ch)
            grp["pop"] = elites + children + [rand_genome(rng) for _ in range(IMMI)]
            grp["gen"] += 1
            STATE.write_text(json.dumps(st))
        done += 1
    if done == GROUPS:
        logline("== EVOLUTION-VI COMPLETE — run `final` ==")


def final():
    st = json.loads(STATE.read_text())
    sc = st["scores"]
    taken, champs = set(), []
    for k, grp in enumerate(st["groups"]):
        ranked = sorted(grp["seen"], key=lambda key: -sc[key])
        champ = next((key for key in ranked if key not in taken and sc[key] > -99), None)
        if champ:
            taken.add(champ)
            champs.append((k, champ))
    print(f"=== EVOLUTION-VI: {len(champs)} unique island champions -> GATE ===")
    gated = []
    for k, key in champs:
        g = json.loads(key)
        proba = gproba6(g, GATE0)
        r = np.array(sim(g, proba, GATE0, GATE1, GATE0)) if proba is not None else np.array([])
        tot = float(r.sum() * 100) if len(r) >= 10 else -99.0
        print(f"  ISLAND {k:02d} arena={sc[key]:+.2f}% | gate: n={len(r)} "
              f"win={(r > 0).mean() if len(r) else 0:.0%} tot={tot:+.2f}%  {key}")
        gated.append((k, key, tot))
        sys.stdout.flush()
    survivors = [(k, key, t) for k, key, t in gated if t > 0]
    print(f"\n=== FINAL one-shot (gate-positive only: {len(survivors)}) ===")
    for k, key, t in survivors:
        g = json.loads(key)
        proba = gproba6(g, GATE1)
        r = np.array(sim(g, proba, GATE1, "2099-01-01", GATE1))
        if len(r) >= 10:
            tstat = r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else 0
            print(f"  ISLAND {k:02d}: n={len(r)} win={(r > 0).mean():.0%} "
                  f"avg={r.mean() * 1e4:+.1f}bp tot={r.sum() * 100:+.2f}% t={tstat:+.2f}")
        else:
            print(f"  ISLAND {k:02d}: n={len(r)} (too few)")
        print(f"    {key}")
        sys.stdout.flush()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "step"
    if mode == "step":
        step()
    else:
        final()
