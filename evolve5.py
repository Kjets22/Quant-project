"""
evolve5.py — TOURNAMENT V: the RTH day-trading islands. 100 agents, 10 ISOLATED groups.

The mission (user spec): best P&L on QQQ with
  - entries ONLY while the market is open (9:30-16:00 ET, so options could trade it),
  - max hold 2-4 hours, and NEVER holding into the next day (forced close by 16:00),
  - the most in-depth gene space yet: features F2-F5, 11 model families (tree presets +
    deep/slow hyperparameter variants + RF + logreg + LSTM/GRU), pct targets 0.15-0.8%,
    stops 0.1-0.5%, clocks {2h,3h,4h}, two gate types x five selectivities, and
    morning/afternoon/full entry windows.

Structure: 10 groups x 10 agents x 10 generations = 100 group-tournaments. Groups are
ISLANDS — no migration, no interaction; each is seeded toward a different model family
so the 10 champions diverge. Fitness = WORST of three half-year P&Ls (2023-2024 arena).
After evolution: every group champion (forced unique) runs the gate year (2024-25) and
the one-shot final year (2025-now).

  python evolve5.py step    (repeat until complete)    |    python evolve5.py final
"""

from __future__ import annotations

import json
import random
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
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evolve3 import (data, MODELS3, _rnn_proba, SUBS, TRAIN_END, GATE0, GATE1,
                     EFF_COST, CACHE)

GROUPS, POP, GENS, ELITE, IMMI, MUT = 10, 10, 10, 3, 1, 0.40

MODELS5 = dict(MODELS3)
MODELS5["lgbm_deep"] = lambda: lgb.LGBMClassifier(
    n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
MODELS5["lgbm_slow"] = lambda: lgb.LGBMClassifier(
    n_estimators=800, learning_rate=0.01, num_leaves=15, min_child_samples=40,
    subsample=0.7, colsample_bytree=0.7, reg_lambda=2.0, verbose=-1)
MODELS5["rf"] = lambda: RandomForestClassifier(
    n_estimators=300, min_samples_leaf=50, n_jobs=-1, random_state=0)
MODELS5["logreg"] = lambda: make_pipeline(
    StandardScaler(), LogisticRegression(max_iter=500, C=0.5))
MODEL_GENE = list(MODELS5) + ["lstm", "gru"]

GENES = {"feat": ["F2", "F3", "F4", "F5"],
         "model": MODEL_GENE,
         "tgt": [0.0015, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008],
         "stop": [0.001, 0.0015, 0.002, 0.003, 0.004, 0.005],
         "H": [24, 36, 48],
         "gate": ["q", "conf"],
         "qv": [0.85, 0.90, 0.93, 0.95, 0.97],
         "win": ["full", "am", "pm"]}
WINDOWS = {"full": (570, 930), "am": (570, 750), "pm": (750, 930)}
STATE = Path("runs/evolve5_state.json")
LOGF = Path("runs/evolve5_log.txt")

_ET = ZoneInfo("America/New_York")
_INFO = {}


def info():
    """RTH mask, minute-of-day, and last-RTH-bar-of-day index per bar."""
    if _INFO:
        return _INFO["v"]
    ts, feats, h, l, c, hours = data()
    idx = pd.DatetimeIndex(ts).tz_localize("UTC").tz_convert(_ET)
    m = np.asarray(idx.hour * 60 + idx.minute)
    rth = (m >= 570) & (m < 960)
    dates = np.asarray(idx.date)
    last_rth = np.full(len(m), -1, dtype=np.int64)
    cur_day, cur_last = None, -1
    for i in range(len(m) - 1, -1, -1):
        if dates[i] != cur_day:
            cur_day, cur_last = dates[i], -1
        if rth[i] and cur_last == -1:
            cur_last = i
        last_rth[i] = cur_last
    _INFO["v"] = (rth, m, last_rth)
    return _INFO["v"]


def glabel(tgt, stop, H):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"y5rth_{tgt}_{stop}_{H}.npy"
    if f.exists():
        return np.load(f)
    ts, feats, h, l, c, hours = data()
    rth, m, last_rth = info()
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        if not (rth[i] and 570 <= m[i] < 930):
            continue
        jmax = min(i + H, last_rth[i])
        if jmax <= i:
            continue
        up, dn = c[i] * (1 + tgt), c[i] * (1 - stop)
        for j in range(i + 1, jmax + 1):
            hu, hd = h[j] >= up, l[j] <= dn
            if hu and hd:
                break
            if hd:
                y[i] = 0; break
            if hu:
                y[i] = 1; break
    np.save(f, y)
    return y


def gproba(g, train_before):
    key = f"p5_{g['feat']}_{g['model']}_{g['tgt']}_{g['stop']}_{g['H']}_{train_before}"
    f = CACHE / f"{key}.npy"
    if f.exists():
        return np.load(f)
    ts, feats, h, l, c, hours = data()
    X = feats[g["feat"]]
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


def sim(g, proba, lo, hi, train_before):
    """RTH-only entries, day-bounded exits (never holds overnight)."""
    ts, feats, h, l, c, hours = data()
    rth, m, last_rth = info()
    w0, w1 = WINDOWS[g["win"]]
    elig = np.isfinite(proba) & rth & (m >= w0) & (m < w1)
    tr_mask = elig & (ts < np.datetime64(train_before))
    if tr_mask.sum() < 500:
        return []
    if g["gate"] == "conf":
        thr = 0.5 + np.quantile(np.abs(proba[tr_mask] - 0.5), g["qv"])
    else:
        thr = np.quantile(proba[tr_mask], g["qv"])
    idx = np.where(elig & (ts >= np.datetime64(lo)) & (ts < np.datetime64(hi)))[0]
    if len(idx) == 0:
        return []
    rets = []
    i, last = int(idx[0]), int(idx[-1])
    n = len(c)
    while i <= last:
        p = proba[i]
        hit = (elig[i] and np.isfinite(p)
               and (abs(p - 0.5) + 0.5 >= thr if g["gate"] == "conf" else p >= thr))
        if not hit:
            i += 1; continue
        jmax = min(i + g["H"], last_rth[i], n - 1)
        if jmax <= i:
            i += 1; continue
        up, dn = c[i] * (1 + g["tgt"]), c[i] * (1 - g["stop"])
        res, j = None, i + 1
        while j <= jmax:
            hu, hd = h[j] >= up, l[j] <= dn
            if (hu and hd) or hd:
                res = 0; break
            if hu:
                res = 1; break
            j += 1
        ex = min(j, jmax)
        if res is None:
            rets.append((c[ex] - c[i]) / c[i] - EFF_COST)   # day/time exit, flat by close
        else:
            rets.append((g["tgt"] if res == 1 else -g["stop"]) - EFF_COST)
        i = ex + 1
    return rets


def fitness(g):
    proba = gproba(g, TRAIN_END)
    if proba is None:
        return -99.0, []
    subs = []
    for lo, hi in SUBS:
        r = sim(g, proba, lo, hi, TRAIN_END)
        if len(r) < 8:
            return -99.0, []
        subs.append(float(np.sum(r) * 100))
    return min(subs), subs


def gkey(g):
    return json.dumps(g, sort_keys=True)


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


def logline(s):
    print(s, flush=True)
    LOGF.parent.mkdir(exist_ok=True)
    with LOGF.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def init_state():
    groups = []
    for k in range(GROUPS):
        rng = random.Random(1000 + k)
        bias = MODEL_GENE[k % len(MODEL_GENE)]
        pop = []
        for a in range(POP):
            g = rand_genome(rng)
            if a < 3:
                g["model"] = bias          # island specialty seed
            pop.append(g)
        groups.append({"gen": 1, "pop": pop, "seen": [], "hist": [], "bias": bias})
    return {"groups": groups, "scores": {}, "subs": {}}


def step():
    st = json.loads(STATE.read_text()) if STATE.exists() else init_state()
    if not STATE.exists():
        logline(f"== EVOLUTION-V INIT: {GROUPS} isolated islands x {POP} agents x "
                f"{GENS} gens | RTH-only, day-bounded, 2-4h clocks ==")
        STATE.write_text(json.dumps(st))
    t0 = time.time()
    done = 0
    for k, grp in enumerate(st["groups"]):
        rng = random.Random(5000 + k * 97 + grp["gen"])
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
            logline(f"== ISLAND {k:02d} (bias {grp['bias']}) GEN {grp['gen']} BEST "
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
        logline("== EVOLUTION-V COMPLETE (all 100 group-tournaments) — run `final` ==")


def final():
    st = json.loads(STATE.read_text())
    sc = st["scores"]
    taken, champs = set(), []
    for k, grp in enumerate(st["groups"]):
        ranked = sorted(grp["seen"], key=lambda key: -sc[key])
        champ = next((key for key in ranked if key not in taken and sc[key] > -99), None)
        if champ is None:
            continue
        taken.add(champ)
        champs.append((k, grp["bias"], champ))
    print(f"=== EVOLUTION-V FINAL: {len(champs)} unique island champions ===")
    print("(gate = 2024-07..2025-07 confirmation; FINAL = untouched 2025-07..now)\n")
    for k, bias, key in champs:
        g = json.loads(key)
        row = f"ISLAND {k:02d} (bias {bias}) arena={sc[key]:+.2f}%"
        for tag, tb, lo, hi in (("gate", GATE0, GATE0, GATE1),
                                ("FINAL", GATE1, GATE1, "2099-01-01")):
            proba = gproba(g, tb)
            r = np.array(sim(g, proba, lo, hi, tb)) if proba is not None else np.array([])
            if len(r) >= 10:
                row += (f" | {tag}: n={len(r)} win={(r > 0).mean():.0%} "
                        f"avg={r.mean() * 1e4:+.1f}bp tot={r.sum() * 100:+.2f}%")
            else:
                row += f" | {tag}: n={len(r)} (too few)"
        print(row)
        print(f"    {key}")
        sys.stdout.flush()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "step"
    if mode == "step":
        step()
    else:
        final()
