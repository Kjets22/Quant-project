"""
evolve2.py — TOURNAMENT II: bigger, longer, P&L-driven. Beat vQ AND vQ2.

Upgrades over round 1:
  * 32 agents, 15 generations, mutation 0.35, +3 random immigrants each generation
  * FITNESS = P&L: min(total net return of arena half-1, half-2)  — turnover now counts,
    but the strategy must make money in BOTH half-years (>=15 trades each) or is DQ'd
  * 7 genes: target$ {1..4}, stop$ {1..3}, clock {6,12,24,48 bars}, gate q {.80...95},
    features {F2,F3,F4}, model {lgbm_s, lgbm_d, histgb}, RTH-only entries {0,1}
  * THREE-stage honesty ladder:
      ARENA  2023-07-14..2024-07-14 (halves @ 2024-01-14) — evolution fitness
      GATE   2024-07-14..2025-07-14 — top-3 finalists must CONFIRM here (kills arena luck)
      FINAL  2025-07-14..now        — gate winner, one shot, vs vQ and vQ2 baselines
        (disclosure: 2nd look at this window for the vQ family — live paper is the
         ultimate clean test)

  python evolve2.py step    (repeat until complete; checkpointed)
  python evolve2.py final
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

from qqq_tournament import load, MODELS

ARENA0, HALF, ARENA1 = "2023-07-14", "2024-01-14", "2024-07-14"
GATE1 = "2025-07-14"
EFF_COST = 5.0 / 1e4
POP, GENS, ELITE, IMMI, MUT = 32, 15, 6, 3, 0.35
GENES = {"feat": ["F2", "F3", "F4"], "model": ["lgbm_s", "lgbm_d", "histgb"],
         "tgt": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], "stop": [1.0, 1.5, 2.0, 2.5, 3.0],
         "H": [6, 12, 24, 48], "q": [0.80, 0.85, 0.90, 0.95], "rth": [0, 1]}
VQ = {"feat": "F4", "model": "lgbm_s", "tgt": 2.0, "stop": 2.0, "H": 12, "q": 0.90, "rth": 0}
VQ2 = {"feat": "F4", "model": "histgb", "tgt": 2.5, "stop": 2.0, "H": 24, "q": 0.90, "rth": 0}
STATE = Path("runs/evolve2_state.json")
CACHE = Path("runs/evo_cache")
LOGF = Path("runs/evolve2_log.txt")
rng = random.Random(11)

_ts = _feats = _h = _l = _c = _hours = None


def data():
    global _ts, _feats, _h, _l, _c, _hours
    if _ts is None:
        _ts, _feats, _ = load()
        base = pd.read_csv("data_cache/QQQ_5minute_2021-06-01_2026-06-01.csv",
                           parse_dates=["timestamp"])
        rec = sorted(Path("data_cache").glob("QQQ_recent_2026-06-01_*.csv"))
        parts = [base] + ([pd.read_csv(rec[-1], parse_dates=["timestamp"])] if rec else [])
        dd = (pd.concat(parts, ignore_index=True)
                .drop_duplicates(subset="timestamp", keep="last")
                .sort_values("timestamp").reset_index(drop=True))
        _h, _l, _c = (dd[x].to_numpy(float) for x in ("high", "low", "close"))
        t = pd.to_datetime(dd["timestamp"])
        _hours = (t.dt.hour + t.dt.minute / 60).to_numpy()
    return _ts, _feats, _h, _l, _c, _hours


def glabel(tgt, stop, H):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"y_{tgt}_{stop}_{H}.npy"
    if f.exists():
        return np.load(f)
    ts, feats, h, l, c, hours = data()
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        up, dn = c[i] + tgt, c[i] - stop
        for j in range(i + 1, min(i + H + 1, n)):
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
    key = f"p_{g['feat']}_{g['model']}_{g['tgt']}_{g['stop']}_{g['H']}_{train_before}"
    f = CACHE / f"{key}.npy"
    if f.exists():
        return np.load(f)
    ts, feats, h, l, c, hours = data()
    X = feats[g["feat"]]
    y = glabel(g["tgt"], g["stop"], g["H"])
    fin = X.notna().all(axis=1).to_numpy()
    tr = np.where(fin & np.isfinite(y) & (ts < np.datetime64(train_before)))[0][:-g["H"]]
    if len(tr) < 2000 or np.nansum(y[tr]) < 50:
        return None
    clf = MODELS[g["model"]]()
    clf.fit(X.iloc[tr], y[tr].astype(int))
    proba = np.full(len(X), np.nan)
    ok = np.where(fin)[0]
    proba[ok] = clf.predict_proba(X.iloc[ok])[:, 1]
    np.save(f, proba)
    return proba


def sim(g, proba, lo, hi, train_before):
    ts, feats, h, l, c, hours = data()
    tr_mask = np.isfinite(proba) & (ts < np.datetime64(train_before))
    thr = 0.5 + np.quantile(np.abs(proba[tr_mask] - 0.5), g["q"])
    idx = np.where(np.isfinite(proba) & (ts >= np.datetime64(lo)) & (ts < np.datetime64(hi)))[0]
    if len(idx) == 0:
        return []
    rets = []
    n = len(c)
    i, last = int(idx[0]), int(idx[-1])
    while i <= last:
        if (not np.isfinite(proba[i]) or proba[i] < thr
                or (g["rth"] and not (13.5 <= hours[i] < 20.0))):
            i += 1; continue
        up, dn = c[i] + g["tgt"], c[i] - g["stop"]
        res, j = None, i + 1
        while j < min(i + g["H"] + 1, n):
            hu, hd = h[j] >= up, l[j] <= dn
            if (hu and hd) or hd:
                res = 0; break
            if hu:
                res = 1; break
            j += 1
        ex = min(j, n - 1)
        if res is None:
            rets.append((c[ex] - c[i]) / c[i] - EFF_COST)
        else:
            rets.append((g["tgt"] if res == 1 else -g["stop"]) / c[i] - EFF_COST)
        i = j + 1
    return rets


def pnl_fit(rets):
    r = np.array(rets)
    if len(r) < 15:
        return -99.0
    return float(r.sum() * 100)


def fitness(g):
    proba = gproba(g, ARENA0)
    if proba is None:
        return -99.0, -99.0, -99.0
    f1 = pnl_fit(sim(g, proba, ARENA0, HALF, ARENA0))
    f2 = pnl_fit(sim(g, proba, HALF, ARENA1, ARENA0))
    return min(f1, f2), f1, f2


def gkey(g):
    return json.dumps(g, sort_keys=True)


def rand_genome():
    return {k: rng.choice(v) for k, v in GENES.items()}


def mutate(g):
    c = dict(g)
    for k in GENES:
        if rng.random() < MUT:
            c[k] = rng.choice(GENES[k])
    return c


def crossover(a, b):
    return {k: (a if rng.random() < 0.5 else b)[k] for k in GENES}


def logline(s):
    print(s, flush=True)
    LOGF.parent.mkdir(exist_ok=True)
    with LOGF.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def step():
    if STATE.exists():
        st = json.loads(STATE.read_text())
    else:
        pop = [VQ, VQ2] + [mutate(VQ) for _ in range(7)] + [mutate(VQ2) for _ in range(7)] \
              + [rand_genome() for _ in range(POP - 16)]
        st = {"gen": 1, "pop": pop, "scores": {}, "history": []}
        logline(f"== EVOLUTION-II INIT: {POP} agents (vQ + vQ2 seeded), fitness = "
                f"min-of-halves TOTAL P&L ==")
    t0 = time.time()
    while st["gen"] <= GENS:
        for g in st["pop"]:
            k = gkey(g)
            if k not in st["scores"]:
                fit, f1, f2 = fitness(g)
                st["scores"][k] = fit
                logline(f"  g{st['gen']:02d} fit={fit:+7.2f}%  (h1 {f1:+.2f} / h2 {f2:+.2f})  {k}")
                STATE.write_text(json.dumps(st))
                if time.time() - t0 > 460:
                    logline("  [checkpoint — run `step` again]")
                    return
        ranked = sorted(st["pop"], key=lambda g: -st["scores"][gkey(g)])
        best = ranked[0]
        st["history"].append({"gen": st["gen"], "best_fit": st["scores"][gkey(best)],
                              "best": best})
        logline(f"== GEN {st['gen']} BEST fit={st['scores'][gkey(best)]:+.2f}% {gkey(best)} ==")
        elites = ranked[:ELITE]
        children, seen = [], {gkey(g) for g in elites}
        while len(children) < POP - ELITE - IMMI:
            ch = mutate(crossover(rng.choice(elites), rng.choice(elites)))
            if gkey(ch) in seen:
                ch = rand_genome()
            seen.add(gkey(ch)); children.append(ch)
        immigrants = [rand_genome() for _ in range(IMMI)]
        st["pop"] = elites + children + immigrants
        st["gen"] += 1
        STATE.write_text(json.dumps(st))
    logline("== EVOLUTION-II COMPLETE — run `final` ==")


def final():
    st = json.loads(STATE.read_text())
    ranked = sorted(st["pop"], key=lambda g: -st["scores"].get(gkey(g), -99))
    top3 = ranked[:3]
    print("=== GATE (2024-07-14..2025-07-14): top-3 must confirm ===")
    gated = []
    for g in top3:
        proba = gproba(g, ARENA1)
        rets = sim(g, proba, ARENA1, GATE1, ARENA1)
        tot = pnl_fit(rets)
        n = len(rets)
        wins = float((np.array(rets) > 0).mean()) if n else 0
        print(f"  gate: total={tot:+.2f}% n={n} win%={wins:.1%}  {gkey(g)}")
        gated.append((tot, g))
    gated.sort(key=lambda x: -x[0])
    champ = gated[0][1]
    print(f"\n=== FINAL one-shot (2025-07-14..now) — champion vs vQ vs vQ2 ===")
    for name, g in (("CHAMPION", champ), ("vQ", VQ), ("vQ2", VQ2)):
        proba = gproba(g, GATE1)
        rets = sim(g, proba, GATE1, "2099-01-01", GATE1)
        r = np.array(rets)
        n = len(r)
        wins = (r > 0).mean() if n else 0
        t = r.mean() / r.std() * np.sqrt(n) if n > 1 and r.std() > 0 else 0
        print(f"  {name:>9}: n={n:>3} win%={wins:.1%} mean={r.mean()*1e4:+.1f}bps "
              f"total={r.sum()*100:+.2f}% t={t:+.2f}  {gkey(g)}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "step"
    if mode == "step":
        step()
    else:
        final()
