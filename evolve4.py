"""
evolve4.py — TOURNAMENT IV: evolve PERCENTAGE-barrier space around the user's vR spec.
The incumbent to beat: vR (+0.4%/-0.2%, 2h, lgbm_s, top-3% gate) — final-year +7.00%.

Genes: target% {0.2..0.8}, stop% {0.1..0.4}, clock {30min..8h}, top-quantile gate,
features {F2..F5}, models {5 tree presets + lstm + gru}, RTH filter. 32 agents, 12 gens,
mutation .35, 3 immigrants/gen. Fitness = WORST of three half-year P&Ls (evolve3's proven
anti-overfit recipe). Ladder: arena 2023-24 -> gate 2024-25 -> one-shot final vs vR.

  python evolve4.py step   (repeat until complete)   |   python evolve4.py final
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

from evolve3 import (data, MODELS3, _rnn_proba, SUBS, TRAIN_END, GATE0, GATE1,
                     EFF_COST, CACHE)

POP, GENS, ELITE, IMMI, MUT = 32, 12, 6, 3, 0.35
GENES = {"feat": ["F2", "F3", "F4", "F5"],
         "model": list(MODELS3) + ["lstm", "gru"],
         "tgt": [0.002, 0.003, 0.004, 0.005, 0.006, 0.008],
         "stop": [0.001, 0.0015, 0.002, 0.0025, 0.003, 0.004],
         "H": [6, 12, 24, 48, 96],
         "q": [0.90, 0.93, 0.95, 0.97],
         "rth": [0, 1]}
VR = {"feat": "F4", "model": "lgbm_s", "tgt": 0.004, "stop": 0.002, "H": 24,
      "q": 0.97, "rth": 0}
STATE = Path("runs/evolve4_state.json")
LOGF = Path("runs/evolve4_log.txt")
rng = random.Random(31)


def glabel(tgt, stop, H):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"y_pct4_{tgt}_{stop}_{H}.npy"
    if f.exists():
        return np.load(f)
    ts, feats, h, l, c, hours = data()
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        up, dn = c[i] * (1 + tgt), c[i] * (1 - stop)
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
    key = f"p4_{g['feat']}_{g['model']}_{g['tgt']}_{g['stop']}_{g['H']}_{train_before}"
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
    if g["model"] in ("lstm", "gru"):
        y2 = np.where(np.isfinite(y), y, 0.0)
        proba = _rnn_proba(g, X, y2, fin, tr, ts)
        if proba is None:
            return None
    else:
        clf = MODELS3[g["model"]]()
        clf.fit(X.iloc[tr], y[tr].astype(int))
        proba = np.full(len(X), np.nan)
        ok = np.where(fin)[0]
        proba[ok] = clf.predict_proba(X.iloc[ok])[:, 1]
    np.save(f, proba)
    return proba


def sim(g, proba, lo, hi, train_before):
    """Top-quantile gate (correct for asymmetric labels), percentage barriers."""
    ts, feats, h, l, c, hours = data()
    tr_mask = np.isfinite(proba) & (ts < np.datetime64(train_before))
    thr = np.quantile(proba[tr_mask], g["q"])
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
        up, dn = c[i] * (1 + g["tgt"]), c[i] * (1 - g["stop"])
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
            rets.append((g["tgt"] if res == 1 else -g["stop"]) - EFF_COST)
        i = j + 1
    return rets


def fitness(g):
    proba = gproba(g, TRAIN_END)
    if proba is None:
        return -99.0, []
    subs = []
    for lo, hi in SUBS:
        r = sim(g, proba, lo, hi, TRAIN_END)
        if len(r) < 12:
            return -99.0, []
        subs.append(float(np.sum(r) * 100))
    return min(subs), subs


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
        pop = [VR] + [mutate(VR) for _ in range(13)] + [rand_genome() for _ in range(POP - 14)]
        st = {"gen": 1, "pop": pop, "scores": {}, "history": []}
        logline(f"== EVOLUTION-IV INIT: {POP} agents (vR seeded); percentage-barrier space; "
                f"fitness = WORST of 3 half-year P&Ls ==")
    t0 = time.time()
    while st["gen"] <= GENS:
        for g in st["pop"]:
            k = gkey(g)
            if k not in st["scores"]:
                fit, subs = fitness(g)
                st["scores"][k] = fit
                logline(f"  g{st['gen']:02d} fit={fit:+7.2f}%  "
                        f"subs={[round(s, 2) for s in subs]}  {k}")
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
        st["pop"] = elites + children + [rand_genome() for _ in range(IMMI)]
        st["gen"] += 1
        STATE.write_text(json.dumps(st))
    logline("== EVOLUTION-IV COMPLETE — run `final` ==")


def final():
    st = json.loads(STATE.read_text())
    sc = st["scores"]
    allg = [json.loads(k) for k in sc]
    top3 = sorted(allg, key=lambda g: -sc[gkey(g)])[:3]
    print("=== GATE (2024-07-14..2025-07-14): top-3 must confirm ===")
    gated = []
    for g in top3:
        proba = gproba(g, GATE0)
        rets = sim(g, proba, GATE0, GATE1, GATE0)
        r = np.array(rets)
        tot = float(r.sum() * 100) if len(r) >= 15 else -99
        print(f"  gate: total={tot:+.2f}% n={len(r)} "
              f"win%={(r > 0).mean() if len(r) else 0:.1%}  {gkey(g)}")
        gated.append((tot, g))
    gated.sort(key=lambda x: -x[0])
    champ = gated[0][1]
    print(f"\n=== FINAL one-shot (2025-07-14..now) — champion vs vR (incumbent) ===")
    for name, g in (("CHAMPION", champ), ("vR", VR)):
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
