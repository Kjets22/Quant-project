"""
evolve_vq.py — EVOLUTIONARY TOURNAMENT to beat vQ. 20 agents, 10 generations.

Genome (what agents mutate): feature set (F3/F4), model (lgbm_s/histgb), target $,
stop $, clock H (bars), confidence gate q. Long-only (shorts failed twice — excluded).
Learning-from-each-other: top-5 elites survive; 15 children bred by CROSSOVER of two
elite parents + 25% mutation per gene.

ANTI-OVERFITTING (non-negotiable):
  * TRAIN  < 2024-07-14 (model fitting only)
  * ARENA  2024-07-14 .. 2025-07-14 — fitness window, never used in any prior vQ decision
  * FITNESS = min(t-stat of half-1, t-stat of half-2) — must work in BOTH half-years;
    n >= 15 trades per half or disqualified
  * FINAL  2025-07-14 .. now — champion gets ONE shot, head-to-head vs current vQ
    (retrained on < 2025-07-14); adopt only if clearly better.

Checkpointed: run `python evolve_vq.py step` repeatedly until gen 10; `final` for the shot.
Caches label arrays and model probabilities to runs/evo_cache/ so restarts are cheap.
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

ARENA0, ARENA1, HALF = "2024-07-14", "2025-07-14", "2025-01-14"
EFF_COST = 5.0 / 1e4
POP, GENS, ELITE = 20, 10, 5
GENES = {"feat": ["F3", "F4"], "model": ["lgbm_s", "histgb"],
         "tgt": [1.5, 2.0, 2.5, 3.0], "stop": [1.5, 2.0, 2.5, 3.0],
         "H": [6, 12, 24], "q": [0.85, 0.90, 0.95]}
VQ = {"feat": "F4", "model": "lgbm_s", "tgt": 2.0, "stop": 2.0, "H": 12, "q": 0.90}
STATE = Path("runs/evolve_state.json")
CACHE = Path("runs/evo_cache")
LOGF = Path("runs/evolve_log.txt")
rng = random.Random(7)

_ts = _feats = None
_h = _l = _c = None


def data():
    global _ts, _feats, _h, _l, _c
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
    return _ts, _feats, _h, _l, _c


def glabel(tgt, stop, H):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"y_{tgt}_{stop}_{H}.npy"
    if f.exists():
        return np.load(f)
    ts, feats, h, l, c = data()
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
    """Full-series P(up) for genome's (feat, model, label), trained < train_before."""
    key = f"p_{g['feat']}_{g['model']}_{g['tgt']}_{g['stop']}_{g['H']}_{train_before}"
    f = CACHE / f"{key}.npy"
    if f.exists():
        return np.load(f)
    ts, feats, h, l, c = data()
    X = feats[g["feat"]]
    y = glabel(g["tgt"], g["stop"], g["H"])
    fin = X.notna().all(axis=1).to_numpy()
    tr = np.where(fin & np.isfinite(y) & (ts < np.datetime64(train_before)))[0][:-g["H"]]
    clf = MODELS[g["model"]]()
    clf.fit(X.iloc[tr], y[tr].astype(int))
    proba = np.full(len(X), np.nan)
    ok = np.where(fin)[0]
    proba[ok] = clf.predict_proba(X.iloc[ok])[:, 1]
    np.save(f, proba)
    return proba


def sim(g, proba, lo, hi, train_before):
    """Long-only non-overlapping trades in [lo, hi). Returns list of net returns."""
    ts, feats, h, l, c = data()
    tr_mask = np.isfinite(proba) & (ts < np.datetime64(train_before))
    thr = 0.5 + np.quantile(np.abs(proba[tr_mask] - 0.5), g["q"])
    idx = np.where(np.isfinite(proba) & (ts >= np.datetime64(lo)) & (ts < np.datetime64(hi)))[0]
    if len(idx) == 0:
        return []
    rets = []
    n = len(c)
    i, last = int(idx[0]), int(idx[-1])
    while i <= last:
        if not np.isfinite(proba[i]) or proba[i] < thr:
            i += 1; continue
        up, dn = c[i] + g["tgt"], c[i] - g["stop"]
        res, j = None, i + 1
        while j < min(i + g["H"] + 1, n):
            hu, hd = h[j] >= up, l[j] <= dn
            if hu and hd:
                res = 0; break                          # ambiguous -> against us
            if hd:
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


def tstat(rets):
    r = np.array(rets)
    if len(r) < 15 or r.std() == 0:
        return -99.0
    return float(r.mean() / r.std() * np.sqrt(len(r)))


def fitness(g):
    proba = gproba(g, ARENA0)
    t1 = tstat(sim(g, proba, ARENA0, HALF, ARENA0))
    t2 = tstat(sim(g, proba, HALF, ARENA1, ARENA0))
    return min(t1, t2), t1, t2


def gkey(g):
    return json.dumps(g, sort_keys=True)


def rand_genome():
    return {k: rng.choice(v) for k, v in GENES.items()}


def mutate(g):
    c = dict(g)
    for k in GENES:
        if rng.random() < 0.25:
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
        pop = [VQ] + [mutate(VQ) for _ in range(9)] + [rand_genome() for _ in range(10)]
        st = {"gen": 1, "pop": pop, "scores": {}, "history": []}
        logline(f"== EVOLUTION INIT: {POP} agents (vQ seeded) ==")
    t0 = time.time()
    while st["gen"] <= GENS:
        pop = st["pop"]
        # evaluate any unscored genomes (resumable)
        for g in pop:
            k = gkey(g)
            if k not in st["scores"]:
                fit, t1, t2 = fitness(g)
                st["scores"][k] = fit
                logline(f"  g{st['gen']:02d} eval fit={fit:+.2f} (h1 {t1:+.2f}/h2 {t2:+.2f}) {k}")
                STATE.write_text(json.dumps(st))
                if time.time() - t0 > 480:
                    logline("  [checkpoint: time budget — run `step` again to continue]")
                    return
        ranked = sorted(pop, key=lambda g: -st["scores"][gkey(g)])
        best = ranked[0]
        st["history"].append({"gen": st["gen"], "best_fit": st["scores"][gkey(best)],
                              "best": best})
        logline(f"== GEN {st['gen']} BEST fit={st['scores'][gkey(best)]:+.2f} {gkey(best)} ==")
        elites = ranked[:ELITE]
        children, seen = [], {gkey(g) for g in elites}
        while len(children) < POP - ELITE:
            child = mutate(crossover(rng.choice(elites), rng.choice(elites)))
            if gkey(child) in seen:
                child = rand_genome()
            seen.add(gkey(child))
            children.append(child)
        st["pop"] = elites + children
        st["gen"] += 1
        STATE.write_text(json.dumps(st))
    logline("== EVOLUTION COMPLETE (10 generations) — run `final` ==")


def final():
    st = json.loads(STATE.read_text())
    champ = sorted(st["history"], key=lambda x: -x["best_fit"])[-0]["best"] \
        if not st["history"] else st["history"][-1]["best"]
    # champion = best of last generation; compare vs vQ, both retrained < ARENA1
    print(f"FINAL one-shot ({ARENA1}..now) — champion vs current vQ")
    for name, g in (("CHAMPION", champ), ("vQ (baseline)", VQ)):
        proba = gproba(g, ARENA1)
        rets = sim(g, proba, ARENA1, "2099-01-01", ARENA1)
        r = np.array(rets)
        wins = (r > 0).mean() if len(r) else 0
        print(f"  {name:>14}: {gkey(g)}")
        print(f"  {'':>14}  trades={len(r)}  win%={wins:.1%}  mean={r.mean()*1e4:+.1f}bps  "
              f"total={r.sum()*100:+.2f}%  t-stat={tstat(rets):+.2f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "step"
    if mode == "step":
        step()
    elif mode == "final":
        final()
