"""
evolve3.py — TOURNAMENT III: maximize P&L, robustly. Beat vQ (+2.77% final year).

Lesson from Tournament II encoded: raw P&L fitness overfits (its champions died at the
gate). So: FITNESS = the WORST total net P&L across THREE half-year sub-windows
(2023-01..2023-07..2024-01..2024-07), >=12 trades in each, trained strictly before.
Profit must be repeatable across regimes or the genome is worthless.

Search space (everything the user asked to vary):
  * models + hyperparameters: 5 presets (lgbm tiny/standard/big, histgb std/deep)
  * features: F2/F3/F4 + NEW F5 = F4 + QQQ-vs-SPY cross-signals (relative strength)
  * geometry: target $1-4, stop $1-3, clock 6-96 bars, gate q .75-.97, RTH on/off
Seeds: vQ, vQ2, vA lineages. 32 agents x 12 generations, mutation .35, 3 immigrants/gen.
Ladder: arena fitness -> GATE 2024-07..2025-07 (top-3 confirm) -> FINAL 2025-07..now,
one shot vs vQ, vQ2, vA. (Disclosure: 3rd look at the final window for this family —
live paper remains the ultimate referee.)

  python evolve3.py step   (repeat until complete)   |   python evolve3.py final
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
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier

from qqq_tournament import load

TRAIN_END = "2023-01-14"
SUBS = [("2023-01-14", "2023-07-14"), ("2023-07-14", "2024-01-14"),
        ("2024-01-14", "2024-07-14")]
GATE0, GATE1 = "2024-07-14", "2025-07-14"
EFF_COST = 5.0 / 1e4
POP, GENS, ELITE, IMMI, MUT = 32, 12, 6, 3, 0.35

MODELS3 = {
    "lgbm_tiny": lambda: lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                            num_leaves=7, min_child_samples=100,
                                            subsample=0.8, colsample_bytree=0.8,
                                            reg_lambda=2.0, verbose=-1),
    "lgbm_s":    lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03,
                                            num_leaves=15, min_child_samples=40,
                                            subsample=0.8, colsample_bytree=0.8,
                                            reg_lambda=1.0, verbose=-1),
    "lgbm_big":  lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                            num_leaves=63, min_child_samples=100,
                                            subsample=0.7, colsample_bytree=0.7,
                                            reg_lambda=3.0, verbose=-1),
    "histgb":    lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                        max_leaf_nodes=31,
                                                        min_samples_leaf=50),
    "histgb_d":  lambda: HistGradientBoostingClassifier(max_iter=500, learning_rate=0.04,
                                                        max_leaf_nodes=63,
                                                        min_samples_leaf=100),
}
GENES = {"feat": ["F2", "F3", "F4", "F5"], "model": list(MODELS3) + ["lstm", "gru"],
         "tgt": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], "stop": [1.0, 1.5, 2.0, 2.5, 3.0],
         "H": [6, 12, 24, 48, 96], "q": [0.75, 0.80, 0.85, 0.90, 0.95, 0.97],
         "rth": [0, 1]}
SEQ = 24                                              # RNN lookback: 24 x 5-min = 2 hours
VQ = {"feat": "F4", "model": "lgbm_s", "tgt": 2.0, "stop": 2.0, "H": 12, "q": 0.90, "rth": 0}
VQ2 = {"feat": "F4", "model": "histgb", "tgt": 2.5, "stop": 2.0, "H": 24, "q": 0.90, "rth": 0}
VA = {"feat": "F4", "model": "lgbm_s", "tgt": 1.5, "stop": 2.0, "H": 48, "q": 0.95, "rth": 0}
STATE = Path("runs/evolve3_state.json")
CACHE = Path("runs/evo_cache")
LOGF = Path("runs/evolve3_log.txt")
rng = random.Random(23)

_ts = _feats = _h = _l = _c = _hours = None


def data():
    global _ts, _feats, _h, _l, _c, _hours
    if _ts is None:
        _ts, feats, _ = load()
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
        # F5 = F4 + QQQ-vs-SPY cross-signals (new information source)
        sb = pd.read_csv("data_cache/SPY_5minute_2021-06-01_2026-06-01.csv",
                         parse_dates=["timestamp"])
        srec = sorted(Path("data_cache").glob("SPY_recent_2026-06-01_*.csv"))
        sparts = [sb] + ([pd.read_csv(srec[-1], parse_dates=["timestamp"])] if srec else [])
        sd = (pd.concat(sparts, ignore_index=True)
                .drop_duplicates(subset="timestamp", keep="last").sort_values("timestamp"))
        spy = sd.set_index("timestamp")["close"].reindex(t, method="ffill").to_numpy()
        qr = np.diff(np.log(_c), prepend=np.log(_c[0]))
        sr = np.diff(np.log(np.where(spy > 0, spy, np.nan)), prepend=0.0)
        x5 = pd.DataFrame({
            "s_r1": sr, "s_r6": pd.Series(sr).rolling(6).sum().to_numpy(),
            "s_r24": pd.Series(sr).rolling(24).sum().to_numpy(),
            "rel1": qr - sr,
            "rel6": pd.Series(qr - sr).rolling(6).sum().to_numpy(),
            "rel24": pd.Series(qr - sr).rolling(24).sum().to_numpy(),
        })
        feats = dict(feats)
        feats["F5"] = pd.concat([feats["F4"], x5.reset_index(drop=True)], axis=1)
        _feats = feats
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


def _rnn_proba(g, X, y, fin, tr, ts):
    """Small LSTM/GRU: standardized features, SEQ-bar sequences, 6 epochs, CPU."""
    import torch
    import torch.nn as nn
    torch.manual_seed(7)
    Xn = X.to_numpy(np.float32)
    mu = np.nanmean(Xn[tr], axis=0)
    sd = np.nanstd(Xn[tr], axis=0) + 1e-9
    Z = np.clip((Xn - mu) / sd, -5, 5)
    Z[~np.isfinite(Z)] = 0.0
    ok_row = fin.copy()
    tr_i = np.array([i for i in tr if i >= SEQ])[-60000:]      # recent 60k train samples
    if len(tr_i) < 5000:
        return None
    F = Z.shape[1]
    rnn_cls = nn.LSTM if g["model"] == "lstm" else nn.GRU
    net = nn.Sequential()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn_cls(F, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(o[:, -1, :]).squeeze(-1)

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    yv = y.astype(np.float32)

    def batch_seq(idx):
        return torch.from_numpy(np.stack([Z[i - SEQ + 1:i + 1] for i in idx]))

    for ep in range(6):
        perm = np.random.default_rng(ep).permutation(tr_i)
        for b in range(0, len(perm), 1024):
            bi = perm[b:b + 1024]
            opt.zero_grad()
            out = net(batch_seq(bi))
            loss = lossf(out, torch.from_numpy(yv[bi]))
            loss.backward()
            opt.step()
    net.eval()
    proba = np.full(len(Z), np.nan)
    pred_i = np.where(ok_row)[0]
    pred_i = pred_i[pred_i >= SEQ]
    with torch.no_grad():
        for b in range(0, len(pred_i), 2048):
            bi = pred_i[b:b + 2048]
            proba[bi] = torch.sigmoid(net(batch_seq(bi))).numpy()
    return proba


def gproba(g, train_before):
    key = f"p3_{g['feat']}_{g['model']}_{g['tgt']}_{g['stop']}_{g['H']}_{train_before}"
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


def fitness(g):
    """Worst total net P&L across the three arena sub-windows (>=12 trades each)."""
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
        # inject RNN challengers once (user request): 4 seeded lstm/gru variants
        if not any(g["model"] in ("lstm", "gru") for g in st["pop"]):
            inj = [dict(VQ, model="lstm"), dict(VQ2, model="lstm"),
                   dict(VQ, model="gru"), dict(VA, model="gru")]
            st["pop"] = st["pop"][:-len(inj)] + inj
            logline("  [injected 4 LSTM/GRU challengers into the population]")
    else:
        pop = ([VQ, VQ2, VA] + [mutate(VQ) for _ in range(5)] + [mutate(VQ2) for _ in range(5)]
               + [mutate(VA) for _ in range(5)] + [rand_genome() for _ in range(POP - 18)])
        st = {"gen": 1, "pop": pop, "scores": {}, "history": []}
        logline(f"== EVOLUTION-III INIT: {POP} agents (vQ, vQ2, vA seeded); fitness = "
                f"WORST of 3 half-year P&Ls ==")
    t0 = time.time()
    while st["gen"] <= GENS:
        for g in st["pop"]:
            k = gkey(g)
            if k not in st["scores"]:
                fit, subs = fitness(g)
                st["scores"][k] = fit
                logline(f"  g{st['gen']:02d} fit={fit:+7.2f}%  subs={[round(s,2) for s in subs]}  {k}")
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
    logline("== EVOLUTION-III COMPLETE — run `final` ==")


def final():
    st = json.loads(STATE.read_text())
    sc = st["scores"]
    allg = [json.loads(k) for k in sc]
    top3 = sorted(allg, key=lambda g: -sc[gkey(g)])[:3]
    print("=== GATE (2024-07-14..2025-07-14): top-3 by robust P&L must confirm ===")
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
    print(f"\n=== FINAL one-shot (2025-07-14..now) — champion vs vQ / vQ2 / vA ===")
    for name, g in (("CHAMPION", champ), ("vQ", VQ), ("vQ2", VQ2), ("vA", VA)):
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
