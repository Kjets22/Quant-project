"""
vx_lab.py — the standing IMPROVEMENT LOOP for the two new bots (user goal:
iterate until backtested Sharpe >= 3).

Search space
  vPT (geometric patterns): timeframe {5,15,30,60} x pivot k {1.5,2,3} x
      tolerance {0.2%,0.3%,0.5%} x hold {24,48,96} x sides {both,long,short}
      x pattern-universe (all detectors in vpt_geometric) x tickers (8-basket)
  vOB (order book): timeframe {5,15,30} x bracket {tgt,stop} grid x H x gate q
      x feature set {book, book+price, price-only control}

Honesty protocol
  OPT window  : patterns 2023-01..2024-07 | vob: first 60% of its backfill
  VAL window  : patterns 2024-07..2025-07 | vob: next 20%
  TEST window : patterns 2025-07..now    | vob: last 20% — only champions
                (VAL Sharpe >= 2) ever touch TEST, and each config touches it ONCE.
  Sharpe = daily-P&L Sharpe, annualized sqrt(252), >= 40 trades required.

State: runs/vx_lab_state.json (leaderboard, tried-set, test-touch log).
Each invocation runs ~N configs then checkpoints (chain like the tournaments).

  python vx_lab.py step [N]     |     python vx_lab.py board
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

from triple_barrier_breadth import TICKERS
import vpt_geometric as G

STATE = Path("runs/vx_lab_state.json")
LOGF = Path("runs/vx_lab_log.txt")
VOB = Path("data_cache/vob")

P_OPT = ("2023-01-01", "2024-07-01")
P_VAL = ("2024-07-01", "2025-07-01")
P_TEST = ("2025-07-01", "2099-01-01")

VPT_SPACE = {"tf": [5, 15, 30, 60], "k": [1.5, 2.0, 3.0],
             "tol": [0.002, 0.003, 0.005], "H": [24, 48, 96],
             "sides": ["both", "long", "short"]}
VOB_SPACE = {"tf": [5, 15, 30], "tgt": [0.002, 0.003, 0.004, 0.006],
             "stop": [0.0015, 0.002, 0.003], "H": [12, 24, 48],
             "q": [0.85, 0.90, 0.95], "feats": ["book", "bookprice", "price"]}


def log(s):
    print(s, flush=True)
    LOGF.parent.mkdir(exist_ok=True)
    with LOGF.open("a", encoding="utf-8") as f:
        f.write(s + "\n")


def sharpe_daily(times, rets):
    if len(rets) < 40:
        return None
    s = pd.Series(rets, index=pd.to_datetime(times))
    daily = s.resample("D").sum()
    daily = daily[daily.index.dayofweek < 5]
    span = pd.date_range(daily.index.min(), daily.index.max(), freq="B")
    daily = daily.reindex(span, fill_value=0.0)
    if daily.std() == 0:
        return None
    return float(daily.mean() / daily.std() * np.sqrt(252))


def eval_vpt(cfg, lo, hi):
    rets_all, times_all = [], []
    for tk in TICKERS:
        df = G.bars(tk, cfg["tf"])
        m = (df["timestamp"] >= lo) & (df["timestamp"] < hi)
        sub = df[m].reset_index(drop=True)
        if len(sub) < 300:
            continue
        setups = G.detect(sub, k=cfg["k"], tol=cfg["tol"], sides=cfg["sides"])
        r, t, _ = G.simulate(sub, setups, H=cfg["H"])
        rets_all.append(r)
        times_all.append(t)
    if not rets_all:
        return None
    r = np.concatenate(rets_all)
    t = np.concatenate(times_all)
    if len(r) < 40:
        return None
    return {"sharpe": sharpe_daily(t, r), "n": int(len(r)),
            "total": float(r.sum() * 100), "win": float((r > 0).mean())}


def load_vob():
    fs = sorted(VOB.glob("QQQ_*.parquet"))
    if not fs:
        return None
    b = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    b["timestamp"] = pd.to_datetime(b["timestamp"])
    return b.drop_duplicates(subset="timestamp").sort_values("timestamp") \
        .reset_index(drop=True)


def eval_vob(cfg, book, seg):
    """seg in {'opt','val','test'} by time-order fractions 60/20/20."""
    import lightgbm as lgb
    from alpaca_bot2 import full_series
    px = full_series("QQQ")[["timestamp", "high", "low", "close"]]
    d = book.merge(px, on="timestamp", how="inner").sort_values("timestamp") \
        .reset_index(drop=True)
    if cfg["tf"] > 5:
        g = d.set_index("timestamp").resample(f"{cfg['tf']}min").agg(
            imb=("imb", "mean"), micro_dev=("micro_dev", "mean"),
            spread_bps=("spread_bps", "mean"), quote_hz=("quote_hz", "mean"),
            bs=("bs", "last"), as_=("as_", "last"), high=("high", "max"),
            low=("low", "min"), close=("close", "last")).dropna().reset_index()
        d = g
    for col in ("imb", "micro_dev", "spread_bps", "quote_hz"):
        d[f"{col}_m3"] = d[col].rolling(3).mean().shift(1)
        d[f"{col}_m12"] = d[col].rolling(12).mean().shift(1)
    d["ret1"] = d["close"].pct_change()
    d["ret6"] = d["close"].pct_change(6)
    d["vol6"] = d["ret1"].rolling(6).std()
    book_f = ["imb", "micro_dev", "spread_bps", "quote_hz", "bs", "as_",
              "imb_m3", "imb_m12", "micro_dev_m3", "micro_dev_m12",
              "spread_bps_m3", "spread_bps_m12", "quote_hz_m3", "quote_hz_m12"]
    price_f = ["ret1", "ret6", "vol6"]
    feats = {"book": book_f, "bookprice": book_f + price_f, "price": price_f}[
        cfg["feats"]]
    c = d["close"].to_numpy()
    h = d["high"].to_numpy()
    l = d["low"].to_numpy()
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        up, dn = c[i] * (1 + cfg["tgt"]), c[i] * (1 - cfg["stop"])
        for j in range(i + 1, min(i + cfg["H"] + 1, n)):
            hu, hd = h[j] >= up, l[j] <= dn
            if hu and hd:
                break
            if hd:
                y[i] = 0; break
            if hu:
                y[i] = 1; break
    X = d[feats]
    fin = X.notna().all(axis=1).to_numpy() & np.isfinite(y)
    i60, i80 = int(n * 0.6), int(n * 0.8)
    seg_idx = {"opt": (0, i60), "val": (i60, i80), "test": (i80, n)}[seg]
    tr = np.where(fin)[0]
    tr = tr[tr < i60]
    tr = tr[:-cfg["H"]] if len(tr) > cfg["H"] else tr
    if len(tr) < 1500 or y[tr].sum() < 40:
        return None
    clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15,
                             min_child_samples=40, subsample=0.8,
                             colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
    clf.fit(X.iloc[tr], y[tr].astype(int))
    thr = np.quantile(clf.predict_proba(X.iloc[tr])[:, 1], cfg["q"])
    a, b2 = seg_idx
    te = np.where(fin)[0]
    te = te[(te >= a) & (te < b2)]
    if seg == "opt":
        te = te[te >= cfg["H"]]
    p = clf.predict_proba(X.iloc[te])[:, 1]
    rets, times = [], []
    last_exit = -1
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    for ii, i in enumerate(te):
        if p[ii] < thr or i <= last_exit:
            continue
        up, dn = c[i] * (1 + cfg["tgt"]), c[i] * (1 - cfg["stop"])
        res, j = None, i + 1
        while j < min(i + cfg["H"] + 1, n):
            if l[j] <= dn:
                res = -cfg["stop"]; break
            if h[j] >= up:
                res = cfg["tgt"]; break
            j += 1
        ex = min(j, n - 1)
        if res is None:
            res = (c[ex] - c[i]) / c[i]
        rets.append(res - G.COST)
        times.append(ts[i])
        last_exit = ex
    r = np.array(rets)
    if len(r) < 25:
        return None
    return {"sharpe": sharpe_daily(times, r), "n": int(len(r)),
            "total": float(r.sum() * 100), "win": float((r > 0).mean())}


def ckey(bot, cfg):
    return bot + "|" + json.dumps(cfg, sort_keys=True)


def sample(rng, st):
    bot = rng.choice(["vpt", "vpt", "vob"])          # patterns get 2/3 of budget
    space = VPT_SPACE if bot == "vpt" else VOB_SPACE
    # exploit: 30% of the time mutate a current leader instead of pure random
    board = [b for b in st["board"] if b["bot"] == bot]
    if board and rng.random() < 0.3:
        base = dict(rng.choice(board[:5])["cfg"])
        k = rng.choice(list(space))
        base[k] = rng.choice(space[k])
        return bot, base
    return bot, {k: rng.choice(v) for k, v in space.items()}


def step(budget=25):
    st = json.loads(STATE.read_text()) if STATE.exists() else \
        {"board": [], "tried": [], "test_log": []}
    tried = set(st["tried"])
    rng = random.Random(len(tried) * 977 + 13)
    book = load_vob()
    t0 = time.time()
    done = 0
    while done < budget and time.time() - t0 < 480:
        bot, cfg = sample(rng, st)
        key = ckey(bot, cfg)
        if key in tried:
            continue
        tried.add(key)
        st["tried"] = list(tried)
        try:
            if bot == "vpt":
                o = eval_vpt(cfg, *P_OPT)
                v = eval_vpt(cfg, *P_VAL) if o and (o["sharpe"] or 0) > 1.0 else None
            else:
                if book is None or len(book) < 4000:
                    continue
                o = eval_vob(cfg, book, "opt")
                v = eval_vob(cfg, book, "val") if o and (o["sharpe"] or 0) > 1.0 \
                    else None
        except Exception as e:
            log(f"  [eval error {bot} {cfg}: {e}]")
            continue
        done += 1
        os_ = o["sharpe"] if o and o["sharpe"] is not None else None
        vs_ = v["sharpe"] if v and v["sharpe"] is not None else None
        log(f"  {bot} {json.dumps(cfg, sort_keys=True)} "
            f"OPT {os_ if os_ is None else round(os_, 2)} "
            f"VAL {vs_ if vs_ is None else round(vs_, 2)} "
            f"(n_opt={o['n'] if o else 0})")
        if o and v and vs_ is not None:
            st["board"].append({"bot": bot, "cfg": cfg, "opt": o, "val": v,
                                "val_sharpe": vs_})
            st["board"].sort(key=lambda z: -(z["val_sharpe"] or -9))
            st["board"] = st["board"][:40]
        STATE.write_text(json.dumps(st))
    STATE.write_text(json.dumps(st))
    log(f"step done: {done} evals, board size {len(st['board'])}, "
        f"tried {len(tried)}")
    if st["board"]:
        b = st["board"][0]
        log(f"leader: {b['bot']} VAL Sharpe {b['val_sharpe']:.2f} "
            f"(n={b['val']['n']}, total {b['val']['total']:+.1f}%) {b['cfg']}")


def board():
    st = json.loads(STATE.read_text())
    print(f"{'bot':>4} {'VAL Shp':>8} {'VAL tot':>8} {'n':>5} {'OPT Shp':>8}  cfg")
    for b in st["board"][:15]:
        print(f"{b['bot']:>4} {b['val_sharpe']:>8.2f} "
              f"{b['val']['total']:>+7.1f}% {b['val']['n']:>5} "
              f"{(b['opt']['sharpe'] or 0):>8.2f}  {json.dumps(b['cfg'])}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "board":
        board()
    else:
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 25
        step(n)
