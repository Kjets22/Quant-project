"""
qqq_tournament.py — TOURNAMENT: predict which comes first on QQQ, +$2 or -$2 (60-min window).
Metric = directional ACCURACY on resolved races (unresolved/ambiguous bars excluded).

Honest structure (a tournament is mass multiple-testing, so):
  TRAIN  : all data < 2025-07-14
  ARENA  : 2025-07-14 .. 2026-01-14  (leaderboard; rounds 1-2 select here)
  FINAL  : 2026-01-14 .. now          (untouched; champion gets ONE shot)

Round 1 (`arena <featset>`): 5 model families x 4 feature sets fight on the arena.
Round 2 (`round2`): cross-pollination — ensemble the top-3 arena combos (probability
  averaging) + tuned variants of the winner; best becomes champion.
Round 3 (`final`): champion refit on train+arena, evaluated once on FINAL.

  python qqq_tournament.py arena F1|F2|F3|F4
  python qqq_tournament.py round2
  python qqq_tournament.py final
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from triple_barrier_ml import features
from sr_features import sr_features
from wide_hunter import atr_fixed, trend_features

H, TGT = 12, 2.0
DEV0, DEV1 = "2025-07-14", "2026-01-14"
RUNS = Path("runs")


def extra_features(d, h, l, c, v, A):
    ts = pd.to_datetime(d["timestamp"])
    r = np.diff(np.log(c), prepend=np.log(c[0]))
    f = {"x_hour_sin": np.sin(2 * np.pi * ts.dt.hour / 24),
         "x_hour_cos": np.cos(2 * np.pi * ts.dt.hour / 24),
         "x_dow": ts.dt.dayofweek.to_numpy(float)}
    for k in range(1, 7):
        f[f"x_r{k}"] = pd.Series(r).shift(k - 1).to_numpy()
    atr6 = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(6).mean().to_numpy()
    f["x_volratio"] = atr6 / (A + 1e-9)
    f["x_rng12"] = (c - pd.Series(l).rolling(12).min().to_numpy()) / \
                   (pd.Series(h).rolling(12).max().to_numpy()
                    - pd.Series(l).rolling(12).min().to_numpy() + 1e-9)
    return pd.DataFrame(f)


def load():
    base = pd.read_csv("data_cache/QQQ_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    parts = [base]
    rec = sorted(Path("data_cache").glob("QQQ_recent_2026-06-01_*.csv"))
    if rec:
        parts.append(pd.read_csv(rec[-1], parse_dates=["timestamp"]))
    d = (pd.concat(parts, ignore_index=True)
           .drop_duplicates(subset="timestamp", keep="last")
           .sort_values("timestamp").reset_index(drop=True))
    ts = pd.to_datetime(d["timestamp"]).to_numpy()
    h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
    A = atr_fixed(h, l, c)
    Fb = features(h, l, c, v).reset_index(drop=True)
    Fs = sr_features(d).reset_index(drop=True)
    Ft = trend_features(h, l, c, A).reset_index(drop=True)
    Fx = extra_features(d, h, l, c, v, A).reset_index(drop=True)
    feats = {"F1": Fb, "F2": pd.concat([Fb, Fs], axis=1),
             "F3": pd.concat([Fb, Fs, Ft], axis=1),
             "F4": pd.concat([Fb, Fs, Ft, Fx], axis=1)}
    # symmetric race label: 1 = +$2 first, 0 = -$2 first, NaN = unresolved/ambiguous
    n = len(c)
    y = np.full(n, np.nan)
    for i in range(n - 1):
        up, dn = c[i] + TGT, c[i] - TGT
        for j in range(i + 1, min(i + H + 1, n)):
            hit_up, hit_dn = h[j] >= up, l[j] <= dn
            if hit_up and hit_dn:
                break                      # ambiguous same-bar double touch -> exclude
            if hit_dn:
                y[i] = 0; break
            if hit_up:
                y[i] = 1; break
    return ts, feats, y


MODELS = {
    "lgbm_s": lambda: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                         min_child_samples=40, subsample=0.8,
                                         colsample_bytree=0.8, reg_lambda=1.0, verbose=-1),
    "lgbm_d": lambda: lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                         min_child_samples=100, subsample=0.8,
                                         colsample_bytree=0.7, reg_lambda=3.0, verbose=-1),
    "histgb": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                     max_leaf_nodes=31, min_samples_leaf=50),
    "rf":     lambda: RandomForestClassifier(n_estimators=60, min_samples_leaf=50,
                                             max_features=0.5, n_jobs=-1),
    "logreg": lambda: LogisticRegression(max_iter=300),
}


def masked(feats_df, y, ts, lo=None, hi=None):
    m = feats_df.notna().all(axis=1).to_numpy() & np.isfinite(y)
    if lo is not None:
        m &= ts >= np.datetime64(lo)
    if hi is not None:
        m &= ts < np.datetime64(hi)
    return np.where(m)[0]


def fit_eval(model_name, X, y, tr, te):
    Xtr, Xte = X.iloc[tr].to_numpy(), X.iloc[te].to_numpy()
    if model_name == "logreg":
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    clf = MODELS[model_name]()
    clf.fit(Xtr, y[tr].astype(int))
    proba = clf.predict_proba(Xte)[:, 1]
    acc = ((proba >= 0.5).astype(int) == y[te]).mean()
    return acc, proba


def arena(featset):
    ts, feats, y = load()
    X = feats[featset]
    tr = masked(X, y, ts, hi=DEV0)
    tr = tr[:-H]                                            # embargo
    te = masked(X, y, ts, lo=DEV0, hi=DEV1)
    base = max(y[te].mean(), 1 - y[te].mean())
    out = {"featset": featset, "n_test": int(len(te)), "base_acc": float(base), "models": {}}
    print(f"ARENA {featset}: {len(tr)} train, {len(te)} arena races, "
          f"majority-class baseline {base:.1%}", flush=True)
    RUNS.mkdir(exist_ok=True)
    for name in MODELS:
        acc, proba = fit_eval(name, X, y, tr, te)
        out["models"][name] = float(acc)
        np.save(RUNS / f"tourn_p_{featset}_{name}.npy", proba)
        print(f"  {name:>7}: accuracy {acc:.2%}", flush=True)
    (RUNS / f"tourn_{featset}.json").write_text(json.dumps(out))


def leaderboard():
    rows = []
    for f in RUNS.glob("tourn_F*.json"):
        d = json.loads(f.read_text())
        for m, a in d["models"].items():
            rows.append((a, d["featset"], m, d["base_acc"]))
    rows.sort(reverse=True)
    return rows


def round2():
    ts, feats, y = load()
    rows = leaderboard()
    base = rows[0][3]
    te = masked(feats["F1"], y, ts, lo=DEV0, hi=DEV1)       # same test rows for all
    print("ROUND-1 LEADERBOARD (arena accuracy | majority baseline "
          f"{base:.1%}):")
    for a, fs, m, _ in rows[:8]:
        print(f"  {a:.2%}  {fs}/{m}")
    top3 = rows[:3]
    # cross-pollination 1: probability-average ensemble of the top 3
    pav = np.mean([np.load(RUNS / f"tourn_p_{fs}_{m}.npy") for _, fs, m, _ in top3], axis=0)
    acc_ens = (((pav >= 0.5).astype(int)) == y[te]).mean()
    print(f"  ENSEMBLE(top3): {acc_ens:.2%}")
    # cross-pollination 2: winner's model family, tuned, on the winner's features
    _, fs_w, m_w, _ = rows[0]
    X = feats[fs_w]
    tr = masked(X, y, ts, hi=DEV0)[:-H]
    tuned_best, tuned_spec = 0.0, None
    if "lgbm" in m_w or m_w == "histgb":
        for nl, ne in ((31, 400), (63, 300), (127, 250)):
            clf = lgb.LGBMClassifier(n_estimators=ne, learning_rate=0.04, num_leaves=nl,
                                     min_child_samples=60, subsample=0.8,
                                     colsample_bytree=0.8, reg_lambda=2.0, verbose=-1)
            clf.fit(X.iloc[tr], y[tr].astype(int))
            acc = ((clf.predict_proba(X.iloc[te])[:, 1] >= 0.5).astype(int) == y[te]).mean()
            print(f"  TUNED lgbm(nl={nl},ne={ne}) on {fs_w}: {acc:.2%}")
            if acc > tuned_best:
                tuned_best, tuned_spec = acc, {"kind": "tuned_lgbm", "featset": fs_w,
                                               "num_leaves": nl, "n_estimators": ne}
    cands = [(rows[0][0], {"kind": "single", "featset": fs_w, "model": m_w}),
             (acc_ens, {"kind": "ensemble", "members": [[fs, m] for _, fs, m, _ in top3]}),
             (tuned_best, tuned_spec)]
    cands = [c for c in cands if c[1] is not None]
    cands.sort(key=lambda x: -x[0])
    champ = {"arena_acc": cands[0][0], "spec": cands[0][1], "baseline": base}
    (RUNS / "tourn_champion.json").write_text(json.dumps(champ))
    print(f"\nCHAMPION: {cands[0][1]}  (arena {cands[0][0]:.2%} vs baseline {base:.1%})")


def final():
    ts, feats, y = load()
    champ = json.loads((RUNS / "tourn_champion.json").read_text())
    spec = champ["spec"]
    te = masked(feats["F1"], y, ts, lo=DEV1)
    base = max(y[te].mean(), 1 - y[te].mean())

    def refit_proba(fs, model_name, extra=None):
        X = feats[fs]
        tr = masked(X, y, ts, hi=DEV1)[:-H]                 # train + arena, embargoed
        if extra:                                            # tuned lgbm
            clf = lgb.LGBMClassifier(n_estimators=extra["n_estimators"], learning_rate=0.04,
                                     num_leaves=extra["num_leaves"], min_child_samples=60,
                                     subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                                     verbose=-1)
            clf.fit(X.iloc[tr], y[tr].astype(int))
            return clf.predict_proba(X.iloc[te])[:, 1]
        _, proba = fit_eval(model_name, X, y, tr, te)
        return proba

    if spec["kind"] == "single":
        proba = refit_proba(spec["featset"], spec["model"])
    elif spec["kind"] == "tuned_lgbm":
        proba = refit_proba(spec["featset"], None, extra=spec)
    else:
        proba = np.mean([refit_proba(fs, m) for fs, m in spec["members"]], axis=0)
    pred = (proba >= 0.5).astype(int)
    acc = (pred == y[te]).mean()
    hiconf = np.abs(proba - 0.5) >= np.quantile(np.abs(proba - 0.5), 0.8)
    acc_hi = (pred[hiconf] == y[te][hiconf]).mean()
    print(f"FINAL (untouched {DEV1}..now): {len(te)} races")
    print(f"  majority baseline: {base:.2%}")
    print(f"  CHAMPION accuracy: {acc:.2%}   (edge over baseline {acc-base:+.2%})")
    print(f"  top-20%-confidence accuracy: {acc_hi:.2%}  ({hiconf.sum()} races)")
    print(f"  champion spec: {spec}  (arena was {champ['arena_acc']:.2%})")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "arena"
    if mode == "arena":
        arena(sys.argv[2] if len(sys.argv) > 2 else "F1")
    elif mode == "round2":
        round2()
    elif mode == "final":
        final()
