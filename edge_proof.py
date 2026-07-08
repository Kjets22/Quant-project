"""
edge_proof.py — (A) FULL backtest in the audited environment (embargo, no-bfill ATR,
0.12% ATR floor, 3 bps cost + 1 bp/side slippage, double exposure allowed) for v3+v4
pooled: total, monthly Sharpe, maxDD. (B) PROOF of where the edge comes from:

  B1. RANDOM-ENTRY baseline — same brackets/frequency/names, entries chosen at random.
      Whatever random earns = bracket geometry + market drift. Model minus random =
      the model's true timing edge.
  B2. FEATURE-GROUP KNOCKOUT — at test time, shuffle one feature family (volume /
      volatility / momentum / location) and measure the win-rate drop. The family whose
      destruction hurts most is the signal carrier. (Permutation importance, OOS.)
  B3. SIGNAL CHARACTERIZATION — z-scored mean of each feature at the model's chosen
      entries vs all bars: a picture of WHAT the model buys.

Standalone; touches no frozen snapshot.
"""

from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import lightgbm as lgb

from triple_barrier_ml import features, label
from triple_barrier_breadth import TICKERS
from sr_features import sr_features

SEL_Q, HBAR = 0.93, 24
EFF_COST = (3.0 + 2 * 1.0) / 1e4          # 3 bps + 1 bp slippage per side
MIN_ATR_PCT = 0.0012
GROUPS = {
    "volume":     ["volz"],
    "volatility": ["vol24", "atrpct"],
    "momentum":   ["ret1", "ret6", "ret24", "rsi", "sma20d", "sma50d"],
    "location":   ["rangepos", "sr_res20", "sr_sup20", "sr_res60", "sr_sup60",
                   "sr_pdh", "sr_pdl", "sr_pdc", "sr_round5", "sr_vwap", "sr_rangepos"],
}
RNG = np.random.default_rng(7)


def atr_fixed(h, l, c, n=24):
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n).mean().to_numpy()


def bars(tk, mins):
    df = pd.read_csv(f"data_cache/{tk}_5minute_2021-06-01_2026-06-01.csv", parse_dates=["timestamp"])
    g = df.set_index("timestamp").resample(f"{mins}min").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum")).dropna()
    return g.reset_index()


def trade_loop(take_mask, hv, lv, cv, Av, i0, i1, n, tp, sl):
    """Walk bars i0..i1; enter where take_mask (non-overlapping); return (ts_idx, ret, win)."""
    out, i = [], i0
    while i < i1 - 1:
        if not take_mask[i - i0] or Av[i] / cv[i] < MIN_ATR_PCT:
            i += 1; continue
        a = Av[i]; up, dn = cv[i] + tp * a, cv[i] - sl * a
        res, j = None, i + 1
        while j < min(i + HBAR + 1, n):
            if lv[j] <= dn:
                res = 0; break
            if hv[j] >= up:
                res = 1; break
            j += 1
        if res is None:
            res = 1 if cv[min(j, n - 1)] > cv[i] else 0
        out.append((i, (tp * a if res == 1 else -sl * a) / cv[i] - EFF_COST, res))
        i = j + 1
    return out


def run_config(mins, tp, sl, collect_char=False):
    """Full audited walk-forward. Returns dict of trade lists per mode + characterization."""
    modes = ["model", "random"] + [f"ko_{g}" for g in GROUPS]
    trades = {m: [] for m in modes}
    char_sel, char_all = [], []
    cols_order = None
    for tk in TICKERS:
        d = bars(tk, mins)
        ts = pd.to_datetime(d["timestamp"]).to_numpy()
        h, l, c, v = (d[x].to_numpy(float) for x in ("high", "low", "close", "volume"))
        A = atr_fixed(h, l, c)
        X = pd.concat([features(h, l, c, v).reset_index(drop=True),
                       sr_features(d).reset_index(drop=True)], axis=1)
        cols_order = list(X.columns)
        y = label(h, l, c, A, tp, sl)
        m = (X.notna().all(axis=1) & np.isfinite(y) & np.isfinite(A)).to_numpy()
        idx = np.where(m)[0]
        Xv = X.iloc[idx].reset_index(drop=True)
        yv = y[idx].astype(int)
        hv, lv, cv, Av, tsv = h[idx], l[idx], c[idx], A[idx], ts[idx]
        n = len(idx); K = 5
        bnds = np.linspace(int(n * 0.4), n, K + 1).astype(int)
        for k in range(K):
            tr_end = max(bnds[k] - HBAR, 300)                     # embargo
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                     min_child_samples=40, subsample=0.8, colsample_bytree=0.8,
                                     reg_lambda=1.0, verbose=-1)
            clf.fit(Xv.iloc[:tr_end], yv[:tr_end])
            thr = np.quantile(clf.predict_proba(Xv.iloc[:tr_end])[:, 1], SEL_Q)
            Xte = Xv.iloc[bnds[k]:bnds[k + 1]]
            proba = clf.predict_proba(Xte)[:, 1]
            take = proba >= thr
            # --- model ---
            for i, r, w in trade_loop(take, hv, lv, cv, Av, bnds[k], bnds[k + 1], n, tp, sl):
                trades["model"].append((tsv[i], r, w))
                if collect_char:
                    char_sel.append(Xv.iloc[i].to_numpy(float))
            if collect_char:
                char_all.append(Xte.to_numpy(float))
            # --- random baseline: same take-frequency, random positions ---
            rtake = RNG.random(len(take)) < take.mean()
            for i, r, w in trade_loop(rtake, hv, lv, cv, Av, bnds[k], bnds[k + 1], n, tp, sl):
                trades["random"].append((tsv[i], r, w))
            # --- knockouts: shuffle one group at test time ---
            for g, cols in GROUPS.items():
                Xko = Xte.copy()
                for col in cols:
                    Xko[col] = RNG.permutation(Xko[col].to_numpy())
                pko = clf.predict_proba(Xko)[:, 1]
                for i, r, w in trade_loop(pko >= thr, hv, lv, cv, Av, bnds[k], bnds[k + 1], n, tp, sl):
                    trades[f"ko_{g}"].append((tsv[i], r, w))
    char = None
    if collect_char:
        allX = np.vstack(char_all)
        mu, sd = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-12
        zsel = (np.nanmean(np.vstack(char_sel), 0) - mu) / sd
        char = sorted(zip(cols_order, zsel), key=lambda x: -abs(x[1]))
    return trades, char


def stats(tlist, be):
    if not tlist:
        return 0, 0, 0, 0
    r = np.array([x[1] for x in tlist])
    w = np.mean([x[2] for x in tlist])
    return len(r), w, w - be, r.sum() * 100


def main():
    print("ENVIRONMENT: embargoed, no-bfill ATR, ATR floor 0.12%, cost 5 bps eff (3+2 slip),")
    print("double exposure allowed (v3 and v4 pooled independently)\n")
    all_month = {}
    for mins, tp, sl, name in ((30, 1.5, 1.0, "v3"), (15, 4.0, 1.0, "v4")):
        be = sl / (sl + tp)
        trades, char = run_config(mins, tp, sl, collect_char=(name == "v3"))
        print(f"=== {name} ({mins}-min / {tp:g}:{sl:g}, break-even {be:.0%}) ===")
        print(f"  {'mode':>14} {'trades':>7} {'win%':>6} {'margin':>8} {'total%':>8}")
        for mode in ["model", "random"] + [f"ko_{g}" for g in GROUPS]:
            n, w, mg, tot = stats(trades[mode], be)
            print(f"  {mode:>14} {n:>7} {w:>6.1%} {mg:>+8.1%} {tot:>+8.0f}")
        # monthly series for the full-portfolio Sharpe
        s = pd.DataFrame(trades["model"], columns=["ts", "ret", "win"])
        s["ts"] = pd.to_datetime(s["ts"])
        all_month[name] = s.set_index("ts")["ret"].resample("ME").sum()
        if char:
            print("  signal fingerprint (z-scored feature means at chosen entries, |top 8|):")
            for cname, z in char[:8]:
                print(f"     {cname:>12}: {z:+.2f} sd")
        print()
    idx = all_month["v3"].index.union(all_month["v4"].index)
    port = (all_month["v3"].reindex(idx, fill_value=0) + all_month["v4"].reindex(idx, fill_value=0))
    cum = port.cumsum()
    dd = (cum - cum.cummax()).min()
    sharpe = port.mean() / port.std() * np.sqrt(12)
    print("=== FULL PORTFOLIO (v3 + v4 pooled, double exposure) ===")
    print(f"  months={len(port)}  total={cum.iloc[-1]*100:+.0f}%  "
          f"monthly Sharpe={sharpe:.2f}  maxDD={dd*100:+.1f}%  "
          f"positive months={(port>0).mean():.0%}")


if __name__ == "__main__":
    main()
