"""
options_ml_pipeline.py — learn from options data the RIGHT way (VRP, not prices).

Pipeline: pull underlying + chain -> drop illiquid/broken/far-OTM noise ->
collapse the surface into a few clean IV-space features -> label FORWARD realized
vol (Yang-Zhang) -> LightGBM under a time-ordered, PURGED split. Compare the
model's predicted RV to today's implied vol = the volatility-risk-premium signal.

Runs out of the box on a SYNTHETIC chain (plumbing check). Swap
generate_synthetic_data() for the real Polygon loader (options_data_polygon.py)
once that's verified.

Split (per request): most-recent 12 months -> 10mo TRAIN / 1mo VAL / 1mo TEST.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


@dataclass
class Config:
    min_open_interest: int = 50
    min_volume: int = 1
    max_rel_spread: float = 0.10
    min_abs_delta: float = 0.10
    max_abs_delta: float = 0.90
    min_dte: int = 7
    max_dte: int = 90
    rv_label_horizon: int = 21
    rv_trailing_window: int = 21
    iv_rank_window: int = 252
    target_dte_for_surface: int = 30
    embargo_days: int = 21
    risk_free_rate: float = 0.04
    # 10/1/1-month split
    train_months: int = 10
    val_months: int = 1
    test_months: int = 1


CFG = Config()


# --------------------------------------------------------------------------- #
# Black-Scholes helpers (synthetic generator + real-data IV inversion)        #
# --------------------------------------------------------------------------- #
def _bs_price_and_delta(S, K, T, r, sigma, is_call):
    if T <= 0:
        intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
        return intrinsic, (1.0 if (is_call and S > K) else 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
    return price, delta


def implied_vol(price, S, K, T, r, is_call, lo=1e-3, hi=5.0):
    """Invert Black-Scholes for IV via bisection. Returns NaN if no solution."""
    if T <= 0 or price <= 0:
        return np.nan
    intrinsic = max(S - K, 0) if is_call else max(K - S, 0)
    if price < intrinsic - 1e-6:
        return np.nan
    f = lambda s: _bs_price_and_delta(S, K, T, r, s, is_call)[0] - price
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        return np.nan
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-6:
            return mid
        if flo * fm < 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def generate_synthetic_data(n_days: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime(2024, 6, 1)
    dates = [start + timedelta(days=i) for i in range(n_days)]
    S0, mu, base_vol = 100.0, 0.06, 0.18
    daily_ret = rng.normal(mu / 252, base_vol / np.sqrt(252), n_days)
    close = S0 * np.exp(np.cumsum(daily_ret))
    intraday = base_vol / np.sqrt(252)
    high = close * np.exp(np.abs(rng.normal(0, intraday, n_days)))
    low = close * np.exp(-np.abs(rng.normal(0, intraday, n_days)))
    open_ = np.r_[S0, close[:-1]] * np.exp(rng.normal(0, intraday / 2, n_days))
    vol_regime = base_vol + 0.06 * np.sin(np.linspace(0, 6 * np.pi, n_days)) \
        + rng.normal(0, 0.01, n_days)

    rows = []
    for i, d in enumerate(dates):
        S = close[i]
        atm_iv = max(vol_regime[i], 0.05)
        for dte in (14, 30, 45, 60, 90):
            T = dte / 365.0
            expiry = d + timedelta(days=dte)
            for K in np.arange(round(S * 0.85), round(S * 1.15), 2.5):
                k = np.log(K / S)
                iv = max(atm_iv - 0.6 * k + 2.0 * k**2
                         + 0.02 * (np.sqrt(30 / dte) - 1)
                         + rng.normal(0, 0.004), 0.03)
                for is_call in (True, False):
                    price, delta = _bs_price_and_delta(
                        S, K, T, CFG.risk_free_rate, iv, is_call)
                    if price < 0.02:
                        continue
                    rel = min(0.008 + 0.03 * abs(k) + 0.05 / max(price, 0.5), 0.3)
                    half = price * rel / 2
                    bid = max(price - half + rng.normal(0, half * 0.1), 0.0)
                    ask = price + half + rng.normal(0, half * 0.1)
                    liq = np.exp(-(k / 0.15) ** 2)
                    rows.append({
                        "date": d, "underlying": "SYN", "expiry": expiry,
                        "strike": float(K), "type": "C" if is_call else "P",
                        "bid": round(bid, 2), "ask": round(ask, 2),
                        "volume": int(max(rng.poisson(120 * liq), 0)),
                        "open_interest": int(max(rng.poisson(800 * liq), 0)),
                        "iv": iv, "delta": delta, "underlying_price": S,
                        "open": open_[i], "high": high[i], "low": low[i],
                        "close": close[i],
                    })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["expiry"] = pd.to_datetime(df["expiry"])
    return df


# --------------------------------------------------------------------------- #
# Cleaning + liquidity/moneyness/DTE filter                                   #
# --------------------------------------------------------------------------- #
def clean_quotes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mid"] = (df["bid"] + df["ask"]) / 2
    df["dte"] = (df["expiry"] - df["date"]).dt.days
    before = len(df)
    df = df[(df["bid"] > 0) & (df["ask"] > df["bid"]) & (df["mid"] > 0)]
    intrinsic = np.where(df["type"] == "C",
                         np.maximum(df["underlying_price"] - df["strike"], 0),
                         np.maximum(df["strike"] - df["underlying_price"], 0))
    df = df[df["mid"] >= intrinsic - 0.01]
    df["rel_spread"] = (df["ask"] - df["bid"]) / df["mid"]
    print(f"[clean] dropped {before - len(df):,} broken quotes ({len(df):,} remain)")
    return df


def filter_tradeable(df: pd.DataFrame, cfg: Config = CFG) -> pd.DataFrame:
    before = len(df)
    m = ((df["open_interest"] >= cfg.min_open_interest)
         & (df["volume"] >= cfg.min_volume)
         & (df["rel_spread"] <= cfg.max_rel_spread)
         & (df["delta"].abs() >= cfg.min_abs_delta)
         & (df["delta"].abs() <= cfg.max_abs_delta)
         & (df["dte"] >= cfg.min_dte) & (df["dte"] <= cfg.max_dte))
    out = df[m].copy()
    print(f"[filter] kept {len(out):,} liquid/near-money (removed {before-len(out):,})")
    return out


# --------------------------------------------------------------------------- #
# Yang-Zhang realized vol                                                      #
# --------------------------------------------------------------------------- #
def yang_zhang_rv(ohlc: pd.DataFrame, window: int) -> pd.Series:
    o, h, l, c = (np.log(ohlc[x]) for x in ["open", "high", "low", "close"])
    overnight = o - c.shift(1)
    open_close = c - o
    rs = (h - c) * (h - o) + (l - c) * (l - o)
    n = window
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    yz = (overnight.rolling(n).var() + k * open_close.rolling(n).var()
          + (1 - k) * rs.rolling(n).mean())
    return np.sqrt(yz * 252)


# --------------------------------------------------------------------------- #
# IV-space feature engineering                                                #
# --------------------------------------------------------------------------- #
def _iv_at_delta(group, target_delta, is_call):
    side = group[group["type"] == ("C" if is_call else "P")]
    if len(side) < 2:
        return np.nan
    d = side["delta"].abs().values
    iv = side["iv"].values
    order = np.argsort(d)
    return float(np.interp(target_delta, d[order], iv[order]))


def build_daily_features(df, ohlc_daily, cfg: Config = CFG) -> pd.DataFrame:
    feats = []
    for date, day in df.groupby("date"):
        day = day.assign(dd=(day["dte"] - cfg.target_dte_for_surface).abs())
        near = day[day["dd"] == day["dd"].min()]
        atm = _iv_at_delta(near, 0.50, True)
        c25 = _iv_at_delta(near, 0.25, True)
        p25 = _iv_at_delta(near, 0.25, False)
        short = day[day["dte"] <= 21]
        lng = day[day["dte"] >= 45]
        a_s = _iv_at_delta(short, 0.50, True) if len(short) else np.nan
        a_l = _iv_at_delta(lng, 0.50, True) if len(lng) else np.nan
        cv = day[day["type"] == "C"]["volume"].sum()
        pv = day[day["type"] == "P"]["volume"].sum()
        coi = day[day["type"] == "C"]["open_interest"].sum()
        poi = day[day["type"] == "P"]["open_interest"].sum()
        feats.append({
            "date": date, "atm_iv": atm,
            "skew_25d": (p25 - c25) if (p25 and c25) else np.nan,
            "smile_curv": ((p25 + c25) / 2 - atm)
                          if all(v is not None for v in [p25, c25, atm]) else np.nan,
            "term_slope": (a_l - a_s) if (a_l and a_s) else np.nan,
            "pc_vol_ratio": pv / cv if cv else np.nan,
            "pc_oi_ratio": poi / coi if coi else np.nan,
            "total_vol": cv + pv,
        })
    fdf = pd.DataFrame(feats).set_index("date").sort_index()
    rv_trail = yang_zhang_rv(ohlc_daily, cfg.rv_trailing_window)
    fdf["rv_trailing"] = rv_trail.reindex(fdf.index)
    fdf["vrp_feature"] = fdf["atm_iv"] - fdf["rv_trailing"]
    fdf["iv_rank"] = fdf["atm_iv"].rolling(cfg.iv_rank_window, min_periods=20) \
        .apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-9))
    fdf["atm_iv_chg"] = fdf["atm_iv"].diff()
    fdf["skew_chg"] = fdf["skew_25d"].diff()
    return fdf


def add_label(features, ohlc_daily, cfg: Config = CFG) -> pd.DataFrame:
    fwd_rv = yang_zhang_rv(ohlc_daily, cfg.rv_label_horizon).shift(-cfg.rv_label_horizon)
    out = features.copy()
    out["y_forward_rv"] = fwd_rv.reindex(out.index)
    return out.dropna()


# --------------------------------------------------------------------------- #
# 10/1/1-month split (purged) + LightGBM                                      #
# --------------------------------------------------------------------------- #
def time_split_10_1_1(df, cfg: Config = CFG):
    idx = df.index
    last = idx.max()
    test_start = last - pd.DateOffset(months=cfg.test_months)
    val_start = test_start - pd.DateOffset(months=cfg.val_months)
    train_start = val_start - pd.DateOffset(months=cfg.train_months)
    emb = pd.Timedelta(days=cfg.embargo_days)
    train = df[(idx >= train_start) & (idx < val_start - emb)]
    val = df[(idx >= val_start) & (idx < test_start - emb)]
    test = df[idx >= test_start]
    return train, val, test


def train_model(df: pd.DataFrame):
    fcols = [c for c in df.columns if c != "y_forward_rv"]
    train, val, test = time_split_10_1_1(df)
    print(f"\n[split] train={len(train)}d  val={len(val)}d  test={len(test)}d "
          f"({train.index.min().date()}..{test.index.max().date()})")
    if not HAS_LGB or len(test) < 5:
        print("[model] lightgbm missing or test too small — skipping.")
        return None
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15,
                              min_child_samples=20, subsample=0.8,
                              colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=0, verbose=-1)
    model.fit(train[fcols], train["y_forward_rv"])
    for name, part in (("VAL", val), ("TEST", test)):
        if len(part) < 3:
            continue
        pred = model.predict(part[fcols])
        y = part["y_forward_rv"].values
        iv = part["atm_iv"].values
        mae = np.mean(np.abs(pred - y))
        hit = np.mean(np.sign(iv - y) == np.sign(iv - pred))  # VRP-sign skill
        print(f"  {name}: MAE(fwd RV) {mae:.4f}   VRP-sign hit {hit:.1%} (50%=no skill)")
    imp = pd.Series(model.feature_importances_, index=fcols).sort_values(ascending=False)
    print("  top features:", ", ".join(f"{n}({v:.0f})" for n, v in imp.head(5).items()))
    return model


def main():
    print("Generating synthetic options data (plumbing check)...")
    raw = generate_synthetic_data(n_days=420)
    ohlc = raw.groupby("date")[["open", "high", "low", "close"]].first().sort_index()
    df = clean_quotes(raw)
    df = filter_tradeable(df)
    feats = build_daily_features(df, ohlc)
    data = add_label(feats, ohlc)
    print(f"\nFinal table: {data.shape[0]} days x {data.shape[1]-1} features")
    train_model(data)
    print("\nNB: synthetic numbers are a PLUMBING check only, not an edge.")


if __name__ == "__main__":
    main()
