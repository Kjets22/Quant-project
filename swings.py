"""
swings.py — Part 2: swing detection and oracle leg ranges.

These quantities are derived from the FUTURE (a swing is only confirmed by what
happens after it). They are the oracle. Per the lookahead wall, they may be used
ONLY to compute the reward (reward.py) and MUST NEVER enter the observation.

Definitions (exact):
  * Swing low at index i: low[i] <= min(low[i-w : i])  AND  low[i] < min(low[i+1 : i+1+w])
  * Swing high at index i: high[i] >= max(high[i-w : i]) AND high[i] > max(high[i+1 : i+1+w])
    where w = cfg.swing.confirm.
  * A leg is the move between two consecutive (alternating) swings; its oracle
    range is abs(price_at_swing_b - price_at_swing_a).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config, default_config


def detect_swings(df: pd.DataFrame, cfg: Config) -> list[tuple[int, str, float]]:
    """
    Return confirmed swings as a list of (index, kind, price), kind in {"L","H"}.
    Swing-low price is the bar low; swing-high price is the bar high.
    """
    w = cfg.swing.confirm
    low = df["low"].to_numpy()
    high = df["high"].to_numpy()
    n = len(df)
    swings: list[tuple[int, str, float]] = []

    for i in range(w, n - w):
        prior_low = low[i - w:i]
        next_low = low[i + 1:i + 1 + w]
        if low[i] <= prior_low.min() and low[i] < next_low.min():
            swings.append((i, "L", float(low[i])))
            continue  # a bar cannot be both a confirmed low and high

        prior_high = high[i - w:i]
        next_high = high[i + 1:i + 1 + w]
        if high[i] >= prior_high.max() and high[i] > next_high.max():
            swings.append((i, "H", float(high[i])))

    return swings


def alternate_swings(
    swings: list[tuple[int, str, float]]
) -> list[tuple[int, str, float]]:
    """
    Collapse runs of consecutive same-kind swings into a clean alternating
    L,H,L,H,... sequence, keeping the more extreme swing in each run
    (lowest low for "L", highest high for "H").
    """
    if not swings:
        return []

    out: list[tuple[int, str, float]] = [swings[0]]
    for idx, kind, price in swings[1:]:
        prev_idx, prev_kind, prev_price = out[-1]
        if kind == prev_kind:
            # Same kind: keep the more extreme one.
            if (kind == "L" and price < prev_price) or (
                kind == "H" and price > prev_price
            ):
                out[-1] = (idx, kind, price)
        else:
            out.append((idx, kind, price))
    return out


def build_leg_ranges(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """
    Per-bar oracle range: each index holds abs(price_b - price_a) of the leg it
    belongs to, floored elementwise at min_leg_range_frac * close[i].

    Falls back to a rolling (max - min) range when fewer than 2 swings exist.
    Guaranteed strictly positive everywhere (the floor is positive for any
    positive price).
    """
    n = len(df)
    close = df["close"].to_numpy()
    floor = cfg.swing.min_leg_range_frac * close

    swings = alternate_swings(detect_swings(df, cfg))

    if len(swings) < 2:
        # Fallback: rolling peak-to-trough range over the confirm window.
        w = max(cfg.swing.confirm, 2)
        roll_hi = df["high"].rolling(w, min_periods=1).max().to_numpy()
        roll_lo = df["low"].rolling(w, min_periods=1).min().to_numpy()
        leg = np.abs(roll_hi - roll_lo)
        return np.maximum(leg, floor)

    leg = np.zeros(n, dtype=np.float64)

    # Fill each leg [a_idx .. b_idx] with that leg's oracle range.
    for (a_idx, _ak, a_price), (b_idx, _bk, b_price) in zip(swings[:-1], swings[1:]):
        rng = abs(b_price - a_price)
        leg[a_idx:b_idx + 1] = rng

    # Pad the head (before first swing) and tail (after last swing).
    first_idx = swings[0][0]
    last_idx = swings[-1][0]
    if first_idx > 0:
        leg[:first_idx] = leg[first_idx]
    if last_idx < n - 1:
        leg[last_idx:] = leg[last_idx] if leg[last_idx] > 0 else leg[last_idx - 1]

    # Any residual zeros (degenerate equal-price legs) get the floor.
    leg = np.where(leg > 0, leg, floor)
    return np.maximum(leg, floor)


if __name__ == "__main__":
    from data import make_synthetic

    cfg = default_config()
    df = make_synthetic(2000, seed=cfg.data.synthetic_seed)

    raw = detect_swings(df, cfg)
    alt = alternate_swings(raw)
    leg = build_leg_ranges(df, cfg)

    print("=== swings.py self-test: make_synthetic(2000) ===")
    print("raw swings        :", len(raw))
    print("alternating swings:", len(alt))
    kinds = [k for _, k, _ in alt]
    # Verify strict alternation.
    alternates = all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1))
    print("strictly alternates:", alternates)
    print("leg_range min/mean/max: %.5f / %.5f / %.5f"
          % (leg.min(), leg.mean(), leg.max()))
    assert leg.min() > 0, "leg_range floor failed (found <= 0)"
    assert len(leg) == len(df), "leg_range length mismatch"
    assert alternates, "swings not strictly alternating"
    assert 10 <= len(alt) <= 1000, f"unexpected swing count: {len(alt)}"
    print("OK — all assertions passed")
