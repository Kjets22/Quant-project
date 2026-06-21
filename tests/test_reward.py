import numpy as np

from config import default_config
from data import make_synthetic
from reward import CaptureReward
from swings import alternate_swings, build_leg_ranges, detect_swings


def test_perfect_long_capture_near_one():
    cfg = default_config()
    df = make_synthetic(2000, seed=7)
    close = df["close"].to_numpy()
    leg = build_leg_ranges(df, cfg)
    rw = CaptureReward(close, leg, cfg)

    swings = alternate_swings(detect_swings(df, cfg))
    up = next(((a[0], b[0]) for a, b in zip(swings[:-1], swings[1:])
               if a[1] == "L" and b[1] == "H"), None)
    assert up is not None
    lo, hi = up
    total, prev = 0.0, 0
    for t in range(lo, hi):
        total += rw.step_reward(t, +1, prev)
        prev = +1
    assert 0.3 <= total <= 2.0


def test_reward_is_clipped_and_finite():
    cfg = default_config()
    prices = np.array([100.0, 1e6, 1.0, 100.0])  # extreme jumps
    leg = np.array([1.0, 1.0, 1.0, 1.0])         # tiny range -> would explode
    rw = CaptureReward(prices, leg, cfg)
    for t in range(len(prices)):
        for pos in (-1, 0, 1):
            r = rw.step_reward(t, pos, 0)
            assert np.isfinite(r)
            assert -cfg.reward.reward_clip <= r <= cfg.reward.reward_clip


def test_flat_bonus_applied():
    cfg = default_config()
    prices = np.array([100.0, 100.0, 100.0])
    leg = np.array([5.0, 5.0, 5.0])
    rw = CaptureReward(prices, leg, cfg)
    # Flat position with no price move -> exactly the flat bonus.
    assert abs(rw.step_reward(0, 0, 0) - cfg.reward.flat_bonus) < 1e-9
