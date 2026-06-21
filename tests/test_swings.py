from config import default_config
from data import make_synthetic
from swings import alternate_swings, build_leg_ranges, detect_swings


def test_swings_alternate_and_floor():
    cfg = default_config()
    df = make_synthetic(2000, seed=7)
    alt = alternate_swings(detect_swings(df, cfg))
    kinds = [k for _, k, _ in alt]
    assert len(alt) >= 10
    assert all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1))

    leg = build_leg_ranges(df, cfg)
    assert len(leg) == len(df)
    assert leg.min() > 0  # floor works everywhere


def test_leg_range_fallback_when_too_few_swings():
    cfg = default_config()
    # Very short series -> fewer than 2 swings -> rolling fallback path.
    df = make_synthetic(20, seed=3)
    leg = build_leg_ranges(df, cfg)
    assert len(leg) == len(df)
    assert leg.min() > 0
