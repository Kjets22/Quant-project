import numpy as np

from data import make_synthetic


def test_synthetic_shape_and_validity():
    df = make_synthetic(500, seed=7)
    assert len(df) == 500
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    ohlc = df[["open", "high", "low", "close"]]
    assert (ohlc > 0).all().all()
    assert (df["high"] >= df["low"]).all()
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert not df.isna().any().any()


def test_synthetic_is_deterministic():
    a = make_synthetic(200, seed=1)
    b = make_synthetic(200, seed=1)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
