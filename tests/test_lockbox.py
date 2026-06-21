import pandas as pd
import pytest

from lockbox import Lockbox, LockboxViolation, _range_hash


def _make_df(start, periods):
    ts = pd.date_range(start, periods=periods, freq="5min")
    return pd.DataFrame({"timestamp": ts, "close": range(periods)})


def _guard():
    cutoff = pd.Timestamp("2025-06-01")
    end = pd.Timestamp("2026-06-01")
    return Lockbox(cutoff=cutoff.isoformat(), end=end.isoformat(),
                   hash=_range_hash(cutoff.isoformat(), end.isoformat()))


def test_dev_excludes_lockbox():
    lb = _guard()
    df = _make_df("2025-05-30", 2000)  # straddles the 2025-06-01 cutoff
    dev = lb.dev(df)
    assert (dev["timestamp"] < lb.cutoff_ts).all()
    assert len(dev) < len(df)  # some rows are in the lockbox


def test_lockbox_blocked_outside_phase_f():
    lb = _guard()
    dfs = {"X": _make_df("2025-05-30", 2000)}
    for bad_phase in ("A", "B", "C", "D", "E", "f", "G"):
        with pytest.raises(LockboxViolation):
            lb.open_lockbox(dfs, phase=bad_phase)


def test_lockbox_opens_once_in_phase_f():
    lb = _guard()
    dfs = {"X": _make_df("2025-05-30", 2000)}
    out = lb.open_lockbox(dfs, phase="F")            # first access OK
    assert (out["X"]["timestamp"] >= lb.cutoff_ts).all()
    with pytest.raises(LockboxViolation):            # second access blocked
        lb.open_lockbox(dfs, phase="F")


def test_hash_detects_tampering(tmp_path):
    lb = _guard()
    path = tmp_path / "lockbox.json"
    lb.save(path)
    # Corrupt the cutoff without fixing the hash.
    import json
    d = json.loads(path.read_text())
    d["cutoff"] = "2020-01-01T00:00:00"
    path.write_text(json.dumps(d))
    with pytest.raises(LockboxViolation):
        Lockbox.load(path)
