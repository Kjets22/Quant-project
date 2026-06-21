"""
lockbox.py — Phase A: the final hold-out and its access guard.

THE LOCKBOX is the most recent LOCKBOX_MONTHS of every ticker. It is NEVER used
for training, tuning, feature/model selection, or any decision during Phases
A–E. It is opened exactly ONCE, in Phase F, on the single final model.

The guard enforces this in code:
  * `dev(df)` returns the pre-cutoff slice and is always allowed.
  * `open_lockbox(dfs, phase)` returns the post-cutoff slice ONLY when phase=="F"
    and ONLY once per run; any other phase or a second call raises
    LockboxViolation. The lockbox date range is hashed and persisted to
    runs/lockbox.json so the boundary cannot silently drift.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from basket import BASKET_TICKERS, LOCKBOX_MONTHS, ticker_cfg
from data import load_from_cache

RUNS_DIR = Path(__file__).with_name("runs")
LOCKBOX_JSON = RUNS_DIR / "lockbox.json"


class LockboxViolation(Exception):
    """Raised on any attempt to read the lockbox outside its sanctioned use."""


def _range_hash(cutoff: str, end: str) -> str:
    return hashlib.sha256(f"{cutoff}|{end}".encode("utf-8")).hexdigest()


@dataclass
class Lockbox:
    cutoff: str          # ISO timestamp; lockbox = bars with timestamp >= cutoff
    end: str             # ISO timestamp of the last available bar
    hash: str
    _opened: bool = False

    # ---- construction --------------------------------------------------- #
    @classmethod
    def build(cls, tickers: list[str] = BASKET_TICKERS,
              months: int = LOCKBOX_MONTHS) -> "Lockbox":
        """Compute the lockbox boundary from the cached data and persist it."""
        last_timestamps = []
        for t in tickers:
            df = load_from_cache(ticker_cfg(t))
            if df is None:
                raise FileNotFoundError(f"No cache for {t}; run fetch_data.py --basket")
            last_timestamps.append(pd.Timestamp(df["timestamp"].iloc[-1]))
        end = max(last_timestamps)
        cutoff = end - pd.DateOffset(months=months)
        lb = cls(cutoff=cutoff.isoformat(), end=end.isoformat(),
                 hash=_range_hash(cutoff.isoformat(), end.isoformat()))
        lb.save()
        return lb

    @classmethod
    def load(cls, path: Path = LOCKBOX_JSON) -> "Lockbox":
        d = json.loads(path.read_text(encoding="utf-8"))
        expected = _range_hash(d["cutoff"], d["end"])
        if d["hash"] != expected:
            raise LockboxViolation("lockbox.json hash mismatch — boundary tampered.")
        return cls(cutoff=d["cutoff"], end=d["end"], hash=d["hash"])

    @classmethod
    def load_or_build(cls, tickers: list[str] = BASKET_TICKERS) -> "Lockbox":
        if LOCKBOX_JSON.exists():
            return cls.load()
        return cls.build(tickers)

    def save(self, path: Path = LOCKBOX_JSON) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"cutoff": self.cutoff, "end": self.end, "hash": self.hash},
                       indent=2),
            encoding="utf-8",
        )

    # ---- access --------------------------------------------------------- #
    @property
    def cutoff_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.cutoff)

    def dev(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dev-set slice (everything BEFORE the lockbox). Always allowed."""
        out = df[df["timestamp"] < self.cutoff_ts].reset_index(drop=True)
        return out

    def open_lockbox(self, dfs: dict[str, pd.DataFrame], phase: str) -> dict[str, pd.DataFrame]:
        """
        Return the lockbox slice for each ticker. Allowed ONLY in Phase F and
        ONLY once per process. Any other use raises LockboxViolation.
        """
        if phase != "F":
            raise LockboxViolation(
                f"Lockbox access denied in phase {phase!r}; only Phase F may open it."
            )
        if self._opened:
            raise LockboxViolation(
                "Lockbox already opened once this run; refusing a second access."
            )
        self._opened = True
        return {
            t: df[df["timestamp"] >= self.cutoff_ts].reset_index(drop=True)
            for t, df in dfs.items()
        }


def load_dev_ticker(ticker: str, lockbox: Lockbox) -> pd.DataFrame:
    """Load a ticker's cached data and return ONLY its dev-set slice."""
    df = load_from_cache(ticker_cfg(ticker))
    if df is None:
        raise FileNotFoundError(f"No cache for {ticker}; run fetch_data.py --basket")
    return lockbox.dev(df)


if __name__ == "__main__":
    lb = Lockbox.load_or_build()
    print("=== lockbox.py self-test ===")
    print("cutoff :", lb.cutoff)
    print("end    :", lb.end)
    print("hash   :", lb.hash[:16], "...")
    for t in BASKET_TICKERS:
        full = load_from_cache(ticker_cfg(t))
        dev = lb.dev(full)
        lock_n = len(full) - len(dev)
        print(f"  {t:5s}: full={len(full):>7d}  dev={len(dev):>7d}  lockbox={lock_n:>6d}")
    # Guard behavior.
    try:
        lb.open_lockbox({}, phase="A")
        print("GUARD FAIL: phase A access allowed!")
    except LockboxViolation:
        print("guard OK: phase-A access blocked")
