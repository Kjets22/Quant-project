"""
basket.py — Phase A: the multi-ticker, multi-year universe and per-ticker config.

Centralizes the tickers, the full date range, and the lockbox horizon so every
Phase-2 module (fetch_data, lockbox, control) agrees on exactly the same data.
"""

from __future__ import annotations

import copy

from config import Config, default_config

# Active training universe. Currently SPY + QQQ only (per request).
# (Full sector basket available: SPY, AAPL, MSFT, NVDA, JPM, XLE.)
BASKET_TICKERS: list[str] = ["SPY", "QQQ"]

# Full available range. The most recent LOCKBOX_MONTHS are reserved as lockbox;
# everything before that is the dev set used for all training/tuning.
BASKET_START = "2021-06-01"
BASKET_END = "2026-06-01"
LOCKBOX_MONTHS = 12


def ticker_cfg(ticker: str, base: Config | None = None) -> Config:
    """A deep copy of the default config pointed at one ticker over the full range."""
    cfg = copy.deepcopy(base) if base is not None else default_config()
    cfg.data.ticker = ticker
    cfg.data.start_date = BASKET_START
    cfg.data.end_date = BASKET_END
    return cfg
