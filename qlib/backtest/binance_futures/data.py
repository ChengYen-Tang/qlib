# -*- coding: utf-8 -*-
"""
CSV readers and in-memory stores using pandas.
All bars live in DataFrames:
- Single symbol file: columns = [open, high, low, close, volume, quote_volume, trades, taker_base, taker_quote]
- Index = pandas.DatetimeIndex (UTC); we also keep a 'ts' column in milliseconds for fast join
Multi-symbol drive:
- Either dict[symbol] -> DataFrame
- Or a single MultiIndex DataFrame [datetime, symbol]
"""

from __future__ import annotations
import pandas as pd
from typing import Dict, Iterable, Tuple

CSV_COLUMNS = [
    "open_ts", "open", "high", "low", "close",
    "volume", "close_ts", "quote_volume", "trades",
    "taker_base", "taker_quote",
]

KEEP_COLUMNS = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_base", "taker_quote", "ts"]

def read_symbol_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Read a single-symbol 1m kline CSV (same schema as your C++).
    Returns a DataFrame indexed by datetime, with a 'ts' (ms) column for fast use.
    """
    df = pd.read_csv(path, names=CSV_COLUMNS, header=0)
    # Ensure numeric dtypes
    num_cols = [c for c in CSV_COLUMNS if c not in ("open_ts", "close_ts")]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Sort by open_ts and build index
    df = df.sort_values("open_ts")
    dt = pd.to_datetime(df["open_ts"], unit="ms", utc=True)
    df.index = dt.tz_convert(tz)
    df["ts"] = df["open_ts"].astype("int64")
    return df[KEEP_COLUMNS].copy()

def load_market(symbol_csv: Iterable[Tuple[str, str]]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple symbols into a dict. Keys = symbol, values = DataFrame as defined above.
    """
    out: Dict[str, pd.DataFrame] = {}
    for sym, path in symbol_csv:
        out[sym] = read_symbol_csv(path)
    return out

def align_next_timestamp(cursor: Dict[str, int], data: Dict[str, pd.DataFrame]) -> int | None:
    """
    Find the next global timestamp (ms) across all symbols based on cursor.
    Returns None when all streams are exhausted.
    """
    next_ts = None
    for sym, df in data.items():
        i = cursor.get(sym, 0)
        if i < len(df):
            ts = int(df["ts"].iloc[i])
            next_ts = ts if next_ts is None else min(next_ts, ts)
    return next_ts

def slice_bar_snapshot(ts: int, cursor: Dict[str, int], data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, dict], Dict[str, int]]:
    """
    Build a per-symbol snapshot at timestamp ts:
    - If symbol has bar at ts, return its row dict and advance cursor.
    - Otherwise, return None for that symbol and keep cursor.
    Returns (snapshot, new_cursor)
    """
    snap, new_cursor = {}, dict(cursor)
    for sym, df in data.items():
        i = cursor.get(sym, 0)
        if i < len(df) and int(df["ts"].iloc[i]) == ts:
            row = df.iloc[i]
            snap[sym] = {
                "close": float(row["close"]),
                "volume": float(row["volume"]),  # base-asset volume
                # Keep full row if needed in the future:
                "_row": row,
            }
            new_cursor[sym] = i + 1
        else:
            snap[sym] = None
    return snap, new_cursor
