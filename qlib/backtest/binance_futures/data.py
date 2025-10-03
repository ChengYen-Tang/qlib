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
import numpy as np
import pandas as pd
from typing import Dict, Iterable, Tuple

CSV_COLUMNS = [
    "open_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_ts",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
]

KEEP_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_base",
    "taker_quote",
    "ts",
]


def normalize_symbol_df(df: pd.DataFrame, tz: str = "UTC") -> pd.DataFrame:
    """Ensure a symbol dataframe has the expected schema used by the exchange."""
    if df.empty:
        empty = pd.DataFrame(columns=KEEP_COLUMNS)
        empty.index = pd.DatetimeIndex([], tz="UTC")
        return empty

    norm = df.copy()

    def _series_to_array(series: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(series, pd.Series):
            arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
        else:
            arr = np.asarray(series, dtype="float64")
        return arr

    ts_candidates: list[np.ndarray] = []

    if "open_ts" in norm.columns:
        ts_candidates.append(_series_to_array(norm["open_ts"]))
    if "ts" in norm.columns:
        ts_candidates.append(_series_to_array(norm["ts"]))
    if isinstance(norm.index, pd.DatetimeIndex):
        idx = norm.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        ts_candidates.append((idx.view("int64") // 1_000_000).astype("float64"))

    if not ts_candidates:
        raise ValueError("symbol dataframe must contain 'open_ts'/'ts' column or datetime index")

    ts_array = ts_candidates[0]
    for candidate in ts_candidates[1:]:
        if candidate.shape != ts_array.shape:
            continue
        mask = ~np.isnan(candidate)
        ts_array[mask] = candidate[mask]

    if np.isnan(ts_array).any():
        raise ValueError("timestamp column contains NaN values")

    norm["open_ts"] = ts_array.copy()
    norm["ts"] = ts_array.astype("int64")

    if isinstance(norm.index, pd.DatetimeIndex):
        idx = norm.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_convert("UTC")
        idx = idx.tz_convert(tz)
    else:
        utc_index = pd.to_datetime(norm["ts"].to_numpy(), unit="ms", utc=True)
        idx = utc_index.tz_convert(tz)
    norm.index = pd.DatetimeIndex(idx)
    norm.index.name = None

    numeric_cols = [c for c in KEEP_COLUMNS if c != "ts"]
    for col in numeric_cols:
        if col in norm.columns:
            norm[col] = pd.to_numeric(norm[col], errors="coerce")

    norm = norm.sort_values(by="ts")
    missing = [c for c in KEEP_COLUMNS if c not in norm.columns]
    for col in missing:
        norm[col] = 0.0 if col != "ts" else norm["ts"]

    return norm[KEEP_COLUMNS].copy()

def read_symbol_csv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Read a single-symbol 1m kline CSV (same schema as your C++).
    Returns a DataFrame indexed by datetime, with a 'ts' (ms) column for fast use.
    """
    df = pd.read_csv(path, names=CSV_COLUMNS, header=0)
    return normalize_symbol_df(df, tz=tz)

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
