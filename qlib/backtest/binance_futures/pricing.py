# qlib/backtest/binance/pricing.py
# -*- coding: utf-8 -*-
"""
Execution price models for bar-level matching.
Given a bar snapshot dict {"close": float, "_row": pd.Series-like}, return an execution price.
We support separate models for market (taker) and crossed limit (maker-like).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Dict, Any


MarketPriceModel = Literal["CLOSE", "BAR_VWAP", "TAKER_VWAP"]
LimitPriceModel  = Literal["CLOSE", "BAR_VWAP", "TAKER_VWAP", "CROSS_AT_LIMIT", "BOUNDED_VWAP"]

@dataclass
class ExecModel:
    market: MarketPriceModel = "TAKER_VWAP"   # default for takers
    limit:  LimitPriceModel  = "BOUNDED_VWAP" # default for crossed limits

def _bar_vwap(bar: Dict[str, Any]) -> float:
    row = bar.get("_row")
    vol = float(row["volume"])
    qv  = float(row["quote_volume"])
    return (qv / vol) if vol > 0 else float(row["close"])

def _taker_vwap(bar: Dict[str, Any]) -> float:
    row = bar.get("_row")
    tb  = float(row.get("taker_base", 0.0))
    tq  = float(row.get("taker_quote", 0.0))
    if tb > 0:
        return tq / tb
    # fallback to bar vwap if taker fields are empty
    return _bar_vwap(bar)

def price_for_market(bar: Dict[str, Any], model: MarketPriceModel) -> float:
    if model == "CLOSE":
        return float(bar["close"])
    if model == "BAR_VWAP":
        return _bar_vwap(bar)
    if model == "TAKER_VWAP":
        return _taker_vwap(bar)
    # fallback
    return float(bar["close"])

def price_for_crossed_limit(
    bar: Dict[str, Any],
    model: LimitPriceModel,
    limit_price: float
) -> float:
    c = float(bar["close"])
    if model == "CROSS_AT_LIMIT":
        return float(limit_price)
    if model == "CLOSE":
        return c
    if model == "BAR_VWAP":
        return _bar_vwap(bar)
    if model == "TAKER_VWAP":
        return _taker_vwap(bar)
    if model == "BOUNDED_VWAP":
        v = _bar_vwap(bar)
        # clamp between limit and close (prevents crazy overshoot with only OHLC)
        low = min(c, float(limit_price))
        high = max(c, float(limit_price))
        # clamp vwap into [low, high]
        return min(max(v, low), high)
    # fallback
    return c
