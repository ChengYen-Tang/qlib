# -*- coding: utf-8 -*-
"""
Lightweight order/position/balance structures to avoid Python-object overhead while
still being explicit. Internal math uses numpy/pandas; these dataclasses are just
carriers for state snapshots and logs (similar to C++ DTOs but much thinner).
"""

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Order:
    id: int
    symbol: str
    quantity: float
    price: float              # <= 0: market; > 0: limit
    is_long: bool
    reduce_only: bool
    closing_position_id: int  # -1 for opening
    ts_accepted: int

@dataclass
class Position:
    id: int
    order_id: int
    symbol: str
    quantity: float
    entry_price: float
    is_long: bool
    unrealized_pnl: float
    notional: float
    initial_margin: float
    maintenance_margin: float
    fee: float
    leverage: float
    fee_rate: float

@dataclass
class FillReport:
    order_id: int
    symbol: str
    qty: float
    price: float
    fee: float
    is_close: bool
    is_long: bool
    ts: int
