# -*- coding: utf-8 -*-
"""
Account engine: balance, positions, leverage control, MTM.
Order placement is done in exchange.py; account focuses on state transitions.
"""

from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass, field

import numpy as np

from .config import get_tier_info
from .types import Position


@dataclass
class AccountSnapshot:
    balance: float
    used_margin: float
    unreal_pnl: float
    equity: float

@dataclass
class Account:
    init_balance: float
    vip_level: int = 0
    hedge_mode: bool = False

    balance: float = field(init=False)
    used_margin: float = field(init=False)
    _next_pos_id: int = field(init=False, default=1)

    def __post_init__(self):
        self.balance = float(self.init_balance)
        self.used_margin = 0.0
        self._positions: List[Position] = []
        self._np_cache: Dict[str, np.ndarray] = {}
        self._symbol_index: List[str] = []
        self._np_dirty: bool = True

    # ------------ internal helpers ------------
    def _invalidate_cache(self) -> None:
        self._np_cache = {}
        self._symbol_index = []
        self._np_dirty = True

    def _refresh_cache(self) -> None:
        if not self._np_dirty:
            return

        count = len(self._positions)
        if count == 0:
            self._np_cache = {
                "quantity": np.empty(0, dtype=np.float64),
                "entry_price": np.empty(0, dtype=np.float64),
                "direction": np.empty(0, dtype=np.float64),
                "initial_margin": np.empty(0, dtype=np.float64),
                "maintenance_margin": np.empty(0, dtype=np.float64),
                "fee": np.empty(0, dtype=np.float64),
                "unrealized_pnl": np.empty(0, dtype=np.float64),
            }
            self._symbol_index = []
            self._np_dirty = False
            return

        qty = np.empty(count, dtype=np.float64)
        entry = np.empty(count, dtype=np.float64)
        direction = np.empty(count, dtype=np.float64)
        init_margin = np.empty(count, dtype=np.float64)
        maint_margin = np.empty(count, dtype=np.float64)
        fee = np.empty(count, dtype=np.float64)
        unreal = np.empty(count, dtype=np.float64)
        symbols: List[str] = []

        for idx, p in enumerate(self._positions):
            qty[idx] = p.quantity
            entry[idx] = p.entry_price
            direction[idx] = 1.0 if p.is_long else -1.0
            init_margin[idx] = p.initial_margin
            maint_margin[idx] = p.maintenance_margin
            fee[idx] = p.fee
            unreal[idx] = p.unrealized_pnl
            symbols.append(p.symbol)

        self._np_cache = {
            "quantity": qty,
            "entry_price": entry,
            "direction": direction,
            "initial_margin": init_margin,
            "maintenance_margin": maint_margin,
            "fee": fee,
            "unrealized_pnl": unreal,
        }
        self._symbol_index = symbols
        self._np_dirty = False

    # ------------ query ------------
    def get_all_positions(self) -> List[Position]:
        return self._positions

    def snapshot(self) -> AccountSnapshot:
        unreal = self.total_unrealized_pnl()
        return AccountSnapshot(
            balance=self.balance,
            used_margin=self.used_margin,
            unreal_pnl=unreal,
            equity=self.balance + unreal,
        )

    # ------------ PnL metrics ------------
    def total_unrealized_pnl(self) -> float:
        self._refresh_cache()
        unreal = self._np_cache.get("unrealized_pnl")
        if unreal is None or unreal.size == 0:
            return 0.0
        return float(np.sum(unreal))

    def equity(self) -> float:
        return self.balance + self.total_unrealized_pnl()

    # ------------ core ops ------------
    def open_or_increase(
        self,
        order_id: int,
        symbol: str,
        fill_qty: float,
        fill_price: float,
        is_long: bool,
        lev: float,
        fee_rate: float,
    ) -> bool:
        """
        Open a new position or increase an existing one (same symbol+side).
        Enforce tier max leverage and initial margin requirement.
        """
        notional = fill_qty * fill_price
        mmr, max_lev = get_tier_info(notional)
        if lev > max_lev:
            # reject opening due to leverage cap
            return False

        init_margin = notional / lev
        maint_margin = notional * mmr
        fee = notional * fee_rate
        need = init_margin + fee

        if self.equity() < need:
            # not enough equity
            return False

        # pay margin + fee
        self.balance -= need
        self.used_margin += init_margin

        # find existing pos (same symbol & side)
        tgt: Position | None = None
        for p in self._positions:
            if p.symbol == symbol and p.is_long == is_long:
                tgt = p
                break

        if tgt is None:
            pid = self._next_pos_id
            self._next_pos_id += 1
            self._positions.append(Position(
                id=pid, order_id=order_id, symbol=symbol,
                quantity=fill_qty, entry_price=fill_price, is_long=is_long,
                unrealized_pnl=0.0, notional=notional,
                initial_margin=init_margin, maintenance_margin=maint_margin,
                fee=fee, leverage=lev, fee_rate=fee_rate
            ))
        else:
            old_notional = tgt.notional
            new_notional = old_notional + notional
            new_qty = tgt.quantity + fill_qty
            new_entry = new_notional / new_qty

            tgt.quantity = new_qty
            tgt.entry_price = new_entry
            tgt.notional = new_notional
            tgt.initial_margin += init_margin
            tgt.maintenance_margin += maint_margin
            tgt.fee += fee

        self._invalidate_cache()
        return True

    def adjust_symbol_leverage(self, symbol: str, old_lev: float, new_lev: float) -> bool:
        """Rebalance margin for existing positions when leverage changes.

        Returns True if adjustment succeeded or there was nothing to do.
        Mimics C++ adjust_position_leverage semantics.
        """
        if new_lev <= 0:
            return False

        related: List[Position] = [
            p for p in self._positions
            if p.symbol == symbol and p.quantity > 1e-12
        ]
        if not related:
            return True

        total_diff = 0.0
        new_initial: List[float] = []
        new_maint: List[float] = []
        for p in related:
            mmr, max_lev = get_tier_info(p.notional)
            if new_lev > max_lev:
                return False
            target_initial = p.notional / new_lev
            total_diff += target_initial - p.initial_margin
            new_initial.append(target_initial)
            new_maint.append(p.notional * mmr)

        if total_diff > 0.0:
            if self.equity() < total_diff:
                return False
            self.balance -= total_diff
            self.used_margin += total_diff
        elif total_diff < 0.0:
            adj = -total_diff
            self.balance += adj
            self.used_margin -= adj

        for p, init_val, maint_val in zip(related, new_initial, new_maint):
            p.initial_margin = init_val
            p.maintenance_margin = maint_val
            p.leverage = new_lev

        self._invalidate_cache()
        return True

    def reduce_or_close(self, position_id: int, close_qty: float, price: float, fee: float) -> bool:
        """
        Close by position id (used by explicit closing orders).
        Realize PnL, free margins proportional to closed ratio.
        """
        for p in list(self._positions):
            if p.id != position_id:
                continue
            if close_qty <= 1e-12:
                return True
            close_qty = min(close_qty, p.quantity)

            realized = (price - p.entry_price) * close_qty * (1.0 if p.is_long else -1.0)
            ratio = close_qty / p.quantity

            freed_init = p.initial_margin * ratio
            freed_maint = p.maintenance_margin * ratio
            freed_fee  = p.fee * ratio

            returned = freed_init + realized - fee  # exactly like your C++
            self.balance += returned
            self.used_margin -= freed_init

            p.quantity -= close_qty
            p.initial_margin -= freed_init
            p.maintenance_margin -= freed_maint
            p.fee -= freed_fee
            p.notional = p.entry_price * p.quantity
            if p.quantity <= 1e-12:
                self._positions.remove(p)
            self._invalidate_cache()
            return True
        # position not found
        return False

    # ------------ NEW: 1:1 reduce_only ------------
    def reduce_only(self, symbol: str, is_long: bool, fill_qty: float, price: float, fee: float) -> bool:
        """
        Match your C++ processReduceOnlyOrder():
        - Only reduces an EXISTING position with SAME direction (symbol & is_long).
        - Apply proportional release of margins & fees; realize PnL; update balance and used_margin.
        - If no such position exists, return False (caller may keep order open).
        """
        for p in list(self._positions):
            if p.symbol == symbol and p.is_long == is_long:
                if fill_qty <= 1e-12 or p.quantity <= 1e-12:
                    return True
                exec_qty = min(fill_qty, p.quantity)

                realized = (price - p.entry_price) * exec_qty * (1.0 if p.is_long else -1.0)
                ratio = exec_qty / p.quantity

                freed_init = p.initial_margin * ratio
                freed_maint = p.maintenance_margin * ratio
                freed_fee  = p.fee * ratio

                returned = freed_init + realized - fee  # identical formula to C++
                self.balance += returned
                self.used_margin -= freed_init

                p.quantity -= exec_qty
                p.initial_margin -= freed_init
                p.maintenance_margin -= freed_maint
                p.fee -= freed_fee
                p.notional = p.entry_price * p.quantity
                if p.quantity <= 1e-12:
                    self._positions.remove(p)
                self._invalidate_cache()
                return True
        return False  # no same-side position to reduce

    # ------------ MTM & risk ------------
    def mark_to_market(self, mtm_price: Dict[str, float]) -> None:
        if not self._positions:
            return

        self._refresh_cache()
        prices = np.array([mtm_price.get(sym, np.nan) for sym in self._symbol_index], dtype=np.float64)
        mask = ~np.isnan(prices)
        if not np.any(mask):
            return

        qty = self._np_cache["quantity"]
        entry = self._np_cache["entry_price"]
        direction = self._np_cache["direction"]
        updated = (prices[mask] - entry[mask]) * qty[mask] * direction[mask]

        unreal = self._np_cache["unrealized_pnl"]
        unreal[mask] = updated

        indices = np.nonzero(mask)[0]
        for idx, value in zip(indices, updated):
            self._positions[idx].unrealized_pnl = float(value)

        # positions without price keep prior unrealized_pnl values

    def check_liquidation(self) -> None:
        if not self._positions:
            return

        self._refresh_cache()
        total_maint = float(np.sum(self._np_cache["maintenance_margin"]))
        if self.equity() < total_maint:
            # hard liquidation (same as your C++)
            self.balance = 0.0
            self.used_margin = 0.0
            self._positions.clear()
            self._invalidate_cache()
