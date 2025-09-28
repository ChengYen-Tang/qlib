from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import math

from ..core.dto import Order, Position, Trade
from ..core.config import VIP_FEE_RATES, MARGIN_TIERS


class Account:
    """
    Binance Futures account simulator (one-way or hedge).
    Differences vs. your C++:
    - Fixes the "multi-orders overfill bar capacity" by enforcing a per-bar capacity bucket per symbol.
    - Per-bar MTM using bar close.
    - Leverage upper bound: reject opening orders if user leverage > tier max or equity insufficient.
    """

    def __init__(self, initial_balance: float, vip_level: int = 0, hedge_mode: bool = False):
        self.balance: float = initial_balance
        self.used_margin: float = 0.0
        self.vip_level: int = vip_level
        self.hedge_mode: bool = hedge_mode

        self.symbol_leverage: Dict[str, float] = {}  # default 1.0
        self._next_order_id: int = 1
        self._next_position_id: int = 1
        self._next_trade_id: int = 1

        self.open_orders: List[Order] = []
        self.positions: List[Position] = []
        self.order_to_position: Dict[int, int] = {}  # order_id -> position_id

        # Recorder buffer for trades (optional, can be read by recorder)
        self.executed_trades: List[Trade] = []

        # Per-bar capacity bucket: symbol -> remaining volume for this bar
        self._bar_capacity: Dict[str, float] = {}
        self._bar_time: Optional[int] = None  # ms epoch of current bar

    # ---------- Public getters ----------

    def get_balance(self) -> float:
        return self.balance

    def total_unrealized_pnl(self) -> float:
        total = 0.0
        for p in self.positions:
            total += p.unrealized_pnl
        return total

    def get_equity(self) -> float:
        return self.balance + self.total_unrealized_pnl()

    def get_all_open_orders(self) -> List[Order]:
        return self.open_orders

    def get_all_positions(self) -> List[Position]:
        return self.positions

    # ---------- Leverage settings ----------

    def set_position_mode(self, hedge_mode: bool) -> None:
        if self.positions:
            # Do not allow switching while positions exist
            return
        self.hedge_mode = hedge_mode

    def is_hedge_mode(self) -> bool:
        return self.hedge_mode

    def set_symbol_leverage(self, symbol: str, new_leverage: float) -> None:
        if new_leverage <= 0:
            raise RuntimeError("Leverage must be > 0.")
        old = self.symbol_leverage.get(symbol, 1.0)
        # Adjust existing positions if any (simplified: require enough equity)
        ok = self._adjust_position_leverage(symbol, old, new_leverage)
        if ok:
            self.symbol_leverage[symbol] = new_leverage

    def get_symbol_leverage(self, symbol: str) -> float:
        return self.symbol_leverage.get(symbol, 1.0)

    # ---------- ID gens ----------

    def _gen_order_id(self) -> int:
        oid = self._next_order_id
        self._next_order_id += 1
        return oid

    def _gen_position_id(self) -> int:
        pid = self._next_position_id
        self._next_position_id += 1
        return pid

    def _gen_trade_id(self) -> int:
        tid = self._next_trade_id
        self._next_trade_id += 1
        return tid

    # ---------- Order API ----------

    def place_order(self, symbol: str, quantity: float, price: float, is_long: bool, reduce_only: bool = False) -> None:
        if quantity <= 0:
            return
        # one-way reverse handling
        if not self.hedge_mode and self._handle_oneway_reverse(symbol, quantity, price, is_long):
            return
        oid = self._gen_order_id()
        self.open_orders.append(
            Order(
                id=oid, symbol=symbol, quantity=quantity, price=price,
                is_long=is_long, reduce_only=reduce_only, closing_position_id=-1,
                created_at=None
            )
        )

    def place_order_mkt(self, symbol: str, quantity: float, is_long: bool, reduce_only: bool = False) -> None:
        self.place_order(symbol, quantity, 0.0, is_long, reduce_only)

    def close_position(self, symbol: str, price: float = 0.0) -> None:
        found = False
        for pos in list(self.positions):
            if pos.symbol == symbol:
                found = True
                self._place_closing_order(pos.id, pos.quantity, price)
        # if not found: ignore (consistent with C++ behavior)

    def close_position_one_side(self, symbol: str, is_long: bool, price: float = 0.0) -> None:
        found = False
        for pos in list(self.positions):
            if pos.symbol == symbol and pos.is_long == is_long:
                found = True
                self._place_closing_order(pos.id, pos.quantity, price)

    def cancel_order_by_id(self, order_id: int) -> None:
        self.open_orders = [o for o in self.open_orders if o.id != order_id]

    # ---------- Core matching ----------

    def update_positions(
        self,
        ts_ms: int,
        symbol_price_volume: Dict[str, Tuple[float, float]],
        # if ts_ms changes, reset capacity buckets to this bar's volumes
    ) -> None:
        maker_fee, taker_fee = self._fee_rates()

        # Reset per-bar capacity buckets if we enter a new bar
        if self._bar_time != ts_ms:
            self._bar_capacity = {sym: vol for sym, (_, vol) in symbol_price_volume.items()}
            self._bar_time = ts_ms

        leftover: List[Order] = []

        # Process orders in FIFO; enforce per-symbol bar capacity (100% of Volume)
        for ord in self.open_orders:
            pv = symbol_price_volume.get(ord.symbol)
            if pv is None:
                leftover.append(ord)
                continue
            current_price, bar_volume = pv
            remaining_capacity = max(0.0, self._bar_capacity.get(ord.symbol, 0.0))
            if remaining_capacity <= 0.0:
                # No capacity left for this bar
                leftover.append(ord)
                continue

            # Limit/market check
            is_mkt = ord.price <= 0.0
            can_fill = is_mkt or (
                (ord.is_long and current_price <= ord.price) or
                (not ord.is_long and current_price >= ord.price)
            )
            if not can_fill:
                leftover.append(ord)
                continue

            # Fill quantity capped by both order qty and remaining capacity
            fill_qty = min(ord.quantity, remaining_capacity)
            if fill_qty <= 1e-12:
                leftover.append(ord)
                continue

            fill_price = current_price
            notional = fill_qty * fill_price
            fee_rate = taker_fee if is_mkt else maker_fee
            fee = notional * fee_rate

            if ord.closing_position_id >= 0:
                # Closing fill
                self._process_closing(ord, fill_qty, fill_price, fee)
            else:
                # Opening or reduce-only opening
                self._process_opening(ord, fill_qty, fill_price, notional, fee, fee_rate)

            # Reduce bar capacity
            self._bar_capacity[ord.symbol] = remaining_capacity - fill_qty

            # Remainder to next bars
            if ord.quantity > 1e-12:
                leftover.append(ord)

        # Swap open orders
        self.open_orders = leftover

        # Clean empty positions
        self.positions = [p for p in self.positions if p.quantity > 1e-12]

        # Merge positions with same (symbol, side)
        self._merge_positions()

        # Per-bar MTM (use bar close)
        for p in self.positions:
            cp = symbol_price_volume.get(p.symbol, (p.entry_price, 0.0))[0]
            p.unrealized_pnl = (cp - p.entry_price) * p.quantity * (1.0 if p.is_long else -1.0)

        # Liquidation check: equity < sum maintenance → nuke
        equity = self.get_equity()
        total_maint = sum(p.maintenance_margin for p in self.positions)
        if equity < total_maint - 1e-12:
            # Hard liquidation
            self.balance = 0.0
            self.used_margin = 0.0
            self.positions.clear()
            self.open_orders.clear()
            self.order_to_position.clear()

    # ---------- Internals ----------

    def _fee_rates(self) -> Tuple[float, float]:
        fr = VIP_FEE_RATES.get(self.vip_level, VIP_FEE_RATES[0])
        return (fr.maker_fee_rate, fr.taker_fee_rate)

    def _get_tier_info(self, notional: float) -> Tuple[float, float]:
        """Return (maintenance_margin_rate, max_leverage) for the given notional."""
        for t in MARGIN_TIERS:
            if notional <= t.notional_upper:
                return (t.maintenance_margin_rate, t.max_leverage)
        return (MARGIN_TIERS[0].maintenance_margin_rate, MARGIN_TIERS[0].max_leverage)

    def _place_closing_order(self, position_id: int, quantity: float, price: float):
        # Create a synthetic closing order (reduce_only semantics for the specific position)
        pos = next((p for p in self.positions if p.id == position_id), None)
        if pos is None:
            return
        oid = self._gen_order_id()
        self.open_orders.append(
            Order(
                id=oid, symbol=pos.symbol, quantity=quantity, price=price,
                is_long=not pos.is_long, reduce_only=False, closing_position_id=position_id
            )
        )

    def _handle_oneway_reverse(self, symbol: str, quantity: float, price: float, is_long: bool) -> bool:
        # If an opposite position exists, close it first (and possibly open the reverse remainder)
        for pos in self.positions:
            if pos.symbol == symbol:
                if pos.is_long == is_long:
                    return False  # same direction → additive
                # Reverse logic:
                if quantity <= pos.quantity + 1e-12:
                    self._place_closing_order(pos.id, quantity, price)
                    return True
                else:
                    # Close existing fully, then open remainder as new order
                    self._place_closing_order(pos.id, pos.quantity, price)
                    remainder = max(0.0, quantity - pos.quantity)
                    oid = self._gen_order_id()
                    self.open_orders.append(
                        Order(
                            id=oid, symbol=symbol, quantity=remainder, price=price,
                            is_long=is_long, reduce_only=False, closing_position_id=-1
                        )
                    )
                    return True
        return False

    def _process_closing(self, ord: Order, fill_qty: float, fill_price: float, fee: float):
        # Find position
        pos = next((p for p in self.positions if p.id == ord.closing_position_id), None)
        if pos is None:
            return  # nothing to do; keep order alive for diagnostics if needed

        close_qty = min(fill_qty, pos.quantity)
        realized_pnl = (fill_price - pos.entry_price) * close_qty * (1.0 if pos.is_long else -1.0)
        ratio = close_qty / pos.quantity if pos.quantity > 0 else 1.0

        freed_init = pos.initial_margin * ratio
        freed_maint = pos.maintenance_margin * ratio
        freed_fee = pos.fee * ratio

        returned = freed_init + realized_pnl - fee

        self.balance += returned
        self.used_margin -= freed_init

        pos.quantity -= close_qty
        pos.initial_margin -= freed_init
        pos.maintenance_margin -= freed_maint
        pos.fee -= freed_fee
        pos.notional = pos.entry_price * pos.quantity

        ord.quantity -= close_qty

        # Record trade
        self.executed_trades.append(
            Trade(
                datetime=datetime.utcfromtimestamp((self._bar_time or 0) / 1000.0),
                order_id=ord.id,
                trade_id=self._gen_trade_id(),
                symbol=ord.symbol,
                side="buy" if ord.is_long else "sell",
                price=fill_price,
                qty=close_qty,
                value=fill_price * close_qty,
                commission=fee,
                position_effect="close",
                reduce_only=False,
                tif="GTC",
                leverage=pos.leverage,
                fee_rate=pos.fee_rate,
                realized_pnl=realized_pnl
            )
        )

    def _process_opening(self, ord: Order, fill_qty: float, fill_price: float, notional: float, fee: float, fee_rate: float):
        if ord.reduce_only:
            # Reduce-only opening means: only reduce same-side position (hedge semantics)
            # If no same-side position, ignore (no open).
            for pos in self.positions:
                if pos.symbol == ord.symbol and pos.is_long == ord.is_long:
                    # Treat as partial close of same-side position (cost basis preserved)
                    self._process_closing(
                        Order(
                            id=ord.id, symbol=ord.symbol, quantity=fill_qty, price=fill_price,
                            is_long=(not pos.is_long), reduce_only=False, closing_position_id=pos.id
                        ),
                        fill_qty, fill_price, fee
                    )
                    # Any remainder stays as leftover via ord.quantity update below
                    break
            # If not found, do nothing (i.e., we won't open)
            ord.quantity -= fill_qty
            if ord.quantity < 0:
                ord.quantity = 0.0
            return

        # Normal opening validation
        lev = self.get_symbol_leverage(ord.symbol)
        mmr, max_lev = self._get_tier_info(notional)
        if lev > max_lev + 1e-12:
            # Reject opening when leverage exceeds tier max
            return  # keep order as leftover for later bars if you want; here we simply do not fill
        init_margin = notional / max(lev, 1e-12)
        maint_margin = notional * mmr
        required = init_margin + fee
        if self.get_equity() < required - 1e-12:
            # Not enough equity → reject open this bar
            return

        # Reserve cash/margin
        self.balance -= required
        self.used_margin += init_margin

        # Put into existing or new position (map by original order id)
        pid = self.order_to_position.get(ord.id)
        if pid is None:
            pid = self._gen_position_id()
            self.positions.append(
                Position(
                    id=pid, order_id=ord.id, symbol=ord.symbol, quantity=fill_qty,
                    entry_price=fill_price, is_long=ord.is_long, unrealized_pnl=0.0,
                    notional=notional, initial_margin=init_margin, maintenance_margin=maint_margin,
                    fee=fee, leverage=lev, fee_rate=fee_rate
                )
            )
            self.order_to_position[ord.id] = pid
        else:
            for p in self.positions:
                if p.id == pid:
                    old_notional = p.notional
                    new_notional = old_notional + notional
                    old_qty = p.quantity
                    new_qty = old_qty + fill_qty
                    new_entry = new_notional / max(new_qty, 1e-12)

                    p.quantity = new_qty
                    p.entry_price = new_entry
                    p.notional = new_notional
                    p.initial_margin += init_margin
                    p.maintenance_margin += maint_margin
                    p.fee += fee
                    break

        ord.quantity -= fill_qty
        if ord.quantity < 0:
            ord.quantity = 0.0

        # Record trade
        self.executed_trades.append(
            Trade(
                datetime=datetime.utcfromtimestamp((self._bar_time or 0) / 1000.0),
                order_id=ord.id,
                trade_id=self._gen_trade_id(),
                symbol=ord.symbol,
                side="buy" if ord.is_long else "sell",
                price=fill_price,
                qty=fill_qty,
                value=fill_price * fill_qty,
                commission=fee,
                position_effect="open",
                reduce_only=False,
                tif="GTC",
                leverage=lev,
                fee_rate=fee_rate,
                realized_pnl=0.0
            )
        )

    def _merge_positions(self):
        if not self.positions:
            return
        key_map: Dict[Tuple[str, bool], Position] = {}
        for p in self.positions:
            key = (p.symbol, p.is_long)
            if key not in key_map:
                key_map[key] = p
            else:
                t = key_map[key]
                total_qty = t.quantity + p.quantity
                if total_qty <= 1e-12:
                    t.quantity = 0.0
                    continue
                weighted_entry = (t.entry_price * t.quantity + p.entry_price * p.quantity) / total_qty
                t.quantity = total_qty
                t.entry_price = weighted_entry
                t.notional += p.notional
                t.initial_margin += p.initial_margin
                t.maintenance_margin += p.maintenance_margin
                t.fee += p.fee
        self.positions = [v for v in key_map.values() if v.quantity > 1e-12]

    def _adjust_position_leverage(self, symbol: str, old: float, new: float) -> bool:
        # Adjust leverage for all positions of this symbol
        related = [p for p in self.positions if p.symbol == symbol]
        if not related:
            return True
        total_diff = 0.0
        new_maint: List[float] = []
        for p in related:
            mmr, max_lev = self._get_tier_info(p.notional)
            if new > max_lev + 1e-12:
                return False
            old_m = p.initial_margin
            new_m = p.notional / max(new, 1e-12)
            diff = new_m - old_m
            total_diff += diff
            new_maint.append(p.notional * mmr)
        if total_diff > 0.0:
            if self.get_equity() < total_diff - 1e-12:
                return False
            self.balance -= total_diff
            self.used_margin += total_diff
        else:
            self.balance += abs(total_diff)
            self.used_margin -= abs(total_diff)
        for p, m in zip(related, new_maint):
            p.initial_margin = p.notional / max(new, 1e-12)
            p.leverage = new
            p.maintenance_margin = m
        return True
