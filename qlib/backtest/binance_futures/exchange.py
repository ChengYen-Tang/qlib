from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .types import Order, Position, FillReport
from .account import Account
from .config import get_fee_rates
from .data import align_next_timestamp, slice_bar_snapshot
from .pricing import ExecModel, price_for_market, price_for_crossed_limit

import pandas as pd

@dataclass
class ExchangeConfig:
    init_balance: float = 1_000_000.0
    vip_level: int = 0
    hedge_mode: bool = False
    default_leverage: float = 1.0
    exec_model: ExecModel = field(default_factory=ExecModel)

class Exchange:
    def __init__(self, market_data: Dict[str, pd.DataFrame], cfg: ExchangeConfig):
        self.market = market_data
        self.account = Account(cfg.init_balance, cfg.vip_level, cfg.hedge_mode, cfg.default_leverage)
        self.cfg = cfg
        self.symbol_leverage: Dict[str, float] = {sym: cfg.default_leverage for sym in market_data.keys()}
        self.open_orders: List[Order] = []
        self._next_order_id: int = 1
        self.cursor: Dict[str, int] = {sym: 0 for sym in market_data.keys()}
        self._now_ts: Optional[int] = None

        for sym, lev in self.symbol_leverage.items():
            self.account.set_symbol_leverage(sym, lev)

    # -------- one-way reverse identical to your C++ idea --------
    def _handle_oneway_reverse(self, symbol: str, quantity: float, price: float, is_long: bool) -> bool:
        if self.cfg.hedge_mode:
            return False
        pos = [p for p in self.account.get_all_positions() if p.symbol == symbol]
        if not pos:
            return False

        long_qty = sum(p.quantity for p in pos if p.is_long)
        short_qty = sum(p.quantity for p in pos if not p.is_long)
        if (is_long and long_qty > 1e-12 and short_qty <= 1e-12) or ((not is_long) and short_qty > 1e-12 and long_qty <= 1e-12):
            return False

        to_reduce_side = (not is_long)
        need = quantity
        for p in self.account.get_all_positions():
            if p.symbol != symbol:
                continue
            if p.is_long == to_reduce_side:
                close_qty = min(need, p.quantity)
                if close_qty > 1e-12:
                    oid = self._next_order_id; self._next_order_id += 1
                    self.open_orders.append(Order(
                        id=oid, symbol=symbol, quantity=close_qty, price=float(price),
                        is_long=(not p.is_long), reduce_only=False, closing_position_id=p.id,
                        ts_accepted=self._now_ts or -1
                    ))
                    need -= close_qty
                if need <= 1e-12:
                    break

        if need > 1e-12:
            oid = self._next_order_id; self._next_order_id += 1
            self.open_orders.append(Order(
                id=oid, symbol=symbol, quantity=need, price=float(price),
                is_long=is_long, reduce_only=False, closing_position_id=-1,
                ts_accepted=self._now_ts or -1
            ))
        return True

    # -------- order APIs --------
    def place_order(self, symbol: str, quantity: float, price: float, is_long: bool, reduce_only: bool = False):
        if quantity <= 0:
            return
        if not reduce_only and not self.cfg.hedge_mode:
            if self._handle_oneway_reverse(symbol, quantity, price, is_long):
                return
        oid = self._next_order_id; self._next_order_id += 1
        self.open_orders.append(Order(
            id=oid, symbol=symbol, quantity=float(quantity), price=float(price),
            is_long=bool(is_long), reduce_only=bool(reduce_only),
            closing_position_id=-1, ts_accepted=self._now_ts or -1
        ))

    def place_market(self, symbol: str, quantity: float, is_long: bool, reduce_only: bool = False):
        self.place_order(symbol, quantity, 0.0, is_long, reduce_only)

    def close_position(self, symbol: str, price: float = 0.0, is_long: Optional[bool] = None):
        pos = [p for p in self.account.get_all_positions() if p.symbol == symbol]
        if not pos:
            return
        for p in pos:
            if is_long is not None and (p.is_long != is_long):
                continue
            oid = self._next_order_id; self._next_order_id += 1
            self.open_orders.append(Order(
                id=oid, symbol=symbol, quantity=p.quantity, price=float(price),
                is_long=(not p.is_long), reduce_only=False, closing_position_id=p.id,
                ts_accepted=self._now_ts or -1
            ))

    def set_symbol_leverage(self, symbol: str, lev: float):
        if lev <= 0:
            return False

        current = self.symbol_leverage.get(symbol, self.cfg.default_leverage)
        if abs(current - lev) <= 1e-12:
            return True

        if not self.account.set_symbol_leverage(symbol, lev):
            return False

        self.symbol_leverage[symbol] = float(lev)
        return True

    def get_open_orders(self) -> List[Order]:
        """Get all currently open orders.
        
        Returns:
            List of open Order objects.
        """
        return self.open_orders.copy()

    def cancel_order_by_id(self, order_id: int) -> bool:
        """Cancel an open order by its ID.
        
        Args:
            order_id: Unique identifier of the order to cancel.
            
        Returns:
            True if order was found and cancelled, False otherwise.
        """
        initial_len = len(self.open_orders)
        self.open_orders = [o for o in self.open_orders if o.id != order_id]
        return len(self.open_orders) < initial_len

    # -------- step --------
    def step(self) -> bool:
        ts = align_next_timestamp(self.cursor, self.market)
        if ts is None:
            return False
        self._now_ts = ts
        snap, self.cursor = slice_bar_snapshot(ts, self.cursor, self.market)

        mtm_price: Dict[str, float] = {}
        for sym, bar in snap.items():
            if bar is None:
                continue
            close = float(bar["close"])
            volume_cap = float(bar["volume"])
            mtm_price[sym] = close
            if volume_cap <= 0:
                continue

            candidates = [o for o in self.open_orders if o.symbol == sym and o.quantity > 1e-12]
            if not candidates:
                continue

            def crossable(o: Order) -> bool:
                if o.price <= 0:
                    return True
                if o.is_long and close <= o.price:
                    return True
                if (not o.is_long) and close >= o.price:
                    return True
                return False

            eligible = [o for o in candidates if crossable(o)]
            if not eligible:
                continue

            remain = volume_cap
            maker_fee, taker_fee = get_fee_rates(self.account.vip_level)
            leftovers: List[Order] = []

            for o in eligible:
                if remain <= 1e-12:
                    leftovers.append(o); continue

                is_market = (o.price <= 0.0)
                fee_rate = taker_fee if is_market else maker_fee
                exec_px = (
                    price_for_market(bar, self.cfg.exec_model.market)
                    if is_market else
                    price_for_crossed_limit(bar, self.cfg.exec_model.limit, o.price)
                )

                fill_qty = min(o.quantity, remain)
                if fill_qty <= 1e-12:
                    leftovers.append(o); continue

                if o.closing_position_id >= 0:
                    fee = fill_qty * exec_px * fee_rate
                    ok = self.account.reduce_or_close(o.closing_position_id, fill_qty, exec_px, fee)
                    if not ok:
                        leftovers.append(o); continue

                else:
                    if o.reduce_only:
                        # 1:1 C++ behavior — same-side reduction only
                        # If no position exists, order is discarded (not kept in leftovers)
                        fee = fill_qty * exec_px * fee_rate
                        ok = self.account.reduce_only(o.symbol, o.is_long, fill_qty, exec_px, fee)
                        if not ok:
                            # reduceOnly failed (no position) => discard order (C++ behavior)
                            continue
                    else:
                        ok = self.account.open_or_increase(
                            order_id=o.id, symbol=o.symbol, fill_qty=fill_qty, fill_price=exec_px,
                            is_long=o.is_long, fee_rate=fee_rate
                        )
                        if not ok:
                            leftovers.append(o); continue

                remain -= fill_qty
                o.quantity -= fill_qty
                if o.quantity > 1e-12:
                    leftovers.append(o)

            # keep non-eligible + leftovers
            still_open: List[Order] = []
            eligible_ids = {o.id for o in eligible}
            leftover_map = {o.id: o for o in leftovers}
            for o in self.open_orders:
                if o.symbol != sym:
                    still_open.append(o); continue
                if o.id not in eligible_ids:
                    still_open.append(o); continue
                if o.id in leftover_map:
                    still_open.append(leftover_map[o.id])
            self.open_orders = still_open

        # MTM + liquidations
        self.account.mark_to_market(mtm_price)
        self.account.check_liquidation()
        return True
    