from __future__ import annotations
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from .types import Order, Position, FillReport
from .account import Account
from .config import get_fee_rates
from .data import align_next_timestamp, slice_bar_snapshot, normalize_symbol_df
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
        normalized_market = {sym: normalize_symbol_df(df) for sym, df in market_data.items()}
        self.market = normalized_market
        self.account = Account(cfg.init_balance, cfg.vip_level, cfg.hedge_mode, cfg.default_leverage)
        self.cfg = cfg
        self.symbol_leverage: Dict[str, float] = {
            sym: cfg.default_leverage for sym in normalized_market.keys()
        }
        self.open_orders: List[Order] = []
        self._next_order_id: int = 1
        self.cursor: Dict[str, int] = {sym: 0 for sym in normalized_market.keys()}
        self._now_ts: Optional[int] = None
        self._step_logs: List[Dict[str, float]] = []
        self._fills_by_step: List[Tuple[int, List[FillReport]]] = []
        self._cum_turnover: float = 0.0
        self._cum_cost: float = 0.0

        for sym, lev in self.symbol_leverage.items():
            self.account.set_symbol_leverage(sym, lev)

    # -------- reporting helpers --------
    def get_step_logs(self) -> pd.DataFrame:
        """Return per-step account ledger including turnover and costs."""
        if not self._step_logs:
            return pd.DataFrame(columns=[
                "ts",
                "balance",
                "unreal_pnl",
                "equity",
                "used_margin",
                "step_turnover",
                "step_cost",
                "total_turnover",
                "total_cost",
            ])
        df = pd.DataFrame(self._step_logs)
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

    def get_fill_reports(self) -> pd.DataFrame:
        """Return all trade fills captured during the backtest."""
        records: List[Dict[str, float | int | bool]] = []
        for ts, fills in self._fills_by_step:
            for fill in fills:
                records.append({
                    "ts": ts,
                    "order_id": fill.order_id,
                    "symbol": fill.symbol,
                    "qty": fill.qty,
                    "price": fill.price,
                    "fee": fill.fee,
                    "is_close": fill.is_close,
                    "is_long": fill.is_long,
                })
        if not records:
            return pd.DataFrame(columns=["ts", "order_id", "symbol", "qty", "price", "fee", "is_close", "is_long"])
        df = pd.DataFrame(records)
        df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
        return df

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
        """Get all currently open orders."""
        return self.open_orders.copy()

    def cancel_order_by_id(self, order_id: int) -> bool:
        """Cancel an open order by its ID."""
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
        fills_this_step: List[FillReport] = []
        step_turnover = 0.0
        step_cost = 0.0
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
                    leftovers.append(o)
                    continue

                is_market = o.price <= 0.0
                fee_rate = taker_fee if is_market else maker_fee
                exec_px = (
                    price_for_market(bar, self.cfg.exec_model.market)
                    if is_market
                    else price_for_crossed_limit(bar, self.cfg.exec_model.limit, o.price)
                )

                fill_qty = min(o.quantity, remain)
                if fill_qty <= 1e-12:
                    leftovers.append(o)
                    continue

                notional = fill_qty * exec_px
                executed = False
                fee = 0.0

                if o.closing_position_id >= 0:
                    fee = notional * fee_rate
                    if self.account.reduce_or_close(o.closing_position_id, fill_qty, exec_px, fee):
                        executed = True
                    else:
                        leftovers.append(o)
                        continue
                else:
                    if o.reduce_only:
                        fee = notional * fee_rate
                        if not self.account.reduce_only(o.symbol, o.is_long, fill_qty, exec_px, fee):
                            # reduceOnly failed (no position) => discard order (C++ behavior)
                            continue
                        executed = True
                    else:
                        fee = notional * fee_rate
                        if not self.account.open_or_increase(
                            order_id=o.id,
                            symbol=o.symbol,
                            fill_qty=fill_qty,
                            fill_price=exec_px,
                            is_long=o.is_long,
                            fee_rate=fee_rate,
                        ):
                            leftovers.append(o)
                            continue
                        executed = True

                if executed:
                    fills_this_step.append(
                        FillReport(
                            order_id=o.id,
                            symbol=o.symbol,
                            qty=fill_qty,
                            price=exec_px,
                            fee=fee,
                            is_close=(o.closing_position_id >= 0) or o.reduce_only,
                            is_long=o.is_long,
                            ts=ts,
                        )
                    )
                    remain -= fill_qty
                    o.quantity -= fill_qty
                    step_turnover += notional
                    step_cost += fee
                    if o.quantity > 1e-12:
                        leftovers.append(o)

            still_open: List[Order] = []
            eligible_ids = {o.id for o in eligible}
            leftover_map = {o.id: o for o in leftovers}
            for o in self.open_orders:
                if o.symbol != sym:
                    still_open.append(o)
                    continue
                if o.id not in eligible_ids:
                    still_open.append(o)
                    continue
                if o.id in leftover_map:
                    still_open.append(leftover_map[o.id])
            self.open_orders = still_open

        self.account.mark_to_market(mtm_price)
        self.account.check_liquidation()

        snapshot = self.account.snapshot()
        self._cum_turnover += step_turnover
        self._cum_cost += step_cost
        self._step_logs.append(
            {
                "ts": ts,
                "balance": snapshot.balance,
                "unreal_pnl": snapshot.unreal_pnl,
                "equity": snapshot.equity,
                "used_margin": snapshot.used_margin,
                "step_turnover": step_turnover,
                "step_cost": step_cost,
                "total_turnover": self._cum_turnover,
                "total_cost": self._cum_cost,
            }
        )
        self._fills_by_step.append((ts, fills_this_step))
        return True
    