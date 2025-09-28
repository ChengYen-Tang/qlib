from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ..core.dto import MultiKlineDto, KlineDto, AccountLog, Position, Order
from ..data.market_data import MarketData
from ..engine.account import Account


@dataclass
class _SymbolCursor:
    idx: int = 0


class BinanceExchange:
    """
    Binance-like simulator that:
    - Loads per-symbol CSVs as MarketData.
    - On step(), emits the earliest timestamp across all symbols as a MultiKlineDto.
    - Calls Account.update_positions() with per-symbol (close, volume).
    - Does per-bar MTM and debounced "snapshots" are returned by getters.
    """

    def __init__(self, symbol_csv: List[Tuple[str, str]], init_balance: float = 1_000_000.0, vip_level: int = 0):
        self._md: Dict[str, MarketData] = {sym: MarketData(sym, csv) for sym, csv in symbol_csv}
        self._cursor: Dict[str, _SymbolCursor] = {sym: _SymbolCursor(0) for sym in self._md.keys()}
        self.account = Account(initial_balance=init_balance, vip_level=vip_level)
        self._closed = False

    # --------- Order passthrough ---------

    def place_order(self, symbol: str, quantity: float, price: float, is_long: bool, reduce_only: bool = False):
        self.account.place_order(symbol, quantity, price, is_long, reduce_only)

    def place_order_mkt(self, symbol: str, quantity: float, is_long: bool, reduce_only: bool = False):
        self.account.place_order_mkt(symbol, quantity, is_long, reduce_only)

    def close_position(self, symbol: str, price: float = 0.0):
        self.account.close_position(symbol, price)

    def close_position_one_side(self, symbol: str, is_long: bool, price: float = 0.0):
        self.account.close_position_one_side(symbol, is_long, price)

    # --------- Stepping ---------

    def _next_timestamp(self) -> Optional[int]:
        ts = None
        for sym, md in self._md.items():
            idx = self._cursor[sym].idx
            if idx >= md.get_klines_count():
                continue
            kts = md.get_kline(idx).Timestamp
            ts = kts if ts is None else min(ts, kts)
        return ts

    def _build_multikline(self, ts: int) -> MultiKlineDto:
        out = MultiKlineDto(Timestamp=ts, klines={})
        for sym, md in self._md.items():
            idx = self._cursor[sym].idx
            if idx < md.get_klines_count() and md.get_kline(idx).Timestamp == ts:
                k = md.get_kline(idx)
                out.klines[sym] = k
                self._cursor[sym].idx += 1
            else:
                out.klines[sym] = None
        return out

    def step(self) -> Optional[MultiKlineDto]:
        if self._closed:
            return None
        ts = self._next_timestamp()
        if ts is None:
            self._closed = True
            return None

        dto = self._build_multikline(ts)

        # Collect (price, volume) map for symbols present on this bar
        price_vol = {}
        for sym, k in dto.klines.items():
            if isinstance(k, KlineDto):
                price_vol[sym] = (k.ClosePrice, k.Volume)

        if price_vol:
            self.account.update_positions(ts, price_vol)

        return dto

    # --------- Snapshots ---------

    def get_all_positions(self) -> List[Position]:
        return self.account.get_all_positions()

    def get_all_open_orders(self) -> List[Order]:
        return self.account.get_all_open_orders()

    def get_account_log(self) -> AccountLog:
        eq = self.account.get_equity()
        u = self.account.total_unrealized_pnl()
        bal = self.account.get_balance()
        used = self.account.used_margin
        mr = (used / eq) if eq > 0 else 0.0
        return AccountLog(balance=bal, unreal_pnl=u, equity=eq, used_margin=used, margin_ratio=mr, liquidation=False)
