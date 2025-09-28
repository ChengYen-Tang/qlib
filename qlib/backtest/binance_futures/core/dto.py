from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime


# ---------- Account / Orders / Positions ----------

@dataclass
class AccountLog:
    """Snapshot of account metrics at a point in time."""
    balance: float
    unreal_pnl: float
    equity: float
    used_margin: float = 0.0
    margin_ratio: float = 0.0
    liquidation: bool = False
    datetime: Optional[datetime] = None


@dataclass
class Order:
    """Order description for the futures engine."""
    id: int
    symbol: str
    quantity: float         # remaining quantity
    price: float            # <=0 → market; >0 → limit
    is_long: bool           # True=buy(long), False=sell(short)
    reduce_only: bool = False
    closing_position_id: int = -1
    tif: str = "GTC"        # only GTC honored in this minimal sim
    created_at: Optional[datetime] = None


@dataclass
class Position:
    """Open position snapshot."""
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


# ---------- Market DTOs ----------

@dataclass
class BaseMarketDto:
    """Base market DTO with millisecond epoch timestamp."""
    Timestamp: int  # ms since epoch


@dataclass
class KlineDto(BaseMarketDto):
    """Single bar (kline) data."""
    OpenPrice: float
    HighPrice: float
    LowPrice: float
    ClosePrice: float
    Volume: float  # base-asset volume of the bar
    CloseTime: int
    QuoteVolume: float
    TradeCount: int
    TakerBuyBaseVolume: float
    TakerBuyQuoteVolume: float


@dataclass
class MultiKlineDto(BaseMarketDto):
    """Multiple symbols’ 1-min klines aligned on the same Timestamp."""
    klines: Dict[str, Optional[KlineDto]] = field(default_factory=dict)


# ---------- Trade record (for recorder) ----------

@dataclass
class Trade:
    """Executed trade record for reporting."""
    datetime: datetime
    order_id: int
    trade_id: int
    symbol: str
    side: str                 # "buy" / "sell"
    price: float
    qty: float
    value: float
    commission: float
    position_effect: str      # "open" / "close"
    reduce_only: bool
    tif: str
    leverage: float
    fee_rate: float
    realized_pnl: float = 0.0
