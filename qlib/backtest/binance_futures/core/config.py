from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class FeeRate:
    """Maker/Taker fee rates by VIP level."""
    maker_fee_rate: float
    taker_fee_rate: float


VIP_FEE_RATES: Dict[int, FeeRate] = {
    0: FeeRate(0.00020, 0.00050),
    1: FeeRate(0.00016, 0.00040),
    2: FeeRate(0.00014, 0.00035),
    3: FeeRate(0.00012, 0.00032),
    4: FeeRate(0.00010, 0.00030),
    5: FeeRate(0.00008, 0.00027),
    6: FeeRate(0.00006, 0.00025),
    7: FeeRate(0.00004, 0.00022),
    8: FeeRate(0.00002, 0.00020),
    9: FeeRate(0.00000, 0.00017),
}


@dataclass(frozen=True)
class MarginTier:
    """Maintenance tier by notional upper bound."""
    notional_upper: float
    maintenance_margin_rate: float
    max_leverage: float


# Copied semantics from your C++ Config (USDT-margined)
MARGIN_TIERS: List[MarginTier] = [
    MarginTier(   50_000,   0.0040, 125),
    MarginTier(  600_000,   0.0050, 100),
    MarginTier(3_000_000,   0.0065,  75),
    MarginTier(12_000_000,  0.0100,  50),
    MarginTier(70_000_000,  0.0200,  25),
    MarginTier(100_000_000, 0.0250,  20),
    MarginTier(230_000_000, 0.0500,  10),
    MarginTier(480_000_000, 0.1000,   5),
    MarginTier(600_000_000, 0.1250,   4),
    MarginTier(800_000_000, 0.1500,   3),
    MarginTier(1_200_000_000, 0.2500, 2),
    MarginTier(float("inf"), 0.5000,  1),
]
