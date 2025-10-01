# -*- coding: utf-8 -*-
"""
Centralized config: VIP fee table and maintenance margin tiers.
All rates follow your C++ tables (maker/taker) and margin tiers.
"""

from __future__ import annotations
from typing import Tuple

# VIP maker/taker fee rates (match your C++)
_VIP = {
    0: (0.00020, 0.00050),
    1: (0.00016, 0.00040),
    2: (0.00014, 0.00035),
    3: (0.00012, 0.00032),
    4: (0.00010, 0.00030),
    5: (0.00008, 0.00027),
    6: (0.00006, 0.00025),
    7: (0.00004, 0.00022),
    8: (0.00002, 0.00020),
    9: (0.00000, 0.00017),
}

# Margin tiers (upper_notional, mmr, max_lev) — match your C++
_TIERS = [
    (   50_000,   0.0040, 125),
    (  600_000,   0.0050, 100),
    (3_000_000,   0.0065,  75),
    (12_000_000,  0.0100,  50),
    (70_000_000,  0.0200,  25),
    (100_000_000, 0.0250,  20),
    (230_000_000, 0.0500,  10),
    (480_000_000, 0.1000,   5),
    (600_000_000, 0.1250,   4),
    (800_000_000, 0.1500,   3),
    (1_200_000_000, 0.2500, 2),
    (float("inf"), 0.5000,  1),
]

def get_fee_rates(vip_level: int) -> Tuple[float, float]:
    maker, taker = _VIP.get(vip_level, _VIP[0])
    return maker, taker

def get_tier_info(notional: float) -> Tuple[float, float]:
    for upper, mmr, max_lev in _TIERS:
        if notional <= upper:
            return mmr, max_lev
    # fallback, never hits
    return _TIERS[-1][1], _TIERS[-1][2]
