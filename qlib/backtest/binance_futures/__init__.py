"""
Qlib-compatible Binance Futures backtest (standalone).
This package provides:
- DTOs for market, orders, positions.
- CSV-based market data loader.
- Futures Account engine with margin, tiers, fees, liquidation.
- A bar-by-bar Binance-like exchange simulator with per-bar MTM.
- A recorder that emits trades/positions/account series for Qlib-style reports.
- An adapter to convert target positions to orders if needed.
"""
__all__ = [
    "core", "data", "engine", "sim", "reports", "api"
]
