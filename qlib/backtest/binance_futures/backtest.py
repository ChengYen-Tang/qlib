# -*- coding: utf-8 -*-
"""
A minimal driver that connects:
- data loader -> exchange -> step loop
- exposes a simple Strategy callback to submit orders
- produces pandas-friendly logs if needed (extend later for Qlib Recorder)
"""

from __future__ import annotations
from typing import Callable, Dict, Tuple, List, Optional
import pandas as pd

from .data import load_market, align_next_timestamp, slice_bar_snapshot
from .exchange import Exchange, ExchangeConfig

StrategyFn = Callable[[int, Dict[str, dict], Exchange], None]
# signature: (bar_ts_ms, snapshot_by_symbol, exchange) -> None
# user strategy inspects snapshot & uses exchange.place_order/close_position

def run_backtest(symbol_csv: List[Tuple[str, str]],
                 strategy: StrategyFn,
                 cfg: Optional[ExchangeConfig] = None) -> Dict[str, pd.DataFrame]:
    """
    Run a backtest with the provided strategy.
    Returns dict of logs (placeholder here; extend for Qlib Recorder/parquet).
    """
    cfg = cfg or ExchangeConfig()
    market = load_market(symbol_csv)
    ex = Exchange(market, cfg)

    cursor = {sym: 0 for sym in market.keys()}
    logs_equity: List[Tuple[int, float, float, float]] = []  # ts, balance, unreal, equity

    while True:
        ts = align_next_timestamp(cursor, market)
        if ts is None:
            break
        snap, cursor = slice_bar_snapshot(ts, cursor, market)

        # Strategy can place/close orders before we step (decision at bar open)
        strategy(ts, snap, ex)

        # Exchange advances one bar (match & MTM at close of bar)
        advanced = ex.step()
        if not advanced:
            break

        # Simple account log (balance/unreal/equity) per bar
        acc = ex.account.snapshot()
        logs_equity.append((ts, acc.balance, acc.unreal_pnl, acc.equity))

    log_df = pd.DataFrame(logs_equity, columns=["ts", "balance", "unreal_pnl", "equity"])
    log_df.index = pd.to_datetime(log_df["ts"], unit="ms", utc=True)
    return {"account": log_df[["balance", "unreal_pnl", "equity"]]}
