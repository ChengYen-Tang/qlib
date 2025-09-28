from __future__ import annotations
import csv
from typing import Iterable
from ..core.dto import Trade, Position, AccountLog
from datetime import datetime


class Recorder:
    """
    Minimal recorder that writes Qlib-friendly CSVs:
    - trades.csv
    - positions.csv
    - account.csv
    """

    def __init__(self, out_dir: str):
        self.out_dir = out_dir.rstrip("/")

    def write_trades(self, trades: Iterable[Trade]) -> None:
        path = f"{self.out_dir}/trades.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "datetime","order_id","trade_id","symbol","side","price","qty","value",
                "commission","position_effect","reduce_only","tif","leverage","fee_rate","realized_pnl"
            ])
            for t in trades:
                w.writerow([
                    t.datetime.isoformat(sep=" "),
                    t.order_id, t.trade_id, t.symbol, t.side, f"{t.price:.10f}",
                    f"{t.qty:.10f}", f"{t.value:.10f}", f"{t.commission:.10f}",
                    t.position_effect, t.reduce_only, t.tif, f"{t.leverage:.6f}",
                    f"{t.fee_rate:.6f}", f"{t.realized_pnl:.10f}"
                ])

    def write_positions(self, ts: datetime, positions: Iterable[Position]) -> None:
        path = f"{self.out_dir}/positions.csv"
        header_needed = False
        try:
            with open(path, "r"):
                pass
        except FileNotFoundError:
            header_needed = True
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow([
                    "datetime","symbol","side","qty","entry_price","market_price","unreal_pnl",
                    "notional","init_margin","maint_margin","leverage"
                ])
            for p in positions:
                w.writerow([
                    ts.isoformat(sep=" "),
                    p.symbol,
                    "long" if p.is_long else "short",
                    f"{p.quantity:.10f}",
                    f"{p.entry_price:.10f}",
                    "",  # optional current price (can be reconstructed if needed)
                    f"{p.unrealized_pnl:.10f}",
                    f"{p.notional:.10f}",
                    f"{p.initial_margin:.10f}",
                    f"{p.maintenance_margin:.10f}",
                    f"{p.leverage:.6f}",
                ])

    def write_account(self, alog: AccountLog, ts: datetime) -> None:
        path = f"{self.out_dir}/account.csv"
        header_needed = False
        try:
            with open(path, "r"):
                pass
        except FileNotFoundError:
            header_needed = True
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(["datetime","balance","unreal_pnl","equity","used_margin","margin_ratio","liquidation"])
            w.writerow([
                ts.isoformat(sep=" "),
                f"{alog.balance:.10f}",
                f"{alog.unreal_pnl:.10f}",
                f"{alog.equity:.10f}",
                f"{alog.used_margin:.10f}",
                f"{alog.margin_ratio:.10f}",
                int(alog.liquidation),
            ])
