from __future__ import annotations
from typing import Dict
from datetime import datetime
from ..sim.binance_exchange import BinanceExchange
from ..reports.recorder import Recorder


class BacktestRunner:
    """
    Thin adapter to run the futures simulator and write Qlib-style logs.
    Usage:
        ex = BacktestRunner(symbol_csv=[("BTCUSDT","/path/btc.csv")], out_dir="./run1")
        ex.set_leverage("BTCUSDT", 10)
        ex.run()
    """

    def __init__(self, symbol_csv, out_dir="./out", init_balance=1_000_000.0, vip_level=0):
        self.exchange = BinanceExchange(symbol_csv, init_balance=init_balance, vip_level=vip_level)
        self.rec = Recorder(out_dir)

    def set_leverage(self, symbol: str, lev: float):
        self.exchange.account.set_symbol_leverage(symbol, lev)

    def place_order(self, symbol: str, qty: float, price: float, is_long: bool, reduce_only: bool = False):
        self.exchange.place_order(symbol, qty, price, is_long, reduce_only)

    def place_order_mkt(self, symbol: str, qty: float, is_long: bool, reduce_only: bool = False):
        self.exchange.place_order_mkt(symbol, qty, is_long, reduce_only)

    def close_position(self, symbol: str, price: float = 0.0):
        self.exchange.close_position(symbol, price)

    def run(self):
        """
        Drive the backtest until data exhausted.
        Writes trades.csv once at the end; positions/account are appended per step.
        """
        while True:
            dto = self.exchange.step()
            if dto is None:
                break
            ts = datetime.utcfromtimestamp(dto.Timestamp / 1000.0)

            # Append bar-level positions/account
            self.rec.write_positions(ts, self.exchange.get_all_positions())
            self.rec.write_account(self.exchange.get_account_log(), ts)

        # Dump all executed trades at the end
        self.rec.write_trades(self.exchange.account.executed_trades)
