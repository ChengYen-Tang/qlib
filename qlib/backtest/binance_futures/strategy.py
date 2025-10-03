# -*- coding: utf-8 -*-
"""Signal-driven strategy helpers for the Binance futures backtester."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

from .exchange import Exchange


@dataclass
class SignalRegimeConfig:
    """Hyper-parameters controlling how signals map to exposure."""

    pos_threshold: float = 0.05
    neg_threshold: float = 0.05
    bullish_ratio: float = 0.55
    bearish_ratio: float = 0.55
    bullish_net: float = 0.6
    neutral_net: float = 0.0
    bearish_net: float = -0.6
    gross_exposure: float = 1.0
    rebalance_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if self.pos_threshold < 0 or self.neg_threshold < 0:
            raise ValueError("Signal thresholds must be non-negative")
        for value in (self.bullish_ratio, self.bearish_ratio):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Regime ratios must lie in [0, 1]")
        for value in (self.bullish_net, self.neutral_net, self.bearish_net):
            if not -1.0 <= value <= 1.0:
                raise ValueError("Net exposure targets must lie in [-1, 1]")
        if self.gross_exposure <= 0:
            raise ValueError("Gross exposure must be positive")
        if self.rebalance_tolerance < 0:
            raise ValueError("Rebalance tolerance must be non-negative")


class SignalRegimeExecutor:
    """Callable object that converts model scores into exchange orders."""

    def __init__(self, signals: pd.DataFrame, config: SignalRegimeConfig | None = None):
        if signals.empty:
            raise ValueError("signals dataframe is empty")
        self.config = config or SignalRegimeConfig()
        self._signals = self._prepare_signals(signals)

    @staticmethod
    def _prepare_signals(signals: pd.DataFrame) -> pd.DataFrame:
        df = signals.copy()
        if isinstance(df.index, pd.DatetimeIndex):
            if df.index.tz is None:
                idx = df.index.tz_localize("UTC")
            else:
                idx = df.index.tz_convert("UTC")
            ms_index = (idx.view("int64") // 1_000_000).astype(np.int64)
        elif np.issubdtype(df.index.dtype, np.integer):
            ms_index = df.index.astype(np.int64)
        else:
            raise TypeError("signals index must be DatetimeIndex or integer timestamps")

        df = df.astype(float)
        df.index = pd.Index(ms_index, name="ts_ms")
        df = df.sort_index()
        return df

    def __call__(self, ts: int, snapshot: Dict[str, dict], exchange: Exchange) -> None:
        if ts not in self._signals.index:
            return

        row = self._signals.loc[ts]
        row = row.dropna()
        if row.empty:
            self._rebalance({}, snapshot, exchange)
            return

        long_candidates = {sym: float(score) for sym, score in row.items() if score >= self.config.pos_threshold}
        short_candidates = {sym: float(score) for sym, score in row.items() if score <= -self.config.neg_threshold}
        total_candidates = len(long_candidates) + len(short_candidates)

        net_target = self.config.neutral_net
        if total_candidates > 0:
            positive_ratio = len(long_candidates) / total_candidates if total_candidates else 0.0
            negative_ratio = len(short_candidates) / total_candidates if total_candidates else 0.0
            if positive_ratio >= self.config.bullish_ratio:
                net_target = self.config.bullish_net
            elif negative_ratio >= self.config.bearish_ratio:
                net_target = self.config.bearish_net

        weights = self._construct_weights(long_candidates, short_candidates, net_target)
        self._rebalance(weights, snapshot, exchange)

    def _construct_weights(
        self,
        long_candidates: Dict[str, float],
        short_candidates: Dict[str, float],
        net_target: float,
    ) -> Dict[str, float]:
        config = self.config
        gross_budget = config.gross_exposure
        net_budget = gross_budget * np.clip(net_target, -1.0, 1.0)

        long_budget = max(net_budget, 0.0)
        short_budget = max(-net_budget, 0.0)
        remaining = gross_budget - (long_budget + short_budget)
        if remaining > 1e-12:
            if long_candidates and short_candidates:
                long_budget += remaining * 0.5
                short_budget += remaining * 0.5
            elif long_candidates:
                long_budget += remaining
            elif short_candidates:
                short_budget += remaining

        weights: Dict[str, float] = {}
        if long_candidates and long_budget > 0:
            long_sum = sum(long_candidates.values())
            if long_sum <= 0:
                share = long_budget / len(long_candidates)
                for sym in long_candidates:
                    weights[sym] = share
            else:
                for sym, score in long_candidates.items():
                    weights[sym] = long_budget * score / long_sum

        if short_candidates and short_budget > 0:
            short_sum = sum(abs(v) for v in short_candidates.values())
            if short_sum <= 0:
                share = short_budget / len(short_candidates)
                for sym in short_candidates:
                    weights[sym] = -share
            else:
                for sym, score in short_candidates.items():
                    weights[sym] = -short_budget * abs(score) / short_sum

        return weights

    def _rebalance(self, weights: Dict[str, float], snapshot: Dict[str, dict], exchange: Exchange) -> None:
        account_snapshot = exchange.account.snapshot()
        equity = account_snapshot.equity
        if equity <= 0:
            return

        current_positions = self._current_position_quantities(exchange)
        tolerance = self.config.rebalance_tolerance

        symbols = set(exchange.market.keys()) | set(current_positions.keys()) | set(weights.keys())
        for sym in symbols:
            bar = snapshot.get(sym)
            if not bar:
                continue
            price = float(bar.get("close", 0.0))
            if price <= 0:
                continue

            target_weight = weights.get(sym, 0.0)
            target_qty = (equity * target_weight) / price
            delta = target_qty - current_positions.get(sym, 0.0)
            if abs(delta) <= tolerance:
                continue

            if delta > 0:
                exchange.place_order(sym, delta, 0.0, True)
            else:
                exchange.place_order(sym, abs(delta), 0.0, False)

    @staticmethod
    def _current_position_quantities(exchange: Exchange) -> Dict[str, float]:
        qty: Dict[str, float] = {}
        for pos in exchange.account.get_all_positions():
            signed = pos.quantity if pos.is_long else -pos.quantity
            qty[pos.symbol] = qty.get(pos.symbol, 0.0) + signed
        return qty


def signal_executor(signals: pd.DataFrame, config: SignalRegimeConfig | None = None) -> SignalRegimeExecutor:
    """Factory helper returning a callable executor for run_backtest."""

    return SignalRegimeExecutor(signals, config=config)