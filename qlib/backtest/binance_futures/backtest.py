# -*- coding: utf-8 -*-
"""
A minimal driver that connects:
- data loader -> exchange -> step loop
- exposes a simple Strategy callback to submit orders
- produces pandas-friendly logs if needed (extend later for Qlib Recorder)
"""

from __future__ import annotations
from typing import Callable, Dict, Tuple, List, Optional, Union
import pandas as pd

from .data import load_market, align_next_timestamp, slice_bar_snapshot
from .exchange import Exchange, ExchangeConfig
from .strategy import SignalRegimeConfig, SignalRegimeExecutor
from qlib.backtest.report import PortfolioMetrics

StrategyFn = Callable[[int, Dict[str, dict], Exchange], None]
# signature: (bar_ts_ms, snapshot_by_symbol, exchange) -> None
# user strategy inspects snapshot & uses exchange.place_order/close_position

def _normalize_benchmark_config(benchmark: Optional[Union[str, List[str], pd.Series, Dict]]) -> Optional[dict]:
    if benchmark is None:
        return None
    if isinstance(benchmark, dict):
        return benchmark
    return {"benchmark": benchmark}


def _build_portfolio_metrics(
    step_logs: pd.DataFrame,
    init_balance: float,
    freq: str,
    benchmark: Optional[Union[str, List[str], pd.Series, Dict]] = None,
) -> pd.DataFrame:
    if step_logs.empty:
        return PortfolioMetrics(freq=freq, benchmark_config=_normalize_benchmark_config(benchmark)).generate_portfolio_metrics_dataframe()

    pm = PortfolioMetrics(freq=freq, benchmark_config=_normalize_benchmark_config(benchmark))
    last_account_value = float(init_balance)
    last_total_turnover = 0.0
    last_total_cost = 0.0

    for ts, row in step_logs.iterrows():
        account_value = float(row["equity"])
        cash = float(row["balance"])
        total_turnover = float(row["total_turnover"])
        total_cost = float(row["total_cost"])
        step_turnover = total_turnover - last_total_turnover
        step_cost = total_cost - last_total_cost
        denom = last_account_value if abs(last_account_value) > 1e-12 else 1.0
        return_rate = ((account_value - last_account_value) + step_cost) / denom
        turnover_rate = step_turnover / denom
        cost_rate = step_cost / denom
        stock_value = account_value - cash

        pm.update_portfolio_metrics_record(
            trade_start_time=ts,
            account_value=account_value,
            cash=cash,
            return_rate=return_rate,
            total_turnover=total_turnover,
            turnover_rate=turnover_rate,
            total_cost=total_cost,
            cost_rate=cost_rate,
            stock_value=stock_value,
            bench_value=0.0,
        )

        last_account_value = account_value
        last_total_turnover = total_turnover
        last_total_cost = total_cost

    return pm.generate_portfolio_metrics_dataframe()


def run_backtest(
    symbol_csv: List[Tuple[str, str]],
    strategy: StrategyFn,
    cfg: Optional[ExchangeConfig] = None,
    return_detail: bool = False,
    report_freq: str = "1min",
    benchmark: Optional[Union[str, List[str], pd.Series, Dict]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run a backtest with the provided strategy.
    Returns dict of logs (placeholder here; extend for Qlib Recorder/parquet).
    """
    cfg = cfg or ExchangeConfig()
    market = load_market(symbol_csv)
    ex = Exchange(market, cfg)

    cursor = {sym: 0 for sym in market.keys()}

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

    step_logs = ex.get_step_logs()
    if step_logs.empty:
        account_df = pd.DataFrame(columns=["balance", "unreal_pnl", "equity"])
    else:
        account_df = step_logs[["balance", "unreal_pnl", "equity"]].copy()
    account_df.index.name = "datetime"

    result: Dict[str, pd.DataFrame] = {"account": account_df}

    if return_detail:
        result["step"] = step_logs.copy()
        result["fills"] = ex.get_fill_reports()
        result["portfolio"] = _build_portfolio_metrics(step_logs, cfg.init_balance, report_freq, benchmark)

    return result


def run_signal_backtest(
    symbol_csv: List[Tuple[str, str]],
    signals: pd.DataFrame,
    cfg: Optional[ExchangeConfig] = None,
    regime_config: Optional[SignalRegimeConfig] = None,
    return_detail: bool = False,
    report_freq: str = "1min",
    benchmark: Optional[Union[str, List[str], pd.Series, Dict]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run backtest driven by a signal dataframe using the regime executor."""

    executor = SignalRegimeExecutor(signals, config=regime_config)
    return run_backtest(
        symbol_csv,
        strategy=executor,
        cfg=cfg,
        return_detail=return_detail,
        report_freq=report_freq,
        benchmark=benchmark,
    )
