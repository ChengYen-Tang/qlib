import pandas as pd
import pytest

from qlib.backtest.binance_futures.backtest import run_signal_backtest
from qlib.backtest.binance_futures.data import align_next_timestamp, load_market, slice_bar_snapshot
from qlib.backtest.binance_futures.exchange import Exchange, ExchangeConfig
from qlib.backtest.binance_futures.strategy import SignalRegimeConfig, SignalRegimeExecutor


def _make_df(rows):
    records = []
    for ts, close, volume in rows:
        records.append(
            {
                "open_ts": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": volume,
                "close_ts": ts + 60_000,
                "quote_volume": volume * close,
                "trades": 1,
                "taker_base": volume,
                "taker_quote": volume * close,
            }
        )
    return pd.DataFrame(records)


def _write_symbol_csv(tmp_path, name, rows):
    df = _make_df(rows)
    path = tmp_path / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)


def _run_executor(symbol_csv, signals, cfg=None, regime=None):
    cfg = cfg or ExchangeConfig(init_balance=100_000.0, default_leverage=20.0)
    market = load_market(symbol_csv)
    exchange = Exchange(market, cfg)
    executor = SignalRegimeExecutor(signals, config=regime)

    cursor = {sym: 0 for sym in market.keys()}
    while True:
        ts = align_next_timestamp(cursor, market)
        if ts is None:
            break
        snap, cursor = slice_bar_snapshot(ts, cursor, market)
        executor(ts, snap, exchange)
        if not exchange.step():
            break

    return exchange


def test_signal_executor_transitions_across_regimes(tmp_path):
    rows = [(0, 100.0, 1_000_000.0), (60_000, 100.0, 1_000_000.0), (120_000, 100.0, 1_000_000.0)]
    symbol_csv = [
        ("BTCUSDT", _write_symbol_csv(tmp_path, "btc", rows)),
        ("ETHUSDT", _write_symbol_csv(tmp_path, "eth", rows)),
    ]

    signals = pd.DataFrame(
        [
            {"BTCUSDT": 0.8, "ETHUSDT": 0.6},
            {"BTCUSDT": 0.1, "ETHUSDT": -0.2},
            {"BTCUSDT": -0.6, "ETHUSDT": -0.4},
        ],
        index=pd.to_datetime([0, 60_000, 120_000], unit="ms", utc=True),
    )

    regime = SignalRegimeConfig(
        pos_threshold=0.05,
        neg_threshold=0.05,
        bullish_ratio=0.5,
        bearish_ratio=0.5,
        bullish_net=0.6,
        neutral_net=0.0,
        bearish_net=-0.6,
        gross_exposure=1.0,
        rebalance_tolerance=1e-6,
    )

    exchange = _run_executor(symbol_csv, signals, regime=regime)

    snapshot = exchange.account.snapshot()
    positions = exchange.account.get_all_positions()
    assert len(positions) == 2

    equity = snapshot.equity
    assert equity > 0

    weights = {}
    for pos in positions:
        signed_qty = pos.quantity if pos.is_long else -pos.quantity
        weights[pos.symbol] = (signed_qty * pos.entry_price) / equity
        assert pos.is_long is False

    assert weights["BTCUSDT"] == pytest.approx(-0.6, abs=0.1)
    assert weights["ETHUSDT"] == pytest.approx(-0.4, abs=0.1)


def test_signal_executor_flattens_without_strong_signals(tmp_path):
    rows = [(0, 100.0, 1_000_000.0), (60_000, 100.0, 1_000_000.0)]
    symbol_csv = [
        ("BTCUSDT", _write_symbol_csv(tmp_path, "btc", rows)),
    ]

    signals = pd.DataFrame(
        [
            {"BTCUSDT": 0.01},
            {"BTCUSDT": -0.01},
        ],
        index=pd.to_datetime([0, 60_000], unit="ms", utc=True),
    )

    regime = SignalRegimeConfig(pos_threshold=0.05, neg_threshold=0.05)

    exchange = _run_executor(symbol_csv, signals, regime=regime)
    assert not exchange.account.get_all_positions()


def test_run_signal_backtest_returns_detail(tmp_path):
    rows = [(0, 100.0, 1_000_000.0), (60_000, 100.0, 1_000_000.0)]
    csv_path = _write_symbol_csv(tmp_path, "btc", rows)
    signals = pd.DataFrame(
        [
            {"BTCUSDT": 0.5},
            {"BTCUSDT": 0.6},
        ],
        index=pd.to_datetime([0, 60_000], unit="ms", utc=True),
    )

    result = run_signal_backtest(
        [("BTCUSDT", csv_path)],
        signals,
        cfg=ExchangeConfig(init_balance=10_000.0, default_leverage=10.0),
        return_detail=True,
    )

    assert set(result.keys()) >= {"account", "fills", "portfolio"}
    assert not result["fills"].empty