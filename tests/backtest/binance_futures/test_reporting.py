import pandas as pd
import pytest

from qlib.backtest.binance_futures.backtest import run_backtest
from qlib.backtest.binance_futures.exchange import Exchange, ExchangeConfig
from qlib.backtest.binance_futures.data import load_market


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


def test_exchange_step_logs_and_fills():
    market = {
        "BTCUSDT": _make_df([(0, 100.0, 5.0), (60_000, 105.0, 5.0)])
    }
    cfg = ExchangeConfig(init_balance=10_000.0, default_leverage=20.0)
    ex = Exchange(market, cfg)

    ex.place_order("BTCUSDT", 2.0, 100.0, True)
    assert ex.step()
    assert ex.step()

    step_logs = ex.get_step_logs()
    assert not step_logs.empty
    assert "step_turnover" in step_logs.columns
    fills = ex.get_fill_reports()
    assert not fills.empty
    assert set(fills.columns) == {"ts", "order_id", "symbol", "qty", "price", "fee", "is_close", "is_long"}


def test_run_backtest_with_portfolio_metrics(tmp_path):
    csv_path = tmp_path / "btc.csv"
    df = _make_df([(0, 100.0, 5.0), (60_000, 105.0, 5.0), (120_000, 102.0, 5.0)])
    df.to_csv(csv_path, index=False)

    def strategy(ts, snap, ex):
        if ts == 0:
            ex.place_order("BTCUSDT", 1.0, snap["BTCUSDT"]["close"], True)
        elif ts == 60_000:
            ex.place_order("BTCUSDT", 0.5, snap["BTCUSDT"]["close"], False, reduce_only=True)

    result = run_backtest(
        [("BTCUSDT", str(csv_path))],
        strategy,
        cfg=ExchangeConfig(init_balance=5_000.0, default_leverage=10.0),
        return_detail=True,
        report_freq="1min",
    )

    assert "account" in result
    assert "portfolio" in result
    account_df = result["account"]
    assert not account_df.empty
    portfolio_df = result["portfolio"]
    assert {"account", "return", "total_turnover", "total_cost"}.issubset(portfolio_df.columns)
    fills_df = result["fills"]
    assert not fills_df.empty
    assert (fills_df["symbol"] == "BTCUSDT").all()


def test_exchange_handles_zero_volume_bar():
    market = {
        "BTCUSDT": _make_df([
            (0, 100.0, 5.0),
            (60_000, 110.0, 0.0),
            (120_000, 105.0, 5.0),
        ])
    }
    ex = Exchange(market, ExchangeConfig(init_balance=5_000.0, default_leverage=10.0))

    ex.place_order("BTCUSDT", 1.0, 100.0, True)
    assert ex.step()

    ex.place_order("BTCUSDT", 0.5, 0.0, True, reduce_only=True)

    assert ex.step()
    remaining_orders = [o for o in ex.open_orders if o.symbol == "BTCUSDT"]
    assert len(remaining_orders) == 1
    assert remaining_orders[0].reduce_only is True
    assert remaining_orders[0].quantity == pytest.approx(0.5)

    step_logs = ex.get_step_logs()
    assert pytest.approx(0.0) == step_logs.iloc[-1]["step_turnover"]
    fills_after_zero = ex.get_fill_reports()
    assert len(fills_after_zero) == 1

    assert ex.step()
    fills_final = ex.get_fill_reports()
    assert len(fills_final) == 2
    assert not any(o.symbol == "BTCUSDT" for o in ex.open_orders)


def test_load_market_normalizes_multi_symbol(tmp_path):
    btc_path = tmp_path / "btc.csv"
    eth_path = tmp_path / "eth.csv"

    btc_df = _make_df([
        (120_000, 101.0, 2.0),
        (0, 100.0, 1.0),
        (60_000, 102.0, 1.5),
    ]).iloc[[1, 2, 0]].reset_index(drop=True)
    eth_df = _make_df([
        (90_000, 2_050.0, 3.0),
        (30_000, 2_000.0, 4.0),
        (150_000, 2_100.0, 2.5),
    ]).iloc[[2, 0, 1]].reset_index(drop=True)

    btc_df.to_csv(btc_path, index=False)
    eth_df.to_csv(eth_path, index=False)

    market = load_market([
        ("BTCUSDT", str(btc_path)),
        ("ETHUSDT", str(eth_path)),
    ])

    assert set(market.keys()) == {"BTCUSDT", "ETHUSDT"}

    for sym, df in market.items():
        assert list(df.columns) == [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "trades",
            "taker_base",
            "taker_quote",
            "ts",
        ]
        assert df.index.tz is not None
        assert df.index.is_monotonic_increasing
        assert pd.api.types.is_integer_dtype(df["ts"].dtype)
        assert df["ts"].is_monotonic_increasing
        assert len(df) == 3
