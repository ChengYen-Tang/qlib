import pandas as pd

from qlib.backtest.binance_futures.backtest import run_backtest
from qlib.backtest.binance_futures.exchange import Exchange, ExchangeConfig


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