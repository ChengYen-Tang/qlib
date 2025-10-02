from __future__ import annotations
from typing import Iterable, Tuple

import pandas as pd
import pytest
from qlib.backtest.binance_futures.exchange import Exchange, ExchangeConfig


def make_df(rows: Iterable[Tuple[int, float, float]]) -> pd.DataFrame:
    records = []
    for ts, close, volume in rows:
        records.append({
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": 1,
            "taker_base": volume,
            "taker_quote": volume * close,
            "ts": ts,
        })
    df = pd.DataFrame(records)
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df
def test_step_sequence_with_mixed_symbols():
    market = {
        "BTCUSDT": make_df([(0, 1.0, 100.0), (60_000, 2.0, 200.0)]),
        "ETHUSDT": make_df([(30_000, 10.0, 50.0), (60_000, 20.0, 70.0)]),
    }
    ex = Exchange(market, ExchangeConfig())

    timestamps = []
    while ex.step():
        timestamps.append(ex._now_ts)

    assert timestamps == [0, 30_000, 60_000]
    assert not ex.step()


def test_fifo_partial_fills():
    market = {"BTCUSDT": make_df([(0, 100.0, 1.0), (60_000, 100.0, 1.0)])}
    ex = Exchange(market, ExchangeConfig(init_balance=10_000.0, default_leverage=50.0))

    ex.place_order("BTCUSDT", 1.0, 100.0, True)
    ex.place_order("BTCUSDT", 1.0, 100.0, True)

    assert ex.step()
    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    assert positions[0].quantity == pytest.approx(1.0)

    remaining_orders = [o for o in ex.open_orders if o.symbol == "BTCUSDT"]
    assert len(remaining_orders) == 1
    assert remaining_orders[0].quantity == pytest.approx(1.0)

    assert ex.step()
    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    assert positions[0].quantity == pytest.approx(2.0)
    assert not ex.open_orders


def test_partial_fill_single_order():
    """Test single large order filled progressively (matches C++ UpdatePositionsPartialFillSameOrder)"""
    market = {"BTCUSDT": make_df([
        (0, 1000.0, 2.0),       # First step: only 2 BTC available
        (60_000, 1000.0, 10.0)  # Second step: 10 BTC available (enough for remaining 3)
    ])}
    ex = Exchange(market, ExchangeConfig(init_balance=5000.0, default_leverage=10.0))
    
    # Place single order for 5 BTC
    ex.place_order("BTCUSDT", 5.0, 1000.0, True)
    
    # First step: partial fill of 2 BTC
    assert ex.step()
    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    assert positions[0].quantity == pytest.approx(2.0)
    assert positions[0].symbol == "BTCUSDT"
    
    # Leftover order should have 3 BTC remaining
    remaining_orders = [o for o in ex.open_orders if o.symbol == "BTCUSDT"]
    assert len(remaining_orders) == 1
    assert remaining_orders[0].quantity == pytest.approx(3.0)
    
    # Second step: fill remaining 3 BTC
    assert ex.step()
    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    assert positions[0].quantity == pytest.approx(5.0)  # Merged: 2 + 3 = 5
    
    # No orders should remain
    assert not ex.open_orders


def test_reduce_only_flow():
    market = {"BTCUSDT": make_df([(0, 100.0, 5.0), (60_000, 100.0, 5.0)])}
    ex = Exchange(market, ExchangeConfig(init_balance=10_000.0, default_leverage=20.0))

    ex.place_order("BTCUSDT", 2.0, 100.0, True)
    ex.step()
    pos = ex.account.get_all_positions()[0]
    assert pos.quantity == pytest.approx(2.0)

    ex.place_order("BTCUSDT", 1.0, 100.0, True, reduce_only=True)
    ex.step()
    pos = ex.account.get_all_positions()[0]
    assert pos.quantity == pytest.approx(1.0)
    assert not ex.open_orders


def test_set_symbol_leverage_rebalances():
    market = {"BTCUSDT": make_df([(0, 4000.0, 10.0), (60_000, 4000.0, 10.0), (120_000, 4000.0, 10.0)])}
    ex = Exchange(market, ExchangeConfig(init_balance=10_000.0, default_leverage=20.0))

    assert ex.set_symbol_leverage("BTCUSDT", 20.0)
    ex.place_order("BTCUSDT", 1.0, 4000.0, True)
    ex.step()
    balance_after_open = ex.account.balance

    assert ex.set_symbol_leverage("BTCUSDT", 10.0)
    assert ex.account.balance == pytest.approx(balance_after_open - 200.0)
    assert ex.account.used_margin == pytest.approx(400.0)

    assert ex.set_symbol_leverage("BTCUSDT", 40.0)
    assert ex.account.used_margin == pytest.approx(100.0)

    ex.place_order("BTCUSDT", 5.0, 4000.0, True)
    ex.step()
    before_fail = ex.account.balance
    assert not ex.set_symbol_leverage("BTCUSDT", 1.0)
    assert ex.account.balance == pytest.approx(before_fail)


def test_set_leverage_rejects_invalid_values():
    """Test leverage validation and default values"""
    market = {"BTCUSDT": make_df([(0, 1000.0, 10.0)])}
    ex = Exchange(market, ExchangeConfig(init_balance=10_000.0, default_leverage=1.0))
    
    # Test default leverage is applied
    assert ex.symbol_leverage.get("BTCUSDT") == pytest.approx(1.0)
    
    # Test setting valid leverage
    assert ex.set_symbol_leverage("BTCUSDT", 50.0)
    assert ex.symbol_leverage["BTCUSDT"] == pytest.approx(50.0)
    
    # Test rejecting <= 0 leverage
    assert not ex.set_symbol_leverage("BTCUSDT", 0.0)
    assert ex.symbol_leverage["BTCUSDT"] == pytest.approx(50.0)  # unchanged
    
    assert not ex.set_symbol_leverage("BTCUSDT", -10.0)
    assert ex.symbol_leverage["BTCUSDT"] == pytest.approx(50.0)  # unchanged


def test_oneway_reverse_creates_closing_and_new_order():
    market = {"BTCUSDT": make_df([(0, 100.0, 10.0), (60_000, 100.0, 10.0)])}
    ex = Exchange(market, ExchangeConfig(init_balance=10_000.0, default_leverage=20.0))

    ex.place_order("BTCUSDT", 2.0, 100.0, True)
    ex.step()
    pos = ex.account.get_all_positions()[0]
    assert pos.quantity == pytest.approx(2.0)

    ex.place_order("BTCUSDT", 3.0, 100.0, False)

    closing = [o for o in ex.open_orders if o.closing_position_id >= 0]
    opening = [o for o in ex.open_orders if o.closing_position_id < 0]
    assert len(closing) == 1
    assert closing[0].quantity == pytest.approx(2.0)
    assert closing[0].is_long is False
    assert len(opening) == 1
    assert opening[0].quantity == pytest.approx(1.0)
    assert opening[0].is_long is False

    ex.step()
    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    assert positions[0].is_long is False
    assert positions[0].quantity == pytest.approx(1.0)


def test_hedge_mode_allows_opposite_positions():
    market = {"BTCUSDT": make_df([(0, 100.0, 10.0)])}
    cfg = ExchangeConfig(init_balance=10_000.0, default_leverage=20.0, hedge_mode=True)
    ex = Exchange(market, cfg)

    ex.place_order("BTCUSDT", 2.0, 100.0, True)
    ex.place_order("BTCUSDT", 1.0, 100.0, False)
    ex.step()

    positions = ex.account.get_all_positions()
    assert len(positions) == 2
    long_pos = next(p for p in positions if p.is_long)
    short_pos = next(p for p in positions if not p.is_long)
    assert long_pos.quantity == pytest.approx(2.0)
    assert short_pos.quantity == pytest.approx(1.0)


def test_close_only_long_side_in_hedge_mode():
    market = {"BTCUSDT": make_df([(0, 100.0, 10.0), (60_000, 100.0, 10.0)])}
    cfg = ExchangeConfig(init_balance=10_000.0, default_leverage=20.0, hedge_mode=True)
    ex = Exchange(market, cfg)

    ex.place_order("BTCUSDT", 2.0, 100.0, True)
    ex.place_order("BTCUSDT", 1.0, 100.0, False)
    ex.step()

    ex.close_position("BTCUSDT", is_long=True)
    ex.step()

    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    assert positions[0].is_long is False
    assert positions[0].quantity == pytest.approx(1.0)


def test_close_both_sides_in_hedge_mode():
    market = {"BTCUSDT": make_df([(0, 100.0, 10.0), (60_000, 100.0, 10.0)])}
    cfg = ExchangeConfig(init_balance=10_000.0, default_leverage=20.0, hedge_mode=True)
    ex = Exchange(market, cfg)

    ex.place_order("BTCUSDT", 2.0, 100.0, True)
    ex.place_order("BTCUSDT", 1.0, 100.0, False)
    ex.step()

    ex.close_position("BTCUSDT")
    ex.step()

    positions = ex.account.get_all_positions()
    assert not positions


def test_reduce_only_partial_fill_in_hedge_mode():
    market = {"BTCUSDT": make_df([(0, 9_000.0, 10.0), (60_000, 9_000.0, 2.0)])}
    cfg = ExchangeConfig(init_balance=10_000.0, default_leverage=20.0, hedge_mode=True)
    ex = Exchange(market, cfg)

    ex.place_order("BTCUSDT", 5.0, 9_000.0, True)
    ex.step()
    pos = ex.account.get_all_positions()[0]
    assert pos.quantity == pytest.approx(5.0)

    ex.place_order("BTCUSDT", 5.0, 9_000.0, True, reduce_only=True)
    ex.step()
    pos = ex.account.get_all_positions()[0]
    assert pos.quantity == pytest.approx(3.0)

    leftovers = [o for o in ex.open_orders if o.reduce_only]
    assert leftovers
    assert leftovers[0].quantity == pytest.approx(3.0)


def test_single_mode_reduce_only_with_no_position_discards_order():
    """reduceOnly order without existing position should be discarded (C++ behavior)"""
    market = {"ETHUSDT": make_df([(0, 1_500.0, 5.0)])}
    ex = Exchange(market, ExchangeConfig(init_balance=10_000.0, default_leverage=10.0, hedge_mode=False))

    ex.place_order("ETHUSDT", 2.0, 1_500.0, True, reduce_only=True)
    ex.step()

    # No position should be created
    assert not ex.account.get_all_positions()
    # Order should be discarded (not kept in open_orders)
    leftovers = [o for o in ex.open_orders if o.symbol == "ETHUSDT"]
    assert not leftovers


def test_multi_symbol_partial_fills_in_hedge_mode():
    market = {
        "BTCUSDT": make_df([(0, 20_000.0, 1.0)]),
        "ETHUSDT": make_df([(0, 1_500.0, 3.0)]),
    }
    cfg = ExchangeConfig(init_balance=50_000.0, default_leverage=20.0, hedge_mode=True)
    ex = Exchange(market, cfg)

    ex.place_order("BTCUSDT", 2.0, 20_000.0, True)
    ex.place_order("ETHUSDT", 5.0, 1_500.0, False)
    ex.step()

    positions = ex.account.get_all_positions()
    assert len(positions) == 2
    btc_qty = next(p.quantity for p in positions if p.symbol == "BTCUSDT")
    eth_qty = next(p.quantity for p in positions if p.symbol == "ETHUSDT")
    assert btc_qty == pytest.approx(1.0)
    assert eth_qty == pytest.approx(3.0)

    open_orders = ex.open_orders
    assert len(open_orders) == 2
    remaining = {o.symbol: o.quantity for o in open_orders}
    assert remaining["BTCUSDT"] == pytest.approx(1.0)
    assert remaining["ETHUSDT"] == pytest.approx(2.0)


def test_merge_then_partial_close_in_hedge_mode():
    market = {
        "BTCUSDT": make_df([(0, 20_000.0, 10.0), (60_000, 21_000.0, 2.0)])
    }
    cfg = ExchangeConfig(init_balance=100_000.0, default_leverage=20.0, hedge_mode=True)
    ex = Exchange(market, cfg)

    ex.place_order("BTCUSDT", 1.0, 20_000.0, True)
    ex.place_order("BTCUSDT", 2.0, 20_000.0, True)
    ex.place_order("BTCUSDT", 3.0, 20_000.0, True)
    ex.step()

    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    assert positions[0].quantity == pytest.approx(6.0)

    ex.close_position("BTCUSDT", is_long=True, price=21_000.0)
    ex.step()

    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    assert positions[0].quantity == pytest.approx(4.0)

    closing_orders = [o for o in ex.open_orders if o.closing_position_id >= 0]
    assert closing_orders
    assert closing_orders[0].quantity == pytest.approx(4.0)


def test_single_mode_multiple_symbols_with_partial_reversal():
    market = {
        "BTCUSDT": make_df([(0, 20_000.0, 5.0), (60_000, 20_000.0, 5.0)]),
        "ETHUSDT": make_df([(0, 1_500.0, 10.0), (60_000, 1_500.0, 10.0)]),
    }
    cfg = ExchangeConfig(init_balance=100_000.0, default_leverage=20.0, hedge_mode=False)
    ex = Exchange(market, cfg)

    ex.place_order("BTCUSDT", 1.0, 20_000.0, True)
    ex.place_order("ETHUSDT", 2.0, 1_500.0, False)
    ex.step()

    positions = {p.symbol: p for p in ex.account.get_all_positions()}
    assert positions["BTCUSDT"].is_long is True
    assert positions["BTCUSDT"].quantity == pytest.approx(1.0)
    assert positions["ETHUSDT"].is_long is False
    assert positions["ETHUSDT"].quantity == pytest.approx(2.0)

    ex.place_order("BTCUSDT", 0.5, 20_000.0, False)
    ex.step()

    positions = {p.symbol: p for p in ex.account.get_all_positions()}
    assert positions["BTCUSDT"].is_long is True
    assert positions["BTCUSDT"].quantity == pytest.approx(0.5)
    assert positions["ETHUSDT"].quantity == pytest.approx(2.0)
    assert not [o for o in ex.open_orders if o.symbol == "BTCUSDT"]


def test_hedge_mode_multiple_symbols_reduce_only_orders():
    market = {
        "BTCUSDT": make_df([(0, 20_000.0, 5.0), (60_000, 20_500.0, 5.0)]),
        "ETHUSDT": make_df([(0, 1_500.0, 10.0), (60_000, 1_550.0, 10.0)]),
    }
    cfg = ExchangeConfig(init_balance=100_000.0, default_leverage=20.0, hedge_mode=True)
    ex = Exchange(market, cfg)

    ex.place_order("BTCUSDT", 2.0, 20_000.0, True)
    ex.place_order("ETHUSDT", 3.0, 1_500.0, True)
    ex.step()

    positions = {p.symbol: p for p in ex.account.get_all_positions()}
    assert positions["BTCUSDT"].quantity == pytest.approx(2.0)
    assert positions["ETHUSDT"].quantity == pytest.approx(3.0)

    ex.place_order("BTCUSDT", 1.0, 20_500.0, True, reduce_only=True)
    ex.step()

    positions = {p.symbol: p for p in ex.account.get_all_positions()}
    assert positions["BTCUSDT"].quantity == pytest.approx(1.0)
    assert positions["ETHUSDT"].quantity == pytest.approx(3.0)
    assert not ex.open_orders


# ============================================================
# Missing Test Cases (6 total)
# ============================================================

def test_set_position_mode_with_no_positions():
    """Test switching position mode when no positions exist (C++ SetAndGetHedgeMode)."""
    market = {"BTCUSDT": make_df([(0, 50_000.0, 10.0)])}
    ex = Exchange(market, ExchangeConfig(init_balance=100_000.0))
    
    # Initially one-way mode
    assert not ex.account.is_hedge_mode()
    
    # Switch to hedge mode (should succeed - no positions)
    assert ex.account.set_position_mode(True)
    assert ex.account.is_hedge_mode()
    
    # Switch back to one-way mode (should succeed - no positions)
    assert ex.account.set_position_mode(False)
    assert not ex.account.is_hedge_mode()


def test_set_position_mode_fails_with_open_positions():
    """Test that mode switching fails when positions exist (C++ SetAndGetHedgeMode)."""
    market = {"BTCUSDT": make_df([(0, 50_000.0, 10.0), (60_000, 50_000.0, 10.0)])}
    ex = Exchange(market, ExchangeConfig(init_balance=100_000.0, default_leverage=20.0))
    
    # Place and fill an order to create a position
    ex.place_order("BTCUSDT", 1.0, 50_000.0, True)
    ex.step()
    
    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    
    # Try to switch to hedge mode (should fail - position exists)
    assert not ex.account.set_position_mode(True)
    assert not ex.account.is_hedge_mode()


def test_is_hedge_mode_reflects_current_mode():
    """Test that is_hedge_mode() correctly reflects the account state (C++ SetAndGetHedgeMode)."""
    market = {"BTCUSDT": make_df([(0, 50_000.0, 10.0)])}
    
    # Test one-way mode initialization
    ex1 = Exchange(market, ExchangeConfig(hedge_mode=False))
    assert not ex1.account.is_hedge_mode()
    
    # Test hedge mode initialization
    ex2 = Exchange(market, ExchangeConfig(hedge_mode=True))
    assert ex2.account.is_hedge_mode()


def test_cancel_order_by_id():
    """Test canceling an open order by its ID (C++ CancelOrderByID)."""
    market = {"BTCUSDT": make_df([(0, 50_000.0, 10.0), (60_000, 60_000.0, 10.0)])}
    ex = Exchange(market, ExchangeConfig(init_balance=100_000.0, default_leverage=10.0))
    
    # Place multiple orders that won't immediately fill (prices above market)
    ex.place_order("BTCUSDT", 1.0, 51_000.0, True)  # Buy at 51k (market at 50k)
    ex.place_order("BTCUSDT", 2.0, 52_000.0, True)  # Buy at 52k
    ex.place_order("BTCUSDT", 3.0, 53_000.0, True)  # Buy at 53k
    
    # Get order IDs
    orders = ex.get_open_orders()
    assert len(orders) == 3
    order_id_to_cancel = orders[1].id
    
    # Cancel the middle order
    assert ex.cancel_order_by_id(order_id_to_cancel)
    
    # Verify only 2 orders remain
    remaining_orders = ex.get_open_orders()
    assert len(remaining_orders) == 2
    assert all(o.id != order_id_to_cancel for o in remaining_orders)
    
    # Try to cancel non-existent order (should return False)
    assert not ex.cancel_order_by_id(99999)


def test_place_order_adds_to_open_orders():
    """Test that placing orders adds them to the open orders list (C++ PlaceOrderSuccessCheckOpenOrders)."""
    market = {"BTCUSDT": make_df([(0, 50_000.0, 10.0), (60_000, 50_000.0, 10.0)])}
    ex = Exchange(market, ExchangeConfig(init_balance=100_000.0, default_leverage=10.0))
    
    # Initially no open orders
    assert len(ex.get_open_orders()) == 0
    
    # Place buy limit orders BELOW market (won't fill immediately)
    # Market is at 50k, buy orders at 45k won't execute until price drops to 45k
    ex.place_order("BTCUSDT", 1.0, 45_000.0, True)
    
    # Check order was added
    orders = ex.get_open_orders()
    assert len(orders) == 1
    assert orders[0].symbol == "BTCUSDT"
    assert orders[0].quantity == 1.0
    assert orders[0].price == 45_000.0
    assert orders[0].is_long is True
    
    # Place another order
    ex.place_order("BTCUSDT", 2.0, 44_000.0, True)
    orders = ex.get_open_orders()
    assert len(orders) == 2
    
    # Step forward (price stays at 50k, orders remain open)
    ex.step()
    orders = ex.get_open_orders()
    assert len(orders) == 2


def test_close_position_hedge_mode():
    """Test closing specific position sides in hedge mode (C++ close_position with is_long parameter)."""
    market = {
        "BTCUSDT": make_df([
            (0, 50_000.0, 10.0),
            (60_000, 51_000.0, 10.0),
            (120_000, 52_000.0, 10.0),
        ])
    }
    cfg = ExchangeConfig(init_balance=200_000.0, default_leverage=10.0, hedge_mode=True)
    ex = Exchange(market, cfg)
    
    # Open both long and short positions in hedge mode
    ex.place_order("BTCUSDT", 2.0, 50_000.0, True)   # Long
    ex.place_order("BTCUSDT", 1.5, 50_000.0, False)  # Short
    ex.step()
    
    positions = ex.account.get_all_positions()
    assert len(positions) == 2
    long_pos = [p for p in positions if p.is_long][0]
    short_pos = [p for p in positions if not p.is_long][0]
    assert long_pos.quantity == pytest.approx(2.0)
    assert short_pos.quantity == pytest.approx(1.5)
    
    # Close only the long position using close_position with is_long parameter
    ex.close_position("BTCUSDT", price=51_000.0, is_long=True)
    ex.step()
    
    # Verify long position is closed, short position remains
    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    assert not positions[0].is_long
    assert positions[0].quantity == pytest.approx(1.5)
    
    # Close the short position
    ex.close_position("BTCUSDT", price=52_000.0, is_long=False)
    ex.step()
    
    # All positions should be closed
    positions = ex.account.get_all_positions()
    assert len(positions) == 0
