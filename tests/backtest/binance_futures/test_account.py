import pytest
from qlib.backtest.binance_futures.account import Account


FEE_RATE = 0.0005


def set_leverage(acc: Account, symbol: str, leverage: float) -> None:
    assert acc.set_symbol_leverage(symbol, leverage)


def test_snapshot_initial_state():
    acc = Account(1000.0, 0)
    snap = acc.snapshot()
    assert snap.balance == pytest.approx(1000.0)
    assert snap.unreal_pnl == pytest.approx(0.0)
    assert snap.equity == pytest.approx(1000.0)
    assert acc.used_margin == pytest.approx(0.0)


def test_open_and_merge_positions():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)

    ok = acc.open_or_increase(1, "BTCUSDT", 2.0, 1000.0, True, FEE_RATE)
    assert ok
    pos = acc.get_all_positions()[0]
    assert pos.quantity == pytest.approx(2.0)
    assert pos.entry_price == pytest.approx(1000.0)
    assert pos.initial_margin == pytest.approx(200.0)
    assert acc.balance == pytest.approx(10_000.0 - (200.0 + 2_000.0 * FEE_RATE))
    assert acc.used_margin == pytest.approx(200.0)

    ok = acc.open_or_increase(2, "BTCUSDT", 1.0, 1200.0, True, FEE_RATE)
    assert ok
    pos = acc.get_all_positions()[0]
    assert pos.quantity == pytest.approx(3.0)
    expected_entry = (2.0 * 1000.0 + 1.0 * 1200.0) / 3.0
    assert pos.entry_price == pytest.approx(expected_entry)
    assert pos.initial_margin == pytest.approx((2000.0 + 1200.0) / 10.0)
    assert pos.notional == pytest.approx(3200.0)


def test_reduce_or_close_updates_balance_and_removes_zero():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    acc.open_or_increase(1, "BTCUSDT", 3.0, 1000.0, True, FEE_RATE)
    pos = acc.get_all_positions()[0]

    prior_balance = acc.balance
    prior_used = acc.used_margin

    fee = 1.5 * 1100.0 * FEE_RATE
    acc.reduce_or_close(pos.id, 1.5, 1100.0, fee)

    pos = acc.get_all_positions()[0]
    assert pos.quantity == pytest.approx(1.5)
    freed_init = prior_used * 0.5
    realized = (1100.0 - 1000.0) * 1.5
    expected_balance = prior_balance + freed_init + realized - fee
    assert acc.balance == pytest.approx(expected_balance)
    assert acc.used_margin == pytest.approx(prior_used - freed_init)

    fee_all = 1.5 * 900.0 * FEE_RATE
    acc.reduce_or_close(pos.id, 1.5, 900.0, fee_all)
    assert not acc.get_all_positions()
    assert acc.used_margin == pytest.approx(0.0)


def test_reduce_only_same_side():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    acc.open_or_increase(1, "BTCUSDT", 2.0, 1000.0, True, FEE_RATE)
    pos = acc.get_all_positions()[0]
    fee = 0.5 * 900.0 * FEE_RATE
    ok = acc.reduce_only("BTCUSDT", True, 0.5, 900.0, fee)
    assert ok
    pos = acc.get_all_positions()[0]
    assert pos.quantity == pytest.approx(1.5)

    # no matching side
    ok = acc.reduce_only("ETHUSDT", True, 1.0, 1000.0, fee)
    assert not ok


def test_adjust_symbol_leverage_balance_effects():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 20.0)
    acc.open_or_increase(1, "BTCUSDT", 1.0, 4000.0, True, FEE_RATE)
    pos = acc.get_all_positions()[0]
    assert pos.initial_margin == pytest.approx(200.0)

    balance_after_open = acc.balance
    used_after_open = acc.used_margin

    assert acc.adjust_symbol_leverage("BTCUSDT", 20.0, 10.0)
    assert acc.used_margin == pytest.approx(400.0)
    assert acc.balance == pytest.approx(balance_after_open - 200.0)

    assert acc.adjust_symbol_leverage("BTCUSDT", 10.0, 40.0)
    assert acc.used_margin == pytest.approx(100.0)
    assert acc.balance == pytest.approx(balance_after_open + 100.0)

    # Expand position significantly, then attempt leverage reduction that should fail
    acc.open_or_increase(2, "BTCUSDT", 5.0, 4000.0, True, FEE_RATE)
    before_fail_balance = acc.balance
    before_fail_used = acc.used_margin
    assert not acc.adjust_symbol_leverage("BTCUSDT", 40.0, 1.0)
    assert acc.balance == pytest.approx(before_fail_balance)
    assert acc.used_margin == pytest.approx(before_fail_used)


def test_mark_to_market_and_liquidation():
    acc = Account(2_000.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    acc.open_or_increase(1, "BTCUSDT", 4.0, 500.0, True, FEE_RATE)

    acc.mark_to_market({"BTCUSDT": 50.0})
    acc.check_liquidation()

    assert acc.balance == pytest.approx(0.0)
    assert acc.used_margin == pytest.approx(0.0)
    assert not acc.get_all_positions()


def test_adjust_symbol_leverage_rejects_invalid_inputs():
    acc = Account(1_000.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    acc.open_or_increase(1, "BTCUSDT", 0.5, 1_000.0, True, FEE_RATE)
    before_balance = acc.balance
    before_used = acc.used_margin

    assert not acc.adjust_symbol_leverage("BTCUSDT", 10.0, 0.0)
    assert acc.balance == pytest.approx(before_balance)
    assert acc.used_margin == pytest.approx(before_used)


def test_open_short_position_creates_entry():
    acc = Account(5_000.0, 0)
    set_leverage(acc, "BTCUSDT", 20.0)
    ok = acc.open_or_increase(1, "BTCUSDT", 2.0, 900.0, False, FEE_RATE)
    assert ok

    positions = acc.get_all_positions()
    assert len(positions) == 1
    pos = positions[0]
    assert pos.is_long is False
    assert pos.quantity == pytest.approx(2.0)
    assert pos.entry_price == pytest.approx(900.0)
    expected_margin = (2.0 * 900.0) / 20.0
    assert pos.initial_margin == pytest.approx(expected_margin)
    expected_fee = 2.0 * 900.0 * FEE_RATE
    assert pos.fee == pytest.approx(expected_fee)
    assert acc.used_margin == pytest.approx(expected_margin)


def test_open_or_increase_rejects_over_max_leverage():
    acc = Account(5_000.0, 0)
    before_balance = acc.balance
    set_leverage(acc, "BTCUSDT", 200.0)
    ok = acc.open_or_increase(1, "BTCUSDT", 1.0, 1_000.0, True, FEE_RATE)
    assert not ok
    assert acc.balance == pytest.approx(before_balance)
    assert not acc.get_all_positions()


def test_open_or_increase_rejects_when_equity_insufficient():
    acc = Account(50.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    ok = acc.open_or_increase(1, "BTCUSDT", 1.0, 1_000.0, True, FEE_RATE)
    assert not ok
    assert acc.balance == pytest.approx(50.0)
    assert acc.used_margin == pytest.approx(0.0)


def test_open_or_increase_merges_short_positions():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    ok = acc.open_or_increase(1, "BTCUSDT", 1.0, 1_200.0, False, FEE_RATE)
    assert ok
    ok = acc.open_or_increase(2, "BTCUSDT", 0.5, 800.0, False, FEE_RATE)
    assert ok

    pos = acc.get_all_positions()[0]
    assert pos.is_long is False
    assert pos.quantity == pytest.approx(1.5)
    expected_entry = ((1.0 * 1_200.0) + (0.5 * 800.0)) / 1.5
    assert pos.entry_price == pytest.approx(expected_entry)
    assert pos.initial_margin == pytest.approx((1_200.0 + 400.0) / 10.0)


def test_open_both_directions_creates_two_positions():
    acc = Account(10_000.0, 0, hedge_mode=True)
    set_leverage(acc, "BTCUSDT", 20.0)
    assert acc.open_or_increase(1, "BTCUSDT", 1.0, 1_000.0, True, FEE_RATE)
    assert acc.open_or_increase(2, "BTCUSDT", 2.0, 1_100.0, False, FEE_RATE)

    positions = acc.get_all_positions()
    assert len(positions) == 2
    long_pos = next(p for p in positions if p.is_long)
    short_pos = next(p for p in positions if not p.is_long)
    assert long_pos.quantity == pytest.approx(1.0)
    assert short_pos.quantity == pytest.approx(2.0)


def test_adjust_symbol_leverage_no_positions_returns_true():
    acc = Account(5_000.0, 0)
    before_balance = acc.balance
    before_used = acc.used_margin
    assert acc.set_symbol_leverage("BTCUSDT", 5.0)
    assert acc.adjust_symbol_leverage("BTCUSDT", 5.0, 5.0)
    assert acc.balance == pytest.approx(before_balance)
    assert acc.used_margin == pytest.approx(before_used)


def test_adjust_symbol_leverage_fails_when_equity_insufficient():
    acc = Account(50.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    assert acc.open_or_increase(1, "BTCUSDT", 1.0, 100.0, True, FEE_RATE)
    before_balance = acc.balance
    before_used = acc.used_margin
    assert not acc.set_symbol_leverage("BTCUSDT", 1.0)
    assert acc.balance == pytest.approx(before_balance)
    assert acc.used_margin == pytest.approx(before_used)


def test_reduce_or_close_clamps_quantity_and_removes_position():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    acc.open_or_increase(1, "BTCUSDT", 1.0, 1_000.0, True, FEE_RATE)
    pos = acc.get_all_positions()[0]

    fee = 1.0 * 1_100.0 * FEE_RATE
    acc.reduce_or_close(pos.id, 5.0, 1_100.0, fee)
    assert not acc.get_all_positions()
    assert acc.used_margin == pytest.approx(0.0)


def test_reduce_or_close_unknown_position_returns_false():
    acc = Account(10_000.0, 0)
    assert not acc.reduce_or_close(999, 1.0, 1_000.0, 0.0)


def test_reduce_only_opposite_side_returns_false():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    acc.open_or_increase(1, "BTCUSDT", 1.0, 1_000.0, True, FEE_RATE)
    fee = 1_000.0 * FEE_RATE
    assert not acc.reduce_only("BTCUSDT", False, 1.0, 1_000.0, fee)


def test_reduce_only_exact_close_removes_position():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    acc.open_or_increase(1, "BTCUSDT", 1.0, 1_000.0, True, FEE_RATE)
    fee = 1_100.0 * FEE_RATE
    assert acc.reduce_only("BTCUSDT", True, 1.0, 1_100.0, fee)
    assert not acc.get_all_positions()
    assert acc.used_margin == pytest.approx(0.0)


def test_mark_to_market_updates_only_known_symbols():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 20.0)
    set_leverage(acc, "ETHUSDT", 20.0)
    acc.open_or_increase(1, "BTCUSDT", 1.0, 1_000.0, True, FEE_RATE)
    acc.open_or_increase(2, "ETHUSDT", 2.0, 500.0, False, FEE_RATE)

    acc.mark_to_market({"BTCUSDT": 1_200.0})
    positions = {p.symbol: p for p in acc.get_all_positions()}
    assert positions["BTCUSDT"].unrealized_pnl == pytest.approx(200.0)
    assert positions["ETHUSDT"].unrealized_pnl == pytest.approx(0.0)


def test_mark_to_market_without_positions_is_noop():
    acc = Account(10_000.0, 0)
    acc.mark_to_market({"BTCUSDT": 1_200.0})
    assert acc.balance == pytest.approx(10_000.0)
    assert acc.used_margin == pytest.approx(0.0)


def test_check_liquidation_does_not_trigger_when_equity_sufficient():
    acc = Account(5_000.0, 0)
    set_leverage(acc, "BTCUSDT", 50.0)
    acc.open_or_increase(1, "BTCUSDT", 1.0, 1_000.0, True, FEE_RATE)
    acc.mark_to_market({"BTCUSDT": 950.0})
    acc.check_liquidation()

    positions = acc.get_all_positions()
    assert positions
    assert acc.balance > 0.0


def test_total_unrealized_pnl_updates_after_mark():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 20.0)
    acc.open_or_increase(1, "BTCUSDT", 1.0, 1_000.0, True, FEE_RATE)
    assert acc.total_unrealized_pnl() == pytest.approx(0.0)

    acc.mark_to_market({"BTCUSDT": 1_100.0})
    assert acc.total_unrealized_pnl() == pytest.approx(100.0)
    acc.mark_to_market({"BTCUSDT": 900.0})
    assert acc.total_unrealized_pnl() == pytest.approx(-100.0)


def test_snapshot_reflects_unrealized_pnl():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    acc.open_or_increase(1, "BTCUSDT", 1.0, 1_000.0, True, FEE_RATE)
    acc.mark_to_market({"BTCUSDT": 1_100.0})

    snap = acc.snapshot()
    assert snap.unreal_pnl == pytest.approx(100.0)
    assert snap.equity == pytest.approx(acc.balance + 100.0)


def test_equity_matches_balance_plus_unrealized():
    acc = Account(10_000.0, 0)
    set_leverage(acc, "BTCUSDT", 10.0)
    acc.open_or_increase(1, "BTCUSDT", 1.0, 1_000.0, True, FEE_RATE)
    acc.mark_to_market({"BTCUSDT": 800.0})
    expected_unreal = -200.0
    assert acc.total_unrealized_pnl() == pytest.approx(expected_unreal)
    assert acc.equity() == pytest.approx(acc.balance + expected_unreal)
