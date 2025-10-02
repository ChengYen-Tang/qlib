# -*- coding: utf-8 -*-
"""
Test cases for K-line driven position state updates.

This module tests the "channel-like" behavior where position states
automatically update when K-line data advances, similar to C++ Channel mechanism.
"""

from __future__ import annotations
from typing import Iterable, Tuple

import pandas as pd
import pytest
from qlib.backtest.binance_futures.exchange import Exchange, ExchangeConfig


def make_df(rows: Iterable[Tuple[int, float, float]]) -> pd.DataFrame:
    """Helper to create market data DataFrame."""
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


def test_kline_advance_updates_unrealized_pnl():
    """
    測試 K 線推進時自動更新未實現盈虧 (模擬 C++ Channel 推送價格更新)
    
    場景：
    1. 在 50k 開倉做多 1 BTC
    2. K 線推進到下一根，價格上漲到 55k
    3. 驗證未實現盈虧自動更新為 +5000 USDT
    """
    market = {
        "BTCUSDT": make_df([
            (0, 50_000.0, 10.0),      # 第一根 K 線：50k
            (60_000, 55_000.0, 10.0), # 第二根 K 線：55k (+10%)
        ])
    }
    ex = Exchange(market, ExchangeConfig(init_balance=100_000.0, default_leverage=10.0))
    
    # 開倉：做多 1 BTC @ 50k
    ex.place_order("BTCUSDT", 1.0, 50_000.0, True)
    ex.step()
    
    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    assert positions[0].quantity == pytest.approx(1.0)
    assert positions[0].entry_price == pytest.approx(50_000.0)
    
    # 第一次 step 後，未實現盈虧應該是 0（價格沒變）
    assert positions[0].unrealized_pnl == pytest.approx(0.0, abs=1.0)
    
    # K 線推進：價格從 50k → 55k
    ex.step()
    
    # 驗證：未實現盈虧自動更新
    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    expected_pnl = (55_000.0 - 50_000.0) * 1.0  # (新價格 - 開倉價) × 持倉量 = 5000 USDT
    assert positions[0].unrealized_pnl == pytest.approx(expected_pnl, rel=1e-3)
    
    # 驗證：equity 也相應更新
    snapshot = ex.account.snapshot()
    assert snapshot.unreal_pnl == pytest.approx(expected_pnl, rel=1e-3)


def test_kline_advance_with_multiple_positions():
    """
    測試多個倉位時，K 線推進會更新所有倉位的未實現盈虧
    
    場景：
    1. 同時持有 BTC 多倉和 ETH 多倉
    2. K 線推進，兩個交易對價格都變化
    3. 驗證兩個倉位的未實現盈虧都正確更新
    """
    market = {
        "BTCUSDT": make_df([
            (0, 50_000.0, 10.0),
            (60_000, 52_000.0, 10.0),  # BTC: +4%
        ]),
        "ETHUSDT": make_df([
            (0, 3_000.0, 100.0),
            (60_000, 3_150.0, 100.0),  # ETH: +5%
        ]),
    }
    ex = Exchange(market, ExchangeConfig(init_balance=200_000.0, default_leverage=10.0))
    
    # 開倉：BTC 1.0 @ 50k, ETH 10.0 @ 3k
    ex.place_order("BTCUSDT", 1.0, 50_000.0, True)
    ex.place_order("ETHUSDT", 10.0, 3_000.0, True)
    ex.step()
    
    positions = {p.symbol: p for p in ex.account.get_all_positions()}
    assert len(positions) == 2
    
    # K 線推進
    ex.step()
    
    # 驗證：兩個倉位的未實現盈虧都更新了
    positions = {p.symbol: p for p in ex.account.get_all_positions()}
    
    btc_pnl = (52_000.0 - 50_000.0) * 1.0  # 2000 USDT
    eth_pnl = (3_150.0 - 3_000.0) * 10.0   # 1500 USDT
    
    assert positions["BTCUSDT"].unrealized_pnl == pytest.approx(btc_pnl, rel=1e-3)
    assert positions["ETHUSDT"].unrealized_pnl == pytest.approx(eth_pnl, rel=1e-3)
    
    # 驗證：總未實現盈虧
    total_pnl = ex.account.total_unrealized_pnl()
    assert total_pnl == pytest.approx(btc_pnl + eth_pnl, rel=1e-3)


def test_kline_advance_triggers_liquidation():
    """
    測試 K 線推進時觸發強平 (模擬 C++ Channel 推送價格導致強平)
    
    場景：
    1. 使用高槓桿開倉
    2. K 線推進，價格大幅不利變動
    3. 驗證自動觸發強平，清空所有倉位
    """
    market = {
        "BTCUSDT": make_df([
            (0, 50_000.0, 10.0),
            (60_000, 45_000.0, 10.0),  # 價格下跌 10%
        ])
    }
    # 使用 50 倍槓桿，小額資金，容易觸發強平
    ex = Exchange(market, ExchangeConfig(init_balance=2_000.0, default_leverage=50.0))
    
    # 開倉：做多 1 BTC @ 50k
    # 名義價值 50k，槓桿 50x，保證金僅需 1000 USDT
    ex.place_order("BTCUSDT", 1.0, 50_000.0, True)
    ex.step()
    
    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    
    initial_balance = ex.account.balance
    assert initial_balance > 0
    
    # K 線推進：價格從 50k 跌到 45k (-10%)
    # 虧損：(45k - 50k) × 1.0 = -5000 USDT
    # 權益：2000 - (已用保證金) - 5000 < 維持保證金
    ex.step()
    
    # 驗證：觸發強平，所有倉位被清空
    positions = ex.account.get_all_positions()
    assert len(positions) == 0
    
    # 驗證：餘額歸零（強平後果）
    assert ex.account.balance == pytest.approx(0.0)
    assert ex.account.used_margin == pytest.approx(0.0)


def test_kline_advance_updates_long_and_short_separately():
    """
    測試 K 線推進時，雙向持倉模式下多空倉位分別更新未實現盈虧
    
    場景（雙向持倉）：
    1. 同時持有 BTC 多倉和空倉
    2. K 線推進，價格上漲
    3. 驗證：多倉盈利，空倉虧損
    """
    market = {
        "BTCUSDT": make_df([
            (0, 50_000.0, 10.0),
            (60_000, 51_000.0, 10.0),  # 價格上漲 2%
        ])
    }
    cfg = ExchangeConfig(init_balance=200_000.0, default_leverage=10.0, hedge_mode=True)
    ex = Exchange(market, cfg)
    
    # 開倉：做多 1 BTC, 做空 1 BTC
    ex.place_order("BTCUSDT", 1.0, 50_000.0, True)   # 多倉
    ex.place_order("BTCUSDT", 1.0, 50_000.0, False)  # 空倉
    ex.step()
    
    positions = ex.account.get_all_positions()
    assert len(positions) == 2
    
    # K 線推進：價格上漲
    ex.step()
    
    # 驗證：多倉盈利，空倉虧損
    positions = ex.account.get_all_positions()
    long_pos = [p for p in positions if p.is_long][0]
    short_pos = [p for p in positions if not p.is_long][0]
    
    expected_long_pnl = (51_000.0 - 50_000.0) * 1.0   # +1000 USDT
    expected_short_pnl = (51_000.0 - 50_000.0) * 1.0 * -1.0  # -1000 USDT
    
    assert long_pos.unrealized_pnl == pytest.approx(expected_long_pnl, rel=1e-3)
    assert short_pos.unrealized_pnl == pytest.approx(expected_short_pnl, rel=1e-3)
    
    # 驗證：總未實現盈虧為 0（多空對沖）
    total_pnl = ex.account.total_unrealized_pnl()
    assert total_pnl == pytest.approx(0.0, abs=1.0)


def test_kline_advance_preserves_position_without_price_update():
    """
    測試 K 線推進時，沒有新價格的交易對保持原未實現盈虧
    
    場景：
    1. 持有 BTC 倉位
    2. K 線推進，但該 K 線沒有 BTC 的價格資料（volume=0 或缺失）
    3. 驗證：倉位狀態不變，未實現盈虧保持原值
    """
    market = {
        "BTCUSDT": make_df([
            (0, 50_000.0, 10.0),
            (60_000, 51_000.0, 10.0),
            (120_000, 51_000.0, 0.0),  # 第三根 K 線：沒有成交量
        ])
    }
    ex = Exchange(market, ExchangeConfig(init_balance=100_000.0, default_leverage=10.0))
    
    # 開倉
    ex.place_order("BTCUSDT", 1.0, 50_000.0, True)
    ex.step()
    
    # 第一次推進：價格更新到 51k
    ex.step()
    
    positions = ex.account.get_all_positions()
    pnl_after_first_step = positions[0].unrealized_pnl
    assert pnl_after_first_step == pytest.approx(1000.0, rel=1e-3)
    
    # 第二次推進：volume=0，價格資料無效
    ex.step()
    
    # 驗證：未實現盈虧保持不變（沒有新的有效價格）
    positions = ex.account.get_all_positions()
    assert positions[0].unrealized_pnl == pytest.approx(pnl_after_first_step, rel=1e-3)


def test_kline_advance_with_price_series():
    """
    測試連續多根 K 線推進，未實現盈虧持續更新
    
    場景：模擬連續價格波動
    1. 開倉 @ 100
    2. K 線 1: 105 (+5%)
    3. K 線 2: 110 (+5%)
    4. K 線 3: 108 (-1.8%)
    5. K 線 4: 115 (+6.5%)
    
    驗證每一步的未實現盈虧都正確更新
    """
    market = {
        "BTCUSDT": make_df([
            (0, 100.0, 10.0),
            (60_000, 105.0, 10.0),
            (120_000, 110.0, 10.0),
            (180_000, 108.0, 10.0),
            (240_000, 115.0, 10.0),
        ])
    }
    ex = Exchange(market, ExchangeConfig(init_balance=10_000.0, default_leverage=10.0))
    
    # 開倉 10 BTC @ 100
    ex.place_order("BTCUSDT", 10.0, 100.0, True)
    ex.step()
    
    positions = ex.account.get_all_positions()
    assert positions[0].unrealized_pnl == pytest.approx(0.0, abs=1.0)
    
    # 價格序列和預期盈虧
    price_series = [105.0, 110.0, 108.0, 115.0]
    expected_pnls = [(p - 100.0) * 10.0 for p in price_series]
    
    for expected_pnl in expected_pnls:
        ex.step()
        positions = ex.account.get_all_positions()
        actual_pnl = positions[0].unrealized_pnl
        assert actual_pnl == pytest.approx(expected_pnl, rel=1e-3), \
            f"Expected PnL {expected_pnl}, got {actual_pnl}"


def test_kline_advance_equity_calculation():
    """
    測試 K 線推進時，equity 計算正確（balance + unrealized_pnl）
    
    驗證：
    1. equity = balance + total_unrealized_pnl
    2. K 線推進時 equity 隨未實現盈虧變化
    3. 平倉後 equity = balance（未實現盈虧變為已實現）
    """
    market = {
        "BTCUSDT": make_df([
            (0, 50_000.0, 10.0),
            (60_000, 55_000.0, 10.0),   # 價格上漲
            (120_000, 55_000.0, 10.0),  # 平倉時刻
        ])
    }
    ex = Exchange(market, ExchangeConfig(init_balance=100_000.0, default_leverage=10.0))
    
    initial_equity = ex.account.equity()
    assert initial_equity == pytest.approx(100_000.0)
    
    # 開倉
    ex.place_order("BTCUSDT", 1.0, 50_000.0, True)
    ex.step()
    
    snapshot = ex.account.snapshot()
    # equity = balance (扣除保證金和手續費後) + unrealized_pnl (0)
    equity_after_open = snapshot.equity
    
    # K 線推進：價格上漲，產生未實現盈利
    ex.step()
    
    snapshot = ex.account.snapshot()
    expected_pnl = (55_000.0 - 50_000.0) * 1.0
    
    # 驗證 equity 包含未實現盈虧
    assert snapshot.unreal_pnl == pytest.approx(expected_pnl, rel=1e-3)
    assert snapshot.equity == pytest.approx(snapshot.balance + expected_pnl, rel=1e-3)
    
    # 平倉
    ex.close_position("BTCUSDT")
    ex.step()
    
    # 驗證：平倉後未實現盈虧歸零，equity = balance
    snapshot = ex.account.snapshot()
    assert snapshot.unreal_pnl == pytest.approx(0.0, abs=1.0)
    assert snapshot.equity == pytest.approx(snapshot.balance, rel=1e-3)
    
    # 驗證：最終 balance 應該反映盈利（扣除手續費）
    assert snapshot.balance > equity_after_open  # 有盈利


def test_kline_advance_with_partial_close():
    """
    測試 K 線推進時，部分平倉後剩餘倉位的未實現盈虧正確更新
    
    場景：
    1. 開倉 2 BTC @ 50k
    2. 價格上漲到 55k，未實現盈利 +10k
    3. 平倉 1 BTC，實現盈利 +5k
    4. 價格繼續上漲到 60k
    5. 驗證剩餘 1 BTC 的未實現盈虧為 +10k（60k - 50k）
    """
    market = {
        "BTCUSDT": make_df([
            (0, 50_000.0, 10.0),
            (60_000, 55_000.0, 10.0),
            (120_000, 55_000.0, 10.0),  # 部分平倉
            (180_000, 60_000.0, 10.0),  # 價格繼續上漲
        ])
    }
    ex = Exchange(market, ExchangeConfig(init_balance=200_000.0, default_leverage=10.0))
    
    # 開倉 2 BTC
    ex.place_order("BTCUSDT", 2.0, 50_000.0, True)
    ex.step()
    
    # K 線推進到 55k
    ex.step()
    
    positions = ex.account.get_all_positions()
    assert positions[0].unrealized_pnl == pytest.approx(10_000.0, rel=1e-3)  # (55k-50k) * 2
    
    # 部分平倉 1 BTC
    position_id = positions[0].id
    ex.place_order("BTCUSDT", 1.0, 55_000.0, False)  # 賣出 1 BTC
    ex.step()
    
    # 驗證：剩餘 1 BTC
    positions = ex.account.get_all_positions()
    assert len(positions) == 1
    assert positions[0].quantity == pytest.approx(1.0, rel=1e-3)
    
    # K 線推進到 60k
    ex.step()
    
    # 驗證：剩餘倉位的未實現盈虧更新到 +10k
    positions = ex.account.get_all_positions()
    expected_pnl = (60_000.0 - 50_000.0) * 1.0  # 10k
    assert positions[0].unrealized_pnl == pytest.approx(expected_pnl, rel=1e-3)
