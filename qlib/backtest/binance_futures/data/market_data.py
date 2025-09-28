from __future__ import annotations
import csv
from typing import List
from ..core.dto import KlineDto


class MarketData:
    """Load 1-minute klines from CSV and expose indexed access."""
    def __init__(self, symbol: str, csv_file: str):
        self.symbol = symbol
        self.klines: List[KlineDto] = []
        self._load_csv(csv_file)

    def _load_csv(self, csv_file: str):
        # Expected columns (same order as your C++ loader):
        # 0 openTs, 1 open, 2 high, 3 low, 4 close, 5 volume,
        # 6 closeTs, 7 quoteVol, 8 trades, 9 takerBB, 10 takerBQ
        with open(csv_file, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)  # skip header
            for row in reader:
                if len(row) < 11:
                    continue
                try:
                    self.klines.append(
                        KlineDto(
                            Timestamp=int(row[0]),
                            OpenPrice=float(row[1]),
                            HighPrice=float(row[2]),
                            LowPrice=float(row[3]),
                            ClosePrice=float(row[4]),
                            Volume=float(row[5]),
                            CloseTime=int(row[6]),
                            QuoteVolume=float(row[7]),
                            TradeCount=int(row[8]),
                            TakerBuyBaseVolume=float(row[9]),
                            TakerBuyQuoteVolume=float(row[10]),
                        )
                    )
                except ValueError:
                    continue
        # Ensure chronological order
        self.klines.sort(key=lambda k: k.Timestamp)

    def get_latest_kline(self) -> KlineDto:
        return self.klines[-1]

    def get_kline(self, index: int) -> KlineDto:
        return self.klines[index]

    def get_klines_count(self) -> int:
        return len(self.klines)

    # Iterators
    def __iter__(self):
        return iter(self.klines)
