# src/data/providers/alpaca.py
from __future__ import annotations
import os, pandas as pd
from .base import MarketDataProvider

class AlpacaProvider(MarketDataProvider):
    def __init__(self):
        self.key = os.getenv("APCA_API_KEY_ID")
        self.secret = os.getenv("APCA_API_SECRET_KEY")
        self.base_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        # NOTE: implement real Alpaca fetch if desired; stub returns empty without keys
    def get_bars(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        if not (self.key and self.secret):
            return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)
        # TODO: Implement using alpaca-py (omitted here). Returning empty to fall back.
        return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)
