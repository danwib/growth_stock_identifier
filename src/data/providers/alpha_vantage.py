# src/data/providers/alpha_vantage.py
from __future__ import annotations
import os, pandas as pd
from .base import MarketDataProvider

class AlphaVantageProvider(MarketDataProvider):
    def __init__(self):
        self.key = os.getenv("ALPHA_VANTAGE_API_KEY")
    def get_bars(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        # TODO: Implement via requests to Alpha Vantage; default to empty to allow fallback
        return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)
