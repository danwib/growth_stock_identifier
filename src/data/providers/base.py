# src/data/providers/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd

class MarketDataProvider(ABC):
    @abstractmethod
    def get_bars(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        ...
