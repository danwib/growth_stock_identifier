# src/data/providers/yf.py
from __future__ import annotations
import pandas as pd
import yfinance as yf

from .base import MarketDataProvider

_INTERVAL_MAP = {
    "1min": "1m",
    "5min": "5m",
    "15min": "15m",
    "1h": "60m",
    "1d": "1d",
}

class YFinanceProvider(MarketDataProvider):
    def get_bars(self, symbol: str, start: str, end: str, interval: str) -> pd.DataFrame:
        yf_interval = _INTERVAL_MAP.get(interval, "1d")
        # yfinance returns naive index for daily, tz-aware for intraday; we'll normalize later
        df = yf.download(symbol, start=start, end=end, interval=yf_interval, auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)
        # Standardize columns
        cols = {"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"}
        df = df.rename(columns=cols)[["open","high","low","close","volume"]]
        # yfinance index may be naive; keep as is, upstream will localize to UTC
        return df
