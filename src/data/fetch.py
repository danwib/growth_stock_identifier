# src/data/fetch.py
from __future__ import annotations
import pandas as pd

from .cache import load_cached, save_cache
from .providers.yf import YFinanceProvider
from .providers.alpha_vantage import AlphaVantageProvider
from .providers.alpaca import AlpacaProvider
from .utils_timeseries import restrict_rth, resample_ohlcv, ensure_utc_index

_YF = YFinanceProvider()
_AV = AlphaVantageProvider()
_APCA = AlpacaProvider()

def get_bars(symbol: str, start: str, end: str, interval: str, rth_only: bool) -> pd.DataFrame:
    """Fetch OHLCV bars with provider fallback and local caching.
    Returns a tz-aware (UTC) DataFrame with columns open/high/low/close/volume and DatetimeIndex.
    """
    # Try cache first
    df = load_cached(symbol, interval, start, end)
    if df is not None and not df.empty:
        df = ensure_utc_index(df)
        return df

    # Try providers in order: Alpaca (if keys) → yfinance → AlphaVantage
    for provider in (_APCA, _YF, _AV):
        try:
            df = provider.get_bars(symbol, start, end, interval)
            if df is not None and not df.empty:
                break
        except Exception:
            df = None
    if df is None or df.empty:
        return pd.DataFrame(columns=["open","high","low","close","volume"]).astype(float)

    # Normalize timezone and apply optional RTH filter
    df = ensure_utc_index(df)
    if rth_only and interval in ("1min","5min","15min","1h"):
        df = restrict_rth(df)

    # Resample to requested rule (idempotent if already matching)
    rule_map = {"1min":"1min","5min":"5min","15min":"15min","1h":"1H","1d":"1D"}
    df = resample_ohlcv(df, rule_map.get(interval, interval))

    # Save cache and return
    try:
        if len(df) > 0:
            save_cache(symbol, interval, start, end, df)
    except Exception:
        pass
    return df
