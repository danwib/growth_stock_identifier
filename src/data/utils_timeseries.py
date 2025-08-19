# src/data/utils_timeseries.py
from __future__ import annotations
import numpy as np
import pandas as pd

NY_TZ = "America/New_York"

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def restrict_rth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep Regular Trading Hours (NYSE): 09:30–16:00 America/New_York.
    Assumes df.index is tz-aware UTC; returns filtered UTC-indexed frame.
    """
    if df is None or df.empty:
        return df
    idx_local = df.index.tz_convert(NY_TZ)
    start_t = pd.Timestamp("09:30", tz=NY_TZ).time()
    end_t = pd.Timestamp("16:00", tz=NY_TZ).time()
    mask = (idx_local.time >= start_t) & (idx_local.time <= end_t)
    return df.loc[mask]

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    Resample to a pandas rule, e.g. "15min" -> "15min", "1h" -> "1H", "1d" -> "1D".
    Keeps OHLCV semantics.
    """
    if df is None or df.empty:
        return df
    # Normalize rule capitalization
    rule_map = {"1min": "1min", "5min": "5min", "15min": "15min", "1h": "1H", "1d": "1D"}
    r = rule_map.get(rule, rule)
    out = (
        df.resample(r, origin="start")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    return out

def future_log_return(df: pd.DataFrame, horizon_bars: int, price_col: str = "close") -> pd.Series:
    """Log-return over 'horizon_bars' into the future: log(P_{t+h}/P_t)
    Assumes regular sampling; returns aligned with current row index (NaN at tail).
    """
    if df is None or df.empty:
        return pd.Series(dtype=float)
    p0 = pd.Series(df[price_col])
    p1 = p0.shift(-horizon_bars)
    return np.log(p1) - np.log(p0)
