# src/data/cache.py
from __future__ import annotations
import os, hashlib, pandas as pd, pyarrow.parquet as pq, pyarrow as pa

_CACHE_DIR = os.path.join("data", "_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

def _key(symbol: str, interval: str, start: str, end: str) -> str:
    raw = f"{symbol}|{interval}|{start}|{end}".encode()
    return hashlib.sha1(raw).hexdigest() + ".parquet"

def load_cached(symbol: str, interval: str, start: str, end: str) -> pd.DataFrame | None:
    path = os.path.join(_CACHE_DIR, _key(symbol, interval, start, end))
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

def save_cache(symbol: str, interval: str, start: str, end: str, df: pd.DataFrame) -> None:
    path = os.path.join(_CACHE_DIR, _key(symbol, interval, start, end))
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path)
