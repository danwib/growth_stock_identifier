# src/data/feature_pipeline.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from .utils_timeseries import future_log_return

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def engineer_basic_features(df: pd.DataFrame, horizon_bars: int = 20, price_col: str = "close") -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Compute simple technical features and a forward-return label.
    Returns (X_scaled, y, meta)
    - X_scaled: np.ndarray of shape [n_samples, n_features]
    - y: np.ndarray of shape [n_samples]
    - meta: {feature_cols, target_col, scaler_mean_, scaler_scale_}
    """
    if df is None or df.empty:
        return np.zeros((0,0)), np.zeros((0,)), {"feature_cols":[], "target_col":"y", "scaler_mean_":[], "scaler_scale_":[]}

    close = df[price_col].astype(float)
    ret1 = close.pct_change(1)
    ret5 = close.pct_change(5)
    ret20 = close.pct_change(20)
    mom20 = close / close.shift(20) - 1.0
    vol20 = ret1.rolling(20).std()
    vol60 = ret1.rolling(60).std()
    rsi14 = rsi(close, 14)
    hi20 = close.rolling(20).max()
    lo20 = close.rolling(20).min()
    dd20 = close / hi20 - 1.0

    feats = pd.DataFrame({
        "ret1": ret1,
        "ret5": ret5,
        "ret20": ret20,
        "mom20": mom20,
        "vol20": vol20,
        "vol60": vol60,
        "rsi14": rsi14,
        "dd20": dd20,
    }, index=df.index)

    # Label: forward log return
    y = future_log_return(df, horizon_bars=horizon_bars, price_col=price_col)

    # Drop rows with NaNs in features or label
    valid = feats.dropna().index.intersection(y.dropna().index)
    feats = feats.loc[valid]
    y = y.loc[valid]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feats.values)
    meta = {
        "feature_cols": feats.columns.tolist(),
        "target_col": "y",
        "scaler_mean_": scaler.mean_.tolist(),
        "scaler_scale_": scaler.scale_.tolist(),
    }
    return X_scaled, y.values.astype(float), meta
