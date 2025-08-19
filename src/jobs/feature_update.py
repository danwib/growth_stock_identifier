# src/jobs/feature_update.py
from __future__ import annotations
import argparse, sys, pathlib
import pandas as pd
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
sys.path.insert(0, str(SRC_ROOT))

from data.feature_pipeline import engineer_basic_features

def main():
    ap = argparse.ArgumentParser(description="Build daily features from raw_bars tables into partitioned features_daily.")
    ap.add_argument("--horizon", type=int, default=126, help="Forward horizon in trading days (≈6 months)")
    args = ap.parse_args()

    raw_base = pathlib.Path("data/raw_bars/interval=1d")
    out_base = pathlib.Path("data/features_daily")
    out_base.mkdir(parents=True, exist_ok=True)

    for sym_dir in sorted(raw_base.glob("symbol=*")):
        sym = sym_dir.name.split("=",1)[1]
        table_path = sym_dir / "bars.parquet"
        if not table_path.exists():
            continue
        df = pd.read_parquet(table_path)
        X, y, meta = engineer_basic_features(df, horizon_bars=args.horizon, price_col="close")
        if len(y)==0:
            continue
        n = len(y)
        idx = df.index[:len(df)-args.horizon][-n:]
        feat = pd.DataFrame(X, index=idx, columns=meta["feature_cols"]).assign(symbol=sym)
        for date, chunk in feat.groupby(feat.index.date):
            date_dir = out_base / f"date={str(date)}"
            date_dir.mkdir(parents=True, exist_ok=True)
            # append mode: read existing, concat, drop dup by symbol
            out_path = date_dir / "part.parquet"
            if out_path.exists():
                exist = pd.read_parquet(out_path)
                exist = exist[exist["symbol"] != sym]
                chunk.to_parquet(out_path)
                all_df = pd.concat([exist, chunk]).drop_duplicates(subset=["symbol"], keep="last")
                all_df.to_parquet(out_path)
            else:
                chunk.to_parquet(out_path)
        print(f"[ok] features for {sym}")

if __name__ == "__main__":
    main()
