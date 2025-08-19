# src/scripts/build_features.py
from __future__ import annotations
import argparse, sys, pathlib, json
from dotenv import load_dotenv
import pandas as pd
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
sys.path.insert(0, str(SRC_ROOT))

from data.feature_pipeline import engineer_basic_features
from data.fetch import get_bars

def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Build basic features/labels from fetched bars.")
    ap.add_argument("--symbols", required=True, help="Comma-separated tickers, e.g., AAPL,MSFT")
    ap.add_argument("--start", required=True, help="UTC ISO date (e.g., 2023-01-01)")
    ap.add_argument("--end", required=True, help="UTC ISO date (e.g., 2024-12-31)")
    ap.add_argument("--base-interval", dest="base_interval", default="1d", choices=["1min","5min","15min","1h","1d"])
    ap.add_argument("--label-horizons", default="20", help="Horizon in bars at BASE interval (e.g., 20≈1 trading month @1d)")
    ap.add_argument("--rth-only", action="store_true")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    horizon = int(args.label_horizons.split(",")[0])

    framesX, framesY, sym_ids, times = [], [], [], []
    for sid, sym in enumerate(symbols):
        df = get_bars(sym, args.start, args.end, args.base_interval, args.rth_only)
        X, y, meta = engineer_basic_features(df, horizon_bars=horizon, price_col="close")
        if len(y) == 0:
            print(f"[warn] No usable rows for {sym}; skipping.")
            continue
        # Align with feature rows by taking the last n indices of df after dropping NaNs in engineer
        # We can reconstruct index alignment by re-computing and dropping NaNs on the raw features:
        # For simplicity: infer n rows from X, and take the last n timestamps from df.index[:-horizon]
        n = len(y)
        # Ensure the last n indices align with earlier times (exclude tail horizon)
        idx = df.index[:len(df)-horizon][-n:]
        framesX.append(pd.DataFrame(X, index=idx))
        framesY.append(pd.Series(y, index=idx, name="y"))
        sym_ids.append(pd.Series([sid]*n, index=idx, name="symbol_id"))
        times.append(pd.Series(idx.view('int64')//10**9, index=idx, name="epoch_utc"))

    if not framesX:
        print("[error] No data produced.")
        return

    X_all = pd.concat(framesX).reset_index(drop=True)
    y_all = pd.concat(framesY).reset_index(drop=True).to_frame()
    sym_all = pd.concat(sym_ids).reset_index(drop=True).to_frame()
    t_all = pd.concat(times).reset_index(drop=True).to_frame()

    out_dir = pathlib.Path("data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_all.to_parquet(out_dir/"features.parquet")
    y_all.to_parquet(out_dir/"labels.parquet")
    sym_all.to_parquet(out_dir/"symbol_ids.parquet")
    t_all.to_parquet(out_dir/"times.parquet")

    # Save meta from the last symbol (features are the same schema)
    with open(out_dir/"meta.json","w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] Wrote features/labels/meta to {out_dir}")

if __name__ == "__main__":
    main()
