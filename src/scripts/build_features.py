# src/scripts/build_features.py
from __future__ import annotations
import argparse, sys, pathlib, json
from dotenv import load_dotenv
import pandas as pd

THIS_DIR = pathlib.Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
sys.path.insert(0, str(SRC_ROOT))

from data.feature_pipeline import engineer_basic_features  # noqa: E402
from data.fetch import get_bars  # noqa: E402

def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Build basic features/labels from bars.")
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--base-interval", default="1d", choices=["1min","5min","15min","1h","1d"])
    ap.add_argument("--label-horizons", default="20")
    ap.add_argument("--rth-only", action="store_true")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    horizon = int(args.label_horizons.split(",")[0])

    X_list, y_list, idx_list, sym_id_list = [], [], [], []
    for sid, sym in enumerate(symbols):
        df = get_bars(sym, args.start, args.end, args.base_interval, args.rth_only)
        if df is None or df.empty:
            continue
        X_scaled, y, meta = engineer_basic_features(df, horizon_bars=horizon, price_col="close")
        n = len(y)
        # collect
        import numpy as np
        X_list.append(pd.DataFrame(X_scaled, index=df.index[:n]))
        y_list.append(pd.Series(np.asarray(y).reshape(-1), index=df.index[:n], name="y"))
        idx_list.append(df.index[:n])
        sym_id_list.append(pd.Series([sid]*n, index=df.index[:n], name="symbol_id"))

    if not X_list:
        print("[warn] no data produced")
        return

    X = pd.concat(X_list).reset_index(drop=True)
    y = pd.concat(y_list).reset_index(drop=True)
    symbol_ids = pd.concat(sym_id_list).reset_index(drop=True)
    times = pd.to_datetime(pd.concat(idx_list)).view("int64")//10**9
    times = pd.Series(times, name="epoch_utc").reset_index(drop=True)

    out_dir = pathlib.Path("data/processed"); out_dir.mkdir(parents=True, exist_ok=True)
    X.to_parquet(out_dir/"features.parquet")
    y.to_frame().to_parquet(out_dir/"labels.parquet")
    symbol_ids.to_frame().to_parquet(out_dir/"symbol_ids.parquet")
    times.to_frame().to_parquet(out_dir/"times.parquet")

    with open(out_dir/"meta.json","w") as f:
        json.dump(meta, f, indent=2)

    print(f"[ok] wrote processed artifacts to {out_dir}")

if __name__ == "__main__":
    main()
