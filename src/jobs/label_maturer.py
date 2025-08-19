# src/jobs/label_maturer.py
from __future__ import annotations
import argparse, sys, pathlib
import pandas as pd
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
sys.path.insert(0, str(SRC_ROOT))

def main():
    ap = argparse.ArgumentParser(description="Compute forward-return labels per (date,symbol) once horizon has matured.")
    ap.add_argument("--horizon", type=int, default=126, help="Forward horizon in trading days (≈6 months)")
    args = ap.parse_args()

    raw_base = pathlib.Path("data/raw_bars/interval=1d")
    out_base = pathlib.Path("data/labels_daily")
    out_base.mkdir(parents=True, exist_ok=True)

    for sym_dir in sorted(raw_base.glob("symbol=*")):
        sym = sym_dir.name.split("=",1)[1]
        table_path = sym_dir / "bars.parquet"
        if not table_path.exists():
            continue
        df = pd.read_parquet(table_path).sort_index()
        close = pd.Series(df["close"])
        close_fwd = close.shift(-args.horizon)
        y = (np.log(close_fwd) - np.log(close)).dropna()
        # Emit per date partition with one row
        tmp = pd.DataFrame({"symbol": sym, "y": y})
        for date, chunk in tmp.groupby(tmp.index.date):
            date_dir = out_base / f"date={str(date)}"
            date_dir.mkdir(parents=True, exist_ok=True)
            out_path = date_dir / "part.parquet"
            if out_path.exists():
                exist = pd.read_parquet(out_path)
                exist = exist[exist["symbol"] != sym]
                all_df = pd.concat([exist, chunk]).drop_duplicates(subset=["symbol"], keep="last")
                all_df.to_parquet(out_path)
            else:
                chunk.to_parquet(out_path)
        print(f"[ok] labels for {sym}")

if __name__ == "__main__":
    main()
