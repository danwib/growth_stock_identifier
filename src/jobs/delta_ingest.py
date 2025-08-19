# src/jobs/delta_ingest.py
from __future__ import annotations
import argparse, sys, pathlib, json
from datetime import datetime, timedelta
import pandas as pd

THIS_DIR = pathlib.Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
sys.path.insert(0, str(SRC_ROOT))

from data.fetch import get_bars

def main():
    ap = argparse.ArgumentParser(description="Incrementally fetch daily bars and maintain per-symbol parquet tables.")
    ap.add_argument("--symbols", required=True, help="Comma-separated tickers")
    ap.add_argument("--start", required=False, default=None, help="ISO date; if omitted, derive from existing tables")
    ap.add_argument("--end", required=False, default=None, help="ISO date; default today")
    ap.add_argument("--rth-only", action="store_true")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    end = args.end or datetime.utcnow().date().isoformat()

    base = pathlib.Path("data/raw_bars/interval=1d")
    for sym in symbols:
        sym_dir = base / f"symbol={sym}"
        sym_dir.mkdir(parents=True, exist_ok=True)
        table_path = sym_dir / "bars.parquet"

        # Determine start based on existing data
        if table_path.exists():
            df_existing = pd.read_parquet(table_path)
            last_ts = pd.to_datetime(df_existing.index).max().tz_localize("UTC") if df_existing.index.tz is None else pd.to_datetime(df_existing.index).max()
            start_dt = (last_ts + pd.Timedelta(days=-5)).date().isoformat()  # small overlap to allow corrections
        else:
            start_dt = args.start or "2015-01-01"

        df = get_bars(sym, start_dt, end, "1d", args.rth_only)
        if df is None or df.empty:
            print(f"[warn] no data for {sym}")
            continue

        # Append & de-dup
        if table_path.exists():
            df_all = pd.read_parquet(table_path)
            df_all = pd.concat([df_all, df]).sort_index()
            df_all = df_all[~df_all.index.duplicated(keep="last")]
        else:
            df_all = df
        df_all.to_parquet(table_path)
        print(f"[ok] {sym}: rows={len(df_all)} -> {table_path}")

if __name__ == "__main__":
    main()
