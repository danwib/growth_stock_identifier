# src/scripts/fetch_bars.py
from __future__ import annotations
import argparse, sys, pathlib
from dotenv import load_dotenv

THIS_DIR = pathlib.Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
sys.path.insert(0, str(SRC_ROOT))

from data.fetch import get_bars  # noqa: E402

def main():
    load_dotenv()
    ap = argparse.ArgumentParser(description="Fetch bars and write to data/raw as parquet.")
    ap.add_argument("--symbols", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--interval", default="1d", choices=["1min","5min","15min","1h","1d"])
    ap.add_argument("--rth-only", action="store_true")
    args = ap.parse_args()

    out_dir = pathlib.Path("data/raw"); out_dir.mkdir(parents=True, exist_ok=True)

    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        df = get_bars(sym, args.start, args.end, args.interval, args.rth_only)
        if df is None or df.empty:
            print(f"[warn] {sym}: no data")
            continue
        out = out_dir / f"{sym}_{args.interval}_{args.start}_{args.end}.parquet"
        df.to_parquet(out)
        print(f"[ok] wrote {out} rows={len(df)}")

if __name__ == "__main__":
    main()
