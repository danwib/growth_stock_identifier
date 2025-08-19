# src/jobs/build_panel_monthly.py
from __future__ import annotations
import argparse, sys, pathlib
import pandas as pd
import numpy as np

THIS_DIR = pathlib.Path(__file__).resolve().parent
SRC_ROOT = THIS_DIR.parent
sys.path.insert(0, str(SRC_ROOT))

def month_end_dates(path: pathlib.Path) -> list:
    # derive available dates from features_daily partitions
    dates = []
    for d in path.glob("date=*"):
        try:
            dates.append(pd.to_datetime(d.name.split("=",1)[1]))
        except Exception:
            pass
    if not dates:
        return []
    s = pd.Series(1, index=pd.DatetimeIndex(dates).normalize()).sort_index()
    # pick last calendar day present per month
    return s.groupby([s.index.year, s.index.month]).apply(lambda x: x.index.max()).tolist()

def main():
    ap = argparse.ArgumentParser(description="Join features + matured labels into a cross-sectional panel and groups for LambdaMART.")
    ap.add_argument("--out", default="data/panel", help="Output directory")
    args = ap.parse_args()

    feat_base = pathlib.Path("data/features_daily")
    lab_base = pathlib.Path("data/labels_daily")
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = month_end_dates(feat_base)
    if not dates:
        print("[warn] No features_daily partitions found.")
        return

    panel_rows = []
    groups = []
    for d in dates:
        d_str = str(d.date())
        f_path = feat_base / f"date={d_str}" / "part.parquet"
        l_path = lab_base / f"date={d_str}" / "part.parquet"
        if not f_path.exists() or not l_path.exists():
            continue
        F = pd.read_parquet(f_path)
        L = pd.read_parquet(l_path)
        df = F.merge(L[["symbol","y"]], on="symbol", how="inner")
        df["date"] = pd.to_datetime(d).normalize()
        if df.empty:
            continue
        groups.append(len(df))
        panel_rows.append(df)

    if not panel_rows:
        print("[warn] No overlapping features+labels dates available.")
        return

    panel = pd.concat(panel_rows, ignore_index=True)
    panel.to_parquet(out_dir / "panel.parquet")
    # Save groups as JSON list
    import json
    with open(out_dir / "groups.json","w") as f:
        json.dump(groups, f)

    print(f"[ok] panel rows={len(panel)} dates={len(groups)} -> {out_dir}")

if __name__ == "__main__":
    main()
