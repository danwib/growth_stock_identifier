# Growth Stock Identifier (Starter, Clean)

- **Code package:** `src/data/` (import as `from data.fetch import get_bars`)
- **Storage folder:** `data/` at repo root for outputs (`data/raw`, `data/processed`)

This repo wraps your data ingestion scaffold so you can fetch bars via Alpaca→yfinance→AlphaVantage (with caching)
and build basic features. Next, you can add LambdaMART ranking.

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env  # optional keys for Alpaca/AlphaVantage

# Fetch sample daily bars to data/raw
python -m src.scripts.fetch_bars --symbols AAPL,MSFT --start 2023-01-01 --end 2024-12-31 --interval 1d

# Build basic features (placeholder labels) to data/processed
python -m src.scripts.build_features --symbols AAPL,MSFT --start 2023-01-01 --end 2024-12-31 --base-interval 1d --label-horizons 20
```


## Incremental pipeline (Bronze → Silver → Gold)

This repo includes daily jobs to maintain an incremental, point‑in‑time dataset:

- **Bronze (raw bars):** `src/jobs/delta_ingest.py` → updates `data/raw_bars/interval=1d/symbol=SYM/bars.parquet`
- **Silver (features):** `src/jobs/feature_update.py` → writes `data/features_daily/date=YYYY-MM-DD/part.parquet`
- **Gold (labels + panel):**
  - `src/jobs/label_maturer.py` → writes `data/labels_daily/date=YYYY-MM-DD/part.parquet`
  - `src/jobs/build_panel_monthly.py` → joins features+labels at month‑end into `data/panel/panel.parquet` and `groups.json`

### Run the pipeline
```bash
# 1) Ingest/refresh daily bars (Bronze)
python -m src.jobs.delta_ingest --symbols AAPL,MSFT --start 2018-01-01

# 2) Build/refresh features (Silver)
python -m src.jobs.feature_update --horizon 126   # ≈6 months

# 3) Mature labels once horizon is known (Gold)
python -m src.jobs.label_maturer --horizon 126

# 4) Build monthly cross‑sectional panel + groups for LambdaMART
python -m src.jobs.build_panel_monthly --out data/panel
```
Then train your LightGBM LambdaRank model on `data/panel/panel.parquet` using the group counts in `data/panel/groups.json` (one group per snapshot date).
