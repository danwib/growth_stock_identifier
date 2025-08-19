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
