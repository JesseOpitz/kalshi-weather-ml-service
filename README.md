# Kalshi Weather ML Service

A deployable FastAPI service for weather-history ingestion, station-aware feature engineering, calibrated probability prediction, model training, model promotion, and historical evaluation.

The service is intentionally separated from order execution. It produces model probabilities and eligibility signals; it does not submit, cancel, or manage Kalshi orders and does not promise profits.

## Included

- FastAPI health and readiness endpoints
- Separate service and administrator API keys
- SQLite or PostgreSQL persistence through SQLAlchemy
- Alembic database migration
- Station registry
- Open-Meteo historical weather bootstrap ingestion
- Timestamp-aware feature engineering
- Gradient-boosted median and quantile regressors
- Out-of-fold isotonic probability calibration
- Complete bracket and tail probabilities
- Prediction intervals, feature-quality scoring, explanations, and historical analogues
- Candidate training and explicit champion promotion
- Checksum-verified local model storage
- Historical evaluation with fee, slippage, uncertainty, and exposure assumptions
- Docker and Render deployment files
- Lovable integration documentation
- GitHub Actions compile, lint, and test checks

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'

export ML_SERVICE_API_KEYS="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export ADMIN_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(48))')"
export DATABASE_URL="sqlite:///./data/weather_ml.sqlite3"
export MODEL_STORE_URI="./artifacts"

weather-ml init-db
weather-ml seed-stations
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API.

## Core workflow

1. Register the exact weather station used by the contract.
2. Create a bootstrap historical dataset through `/v1/weather-data/open-meteo`.
3. Train a candidate through `/v1/train-candidate`.
4. Review its out-of-sample metrics.
5. Promote it explicitly through `/v1/promote-model`.
6. Request calibrated probabilities through `/v1/predict-market`.
7. Evaluate historical strategy settings through `/v1/backtest-strategy`.

Open-Meteo data is suitable for bootstrapping, not a substitute for exact official settlement-station observations and timestamped forecast archives. Replace or augment the bootstrap labels before relying on a model operationally.

## API routes

| Method | Route | Access |
|---|---|---|
| GET | `/health` | Public |
| GET | `/ready` | Public |
| GET | `/v1/model-health` | Service key |
| GET | `/v1/model-versions` | Service key |
| GET | `/v1/stations` | Service key |
| POST | `/v1/stations` | Administrator key |
| GET | `/v1/datasets` | Service key |
| POST | `/v1/weather-data/open-meteo` | Administrator key |
| POST | `/v1/train-candidate` | Administrator key |
| POST | `/v1/promote-model` | Administrator key |
| POST | `/v1/predict-market` | Service key |
| POST | `/v1/backtest-strategy` | Service key |

Use `X-API-Key` for service-key routes and `X-Admin-Key` for administrator routes. Keep both keys in server-side Lovable or Supabase secrets; never expose them to the browser.

## Fail-closed behavior

- No champion model: prediction is unavailable.
- Invalid model checksum: loading fails.
- Stale or incomplete input data: `trading_eligible` is false.
- Unseen station or market type: `trading_eligible` is false.
- Non-champion model: `trading_eligible` is false.

See `docs/LOVABLE_INTEGRATION.md`, `docs/DEPLOYMENT.md`, and `docs/TRAINING_DATA_CONTRACT.md`.
