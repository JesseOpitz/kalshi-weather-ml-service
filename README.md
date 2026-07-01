# Kalshi Weather ML Service

A deployable Python service for historical weather-model training, station-aware feature engineering, probabilistic temperature prediction, bracket probability calibration, backtesting, model registry, champion/challenger evaluation, outcome scoring, and secure Lovable integration.

This repository is an actual runnable service, not a UI mock. It does **not** claim guaranteed trading profit and it deliberately refuses production-eligible predictions until an administrator trains and promotes a model.

## Included

- FastAPI with OpenAPI documentation
- Separate API and administrator secrets
- PostgreSQL/SQLite persistence through SQLAlchemy
- Alembic migrations
- Redis/Celery background training and backtests
- CSV and Parquet ingestion
- Open-Meteo bootstrap historical dataset builder
- Exact-station dataset upload path for settlement-grade data
- Time-ordered model evaluation
- Gradient-boosted median and quantile regressors
- Out-of-fold isotonic event-probability calibration
- Bracket and tail probabilities
- Prediction intervals and data-quality scoring
- Local perturbation explanations
- Historical analogue retrieval
- Model artifact SHA-256 verification
- Local or S3-compatible artifact storage
- Candidate, champion, retirement, promotion, and rollback states
- Backtesting with fees, slippage, uncertainty buffer, exposure cap, and P&L
- Official-outcome recording with Brier score, log loss, and temperature error
- Structured JSON logging, request IDs, rate limiting, health checks, and Prometheus metrics
- Dockerfile, Docker Compose, Render blueprint, CI, tests, Lovable Edge Function, and typed client helper

## Fast start

```bash
cp .env.example .env
python scripts/generate_api_key.py   # use for ML_SERVICE_API_KEYS
python scripts/generate_api_key.py   # use a different value for ADMIN_API_KEY
docker compose up --build
```

Then visit `http://localhost:8000/docs`.

For a lightweight local Python run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
weather-ml init-db
weather-ml seed-stations
uvicorn app.main:app --reload
```

## End-to-end production workflow

1. Register the exact settlement station.
2. Build a bootstrap dataset or upload timestamp-correct official station data.
3. Start a candidate training job.
4. Poll the job until complete.
5. Inspect out-of-sample MAE, RMSE, interval coverage, Brier score, and log loss.
6. Backtest against historical Kalshi prices when the dataset includes contract bounds and executable asks.
7. Evaluate the challenger against the champion.
8. Promote only after human approval and shadow validation.
9. Send live weather inputs to `/v1/predict-market`.
10. Let the separate Kalshi execution/risk engine decide whether an order is permitted.
11. Record official outcomes to maintain a permanent performance history.

## Core API

| Method | Route | Key | Purpose |
|---|---|---|---|
| GET | `/health` | none | Process health |
| GET | `/ready` | none | Database readiness |
| GET | `/v1/model-health` | API | Champion and artifact health |
| GET | `/v1/model-versions` | API | Registry listing |
| POST | `/v1/predict-market` | API | Calibrated market prediction |
| POST | `/v1/explain-prediction` | API | Prediction plus drivers/analogues |
| GET/POST | `/v1/stations` | API/Admin | List or register stations |
| GET | `/v1/datasets` | API | Dataset registry |
| POST | `/v1/datasets/upload` | Admin | Upload CSV/Parquet |
| POST | `/v1/datasets/open-meteo` | Admin | Build bootstrap dataset |
| POST | `/v1/train-candidate` | Admin | Queue candidate training |
| GET | `/v1/training-jobs/{id}` | Admin | Poll training |
| POST | `/v1/backtest-strategy` | API | Queue backtest |
| GET | `/v1/backtest-runs/{id}` | API | Poll backtest |
| POST | `/v1/evaluate-candidate` | Admin | Compare champion and challenger |
| POST | `/v1/promote-model` | Admin | Promote candidate |
| POST | `/v1/rollback-model` | Admin | Restore prior model |
| POST | `/v1/record-outcome` | Admin | Score official settlement |

Nested REST-style aliases are also available under `/v1/models`, `/v1/training`, `/v1/backtests`, and `/v1/outcomes`.

## Example prediction

```bash
curl -X POST "$ML_SERVICE_URL/v1/predict-market" \
  -H "X-API-Key: $ML_SERVICE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "market_id": "internal-market-id",
    "market_ticker": "KXHIGHNY-EXAMPLE",
    "station_code": "KNYC",
    "market_type": "daily_high_temperature",
    "target_time": "2026-07-03T21:00:00Z",
    "lower_bound": 90,
    "upper_bound": 92,
    "unit": "F",
    "weather_forecasts": [
      {"predicted_high": 91.4},
      {"predicted_high": 92.1},
      {"predicted_high": 90.9}
    ],
    "weather_observations": [
      {
        "observation_time": "2026-07-03T15:00:00Z",
        "temperature": 85.2,
        "dew_point": 69.1,
        "humidity": 58,
        "wind_speed": 8.2,
        "wind_direction": 220,
        "cloud_cover": 25,
        "pressure": 1011.8
      }
    ],
    "brackets": [
      {"label": "89 or below", "upper_bound": 90},
      {"label": "90-91", "lower_bound": 90, "upper_bound": 92},
      {"label": "92-93", "lower_bound": 92, "upper_bound": 94},
      {"label": "94 or above", "lower_bound": 94}
    ]
  }'
```

## Model behavior

The service trains three gradient-boosted regressors: median, 10th percentile, and 90th percentile. Time-ordered out-of-fold predictions estimate residual uncertainty and fit an isotonic event-probability calibrator. At inference, the service returns a temperature distribution, contract probability, optional complete bracket probabilities, uncertainty interval, feature-quality score, local drivers, and nearest historical analogues.

This is deliberately auditable. A later neural or external numerical-weather model can replace the bundle internals without changing the Lovable contract.

## Data quality

The Open-Meteo route is immediately usable for bootstrap training, but its labels are gridded reanalysis rather than a guarantee of exact Kalshi settlement-station measurements. Serious production training should upload official station observations and timestamped historical forecasts. See `docs/TRAINING_DATA_CONTRACT.md`.

## Production safety

- No champion: prediction fails with `503`.
- Invalid artifact checksum: model health becomes degraded and inference fails.
- Non-champion model: `trading_eligible=false`.
- Missing critical features, stale observations, unseen station, or unseen market type: `trading_eligible=false`.
- Promotion and rollback require the independent admin key and create audit events.
- The service never submits Kalshi orders. Your existing trading engine must enforce account authorization and deterministic risk limits.

## Documentation

- `docs/LOVABLE_INTEGRATION.md`
- `docs/TRAINING_DATA_CONTRACT.md`
- `docs/DEPLOYMENT.md`
- `docs/SECURITY.md`
- `http://host/docs` for interactive OpenAPI

## Tests

```bash
python -m pytest -q
```

The included suite trains a real small model, promotes it, requests a calibrated prediction through FastAPI, and records an official outcome.
