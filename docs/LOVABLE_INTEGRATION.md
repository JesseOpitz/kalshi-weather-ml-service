# Lovable integration

Add these values to Lovable Cloud or Supabase Edge Function secrets:

- `ML_SERVICE_URL`: the deployed FastAPI base URL without a trailing route.
- `ML_SERVICE_API_KEY`: one value configured in `ML_SERVICE_API_KEYS` on the service.
- `ML_SERVICE_ADMIN_API_KEY`: the separate administrator key used for training and promotion actions.

The browser must never receive either key. Lovable should call a server-side Edge Function, and the Edge Function should add the appropriate header:

- `X-API-Key` for model health, station listing, datasets, prediction, model versions, and historical evaluation.
- `X-Admin-Key` for station changes, weather-history ingestion, candidate training, and model promotion.

## Core routes

- `GET /health`
- `GET /ready`
- `GET /v1/model-health`
- `GET /v1/model-versions`
- `GET /v1/stations`
- `POST /v1/stations`
- `GET /v1/datasets`
- `POST /v1/weather-data/open-meteo`
- `POST /v1/train-candidate`
- `POST /v1/promote-model`
- `POST /v1/predict-market`
- `POST /v1/backtest-strategy`

Treat `trading_eligible=false` as an absolute block. This service performs analytics and model lifecycle management only; it does not submit orders.
