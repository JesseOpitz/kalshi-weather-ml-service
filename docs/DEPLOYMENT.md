# Deployment

## Local setup

1. Create a virtual environment with Python 3.12.
2. Install the project with `python -m pip install -e '.[dev]'`.
3. Configure the service environment variables in your hosting platform or local shell.
4. Run `alembic upgrade head`.
5. Start the API with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.

The Dockerfile performs the same package installation and starts Uvicorn as a non-root user.

## Required configuration

- `ML_SERVICE_API_KEYS`: comma-separated service keys.
- `ADMIN_API_KEY`: a separate administrator key.
- `DATABASE_URL`: SQLite for a pilot or PostgreSQL for a durable deployment.
- `MODEL_STORE_URI`: a persistent directory mounted into the service.

## Render

The included `render.yaml` creates the web service. Configure its secret values in the Render dashboard. Mount persistent storage for the database and model directory when using SQLite/local model files, or use a managed PostgreSQL database.

## Verification

- `/health` confirms the process is running.
- `/ready` confirms database access.
- `/v1/model-health` confirms whether a champion model is loaded and verified.
