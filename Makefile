.PHONY: install dev test lint format migrate run worker key

install:
	python -m pip install -e '.[dev]'

dev:
	uvicorn app.main:app --reload --port 8000

test:
	python -m pytest --cov=app --cov-report=term-missing

lint:
	ruff check app tests
	mypy app

format:
	ruff format app tests
	ruff check --fix app tests

migrate:
	alembic upgrade head

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

worker:
	celery -A app.workers.celery_app:celery_app worker --loglevel=INFO

key:
	python scripts/generate_api_key.py
