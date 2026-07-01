from __future__ import annotations

from app.db.session import SessionLocal
from app.services.backtest_service import execute_backtest
from app.services.training_service import execute_training_job
from app.workers.celery_app import celery_app


@celery_app.task(name="weather_ml.train_candidate")
def train_candidate_task(job_id: str) -> str:
    with SessionLocal() as db:
        model = execute_training_job(db, job_id)
        return model.id


@celery_app.task(name="weather_ml.run_backtest")
def run_backtest_task(run_id: str) -> str:
    with SessionLocal() as db:
        run = execute_backtest(db, run_id)
        return run.id
