from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.db.models import BacktestRun, DatasetManifest
from app.ml.backtest import run_backtest
from app.services.dataset_service import load_dataset
from app.services.model_service import load_model_bundle


def execute_backtest(db: Session, run_id: str) -> BacktestRun:
    run = db.get(BacktestRun, run_id)
    if run is None:
        raise LookupError(f"Backtest run not found: {run_id}")
    manifest = db.get(DatasetManifest, run.dataset_id)
    if manifest is None:
        raise LookupError(f"Dataset not found: {run.dataset_id}")
    try:
        run.status = "running"
        run.started_at = datetime.now(UTC)
        db.commit()
        from app.db.models import ModelVersion

        model_record = db.get(ModelVersion, run.model_version_id)
        if model_record is None:
            raise LookupError(f"Model not found: {run.model_version_id}")
        bundle = load_model_bundle(model_record)
        frame = load_dataset(manifest)
        metrics, trades = run_backtest(bundle, frame, run.configuration)
        run.status = "completed"
        run.metrics = metrics
        run.trades = trades[:5000]
        run.completed_at = datetime.now(UTC)
        db.commit()
        db.refresh(run)
        return run
    except Exception as exc:
        db.rollback()
        run = db.get(BacktestRun, run_id)
        if run:
            run.status = "failed"
            run.error_message = str(exc)
            run.completed_at = datetime.now(UTC)
            db.commit()
        raise
