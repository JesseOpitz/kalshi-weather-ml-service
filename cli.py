from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.db.models import DatasetManifest, ModelVersion, TrainingJob
from app.ml.artifact_store import get_artifact_store
from app.ml.train import train_model_bundle
from app.services.dataset_service import load_dataset


def _append_log(job: TrainingJob, message: str, **details) -> None:
    logs = list(job.logs or [])
    logs.append({"time": datetime.now(UTC).isoformat(), "message": message, **details})
    job.logs = logs


def execute_training_job(db: Session, job_id: str) -> ModelVersion:
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise LookupError(f"Training job not found: {job_id}")
    manifest = db.get(DatasetManifest, job.dataset_id)
    if manifest is None:
        raise LookupError(f"Dataset not found: {job.dataset_id}")
    try:
        job.status = "running"
        job.started_at = datetime.now(UTC)
        job.progress = 0.05
        _append_log(job, "Loading and verifying training dataset.")
        db.commit()

        frame = load_dataset(manifest)
        job.progress = 0.15
        _append_log(job, "Training time-aware ensemble and probability calibrator.", rows=len(frame))
        db.commit()

        version = job.requested_version or (
            f"{job.model_name}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        )
        result = train_model_bundle(frame, version=version, config=job.configuration)
        job.progress = 0.85
        _append_log(job, "Saving model artifact and registering candidate.", version=version)
        db.commit()

        artifact_uri, artifact_hash = get_artifact_store().save_bundle(result.bundle)
        model = ModelVersion(
            model_name=job.model_name,
            version=version,
            status="candidate",
            artifact_uri=artifact_uri,
            artifact_sha256=artifact_hash,
            feature_schema_version=result.bundle.feature_schema_version,
            training_dataset_id=manifest.id,
            training_start_date=result.date_start.date(),
            training_end_date=result.date_end.date(),
            metrics=result.metrics,
            hyperparameters=job.configuration,
            metadata_json=result.bundle.metadata,
        )
        db.add(model)
        db.flush()
        job.status = "completed"
        job.result_model_version_id = model.id
        job.progress = 1.0
        job.completed_at = datetime.now(UTC)
        _append_log(job, "Training completed successfully.", model_version_id=model.id)
        db.commit()
        db.refresh(model)
        return model
    except Exception as exc:
        db.rollback()
        job = db.get(TrainingJob, job_id)
        if job:
            job.status = "failed"
            job.error_message = str(exc)
            job.completed_at = datetime.now(UTC)
            _append_log(job, "Training failed.", error=str(exc))
            db.commit()
        raise
