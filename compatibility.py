from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, get_db, require_admin_key
from app.core.config import get_settings
from app.db.models import DatasetManifest, TrainingJob
from app.schemas.training import TrainCandidateRequest, TrainingJobRead
from app.services.audit_service import record_audit
from app.services.training_service import execute_training_job
from app.workers.tasks import train_candidate_task

router = APIRouter(prefix="/training", tags=["training"])


def _read(job: TrainingJob) -> TrainingJobRead:
    return TrainingJobRead.model_validate(job)


def _execute_local(job_id: str) -> None:
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        execute_training_job(db, job_id)


@router.post("/train-candidate", response_model=TrainingJobRead, status_code=status.HTTP_202_ACCEPTED)
def train_candidate(
    payload: TrainCandidateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    principal: ApiPrincipal = Depends(require_admin_key),
) -> TrainingJobRead:
    if db.get(DatasetManifest, payload.dataset_id) is None:
        raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND", "message": payload.dataset_id})
    configuration = payload.model_dump(exclude={"dataset_id", "model_name", "requested_version"})
    job = TrainingJob(
        dataset_id=payload.dataset_id,
        model_name=payload.model_name,
        requested_version=payload.requested_version,
        configuration=configuration,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    settings = get_settings()
    if settings.task_always_eager:
        background_tasks.add_task(_execute_local, job.id)
    else:
        train_candidate_task.delay(job.id)
    record_audit(
        db,
        actor_fingerprint=principal.key_fingerprint,
        action="training.requested",
        entity_type="training_job",
        entity_id=job.id,
        details={"dataset_id": payload.dataset_id},
    )
    return _read(job)


@router.get("/{job_id}", response_model=TrainingJobRead)
def get_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_admin_key),
) -> TrainingJobRead:
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"code": "TRAINING_JOB_NOT_FOUND", "message": job_id})
    return _read(job)
