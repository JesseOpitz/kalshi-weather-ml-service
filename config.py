from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, get_db, require_admin_key, require_api_key
from app.core.config import get_settings
from app.db.models import BacktestRun, DatasetManifest, ModelVersion, TrainingJob
from app.ml.backtest import run_backtest
from app.schemas.backtest import BacktestRequest, BacktestRunRead
from app.schemas.model import (
    ModelHealthResponse,
    ModelPromotionRequest,
    ModelRollbackRequest,
    ModelVersionRead,
)
from app.schemas.outcome import OutcomeCreate, OutcomeRead
from app.schemas.training import (
    ModelComparisonRequest,
    ModelComparisonResponse,
    TrainCandidateRequest,
    TrainingJobRead,
)
from app.services.audit_service import record_audit
from app.services.backtest_service import execute_backtest
from app.services.dataset_service import load_dataset
from app.services.model_service import (
    get_model_record,
    load_model_bundle,
    model_health,
    promote_model,
)
from app.services.outcome_service import record_outcome
from app.services.training_service import execute_training_job
from app.workers.tasks import run_backtest_task, train_candidate_task

router = APIRouter(tags=["lovable-compatibility"])


def _model_read(item: ModelVersion) -> ModelVersionRead:
    return ModelVersionRead(
        id=item.id,
        model_name=item.model_name,
        version=item.version,
        status=item.status,
        artifact_uri=item.artifact_uri,
        artifact_sha256=item.artifact_sha256,
        feature_schema_version=item.feature_schema_version,
        training_dataset_id=item.training_dataset_id,
        training_start_date=item.training_start_date,
        training_end_date=item.training_end_date,
        metrics=item.metrics,
        hyperparameters=item.hyperparameters,
        metadata=item.metadata_json,
        promoted_at=item.promoted_at,
        retired_at=item.retired_at,
        created_at=item.created_at,
    )


def _run_training_local(job_id: str) -> None:
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        execute_training_job(db, job_id)


def _run_backtest_local(run_id: str) -> None:
    from app.db.session import SessionLocal

    with SessionLocal() as db:
        execute_backtest(db, run_id)


@router.get("/model-health", response_model=ModelHealthResponse)
def compatible_model_health(
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> ModelHealthResponse:
    return ModelHealthResponse.model_validate(model_health(db))


@router.get("/model-versions", response_model=list[ModelVersionRead])
def compatible_model_versions(
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> list[ModelVersionRead]:
    records = db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()
    return [_model_read(record) for record in records]


@router.post("/train-candidate", response_model=TrainingJobRead, status_code=status.HTTP_202_ACCEPTED)
def compatible_train_candidate(
    payload: TrainCandidateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    principal: ApiPrincipal = Depends(require_admin_key),
) -> TrainingJobRead:
    if db.get(DatasetManifest, payload.dataset_id) is None:
        raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND", "message": payload.dataset_id})
    job = TrainingJob(
        dataset_id=payload.dataset_id,
        model_name=payload.model_name,
        requested_version=payload.requested_version,
        configuration=payload.model_dump(exclude={"dataset_id", "model_name", "requested_version"}),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    if get_settings().task_always_eager:
        background_tasks.add_task(_run_training_local, job.id)
    else:
        train_candidate_task.delay(job.id)
    record_audit(
        db,
        actor_fingerprint=principal.key_fingerprint,
        action="training.requested",
        entity_type="training_job",
        entity_id=job.id,
        details={"compatibility_route": True},
    )
    return TrainingJobRead.model_validate(job)


@router.post("/promote-model", response_model=ModelVersionRead)
def compatible_promote_model(
    payload: ModelPromotionRequest,
    db: Session = Depends(get_db),
    principal: ApiPrincipal = Depends(require_admin_key),
) -> ModelVersionRead:
    item = db.query(ModelVersion).filter(ModelVersion.version == payload.model_version).one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": payload.model_version})
    try:
        load_model_bundle(item)
    except Exception as exc:
        raise HTTPException(status_code=422, detail={"code": "MODEL_ARTIFACT_INVALID", "message": str(exc)}) from exc
    promote_model(db, item)
    record_audit(
        db,
        actor_fingerprint=principal.key_fingerprint,
        action="model.promoted",
        entity_type="model_version",
        entity_id=item.id,
        details={"reason": payload.reason, "compatibility_route": True},
    )
    return _model_read(item)


@router.post("/rollback-model", response_model=ModelVersionRead)
def compatible_rollback_model(
    payload: ModelRollbackRequest,
    db: Session = Depends(get_db),
    principal: ApiPrincipal = Depends(require_admin_key),
) -> ModelVersionRead:
    item = db.query(ModelVersion).filter(ModelVersion.version == payload.model_version).one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": payload.model_version})
    promote_model(db, item)
    record_audit(
        db,
        actor_fingerprint=principal.key_fingerprint,
        action="model.rolled_back",
        entity_type="model_version",
        entity_id=item.id,
        details={"reason": payload.reason, "compatibility_route": True},
    )
    return _model_read(item)


@router.post("/backtest-strategy", response_model=BacktestRunRead, status_code=status.HTTP_202_ACCEPTED)
def compatible_backtest_strategy(
    payload: BacktestRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> BacktestRunRead:
    if db.get(DatasetManifest, payload.dataset_id) is None:
        raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND", "message": payload.dataset_id})
    try:
        model = get_model_record(db, payload.model_version)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": str(exc)}) from exc
    run = BacktestRun(
        dataset_id=payload.dataset_id,
        model_version_id=model.id,
        configuration=payload.model_dump(exclude={"dataset_id", "model_version"}),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    if get_settings().task_always_eager:
        background_tasks.add_task(_run_backtest_local, run.id)
    else:
        run_backtest_task.delay(run.id)
    return BacktestRunRead.model_validate(run)


@router.post("/record-outcome", response_model=OutcomeRead)
def compatible_record_outcome(
    payload: OutcomeCreate,
    db: Session = Depends(get_db),
    principal: ApiPrincipal = Depends(require_admin_key),
) -> OutcomeRead:
    outcome, evaluations = record_outcome(db, payload)
    record_audit(
        db,
        actor_fingerprint=principal.key_fingerprint,
        action="outcome.recorded",
        entity_type="outcome",
        entity_id=outcome.id,
        details={"evaluations_created": evaluations, "compatibility_route": True},
    )
    return OutcomeRead(
        id=outcome.id,
        market_id=outcome.market_id,
        station_code=outcome.station_code,
        official_value=outcome.official_value,
        official_source=outcome.official_source,
        settled_at=outcome.settled_at,
        metadata=outcome.metadata_json,
        created_at=outcome.created_at,
        evaluations_created=evaluations,
    )


@router.post("/evaluate-candidate", response_model=ModelComparisonResponse)
def compatible_evaluate_candidate(
    payload: ModelComparisonRequest,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_admin_key),
) -> ModelComparisonResponse:
    champion = db.query(ModelVersion).filter(ModelVersion.version == payload.champion_version).one_or_none()
    challenger = db.query(ModelVersion).filter(ModelVersion.version == payload.challenger_version).one_or_none()
    manifest = db.get(DatasetManifest, payload.dataset_id)
    if champion is None or challenger is None:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": "Champion or challenger model was not found."})
    if manifest is None:
        raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND", "message": payload.dataset_id})
    frame = load_dataset(manifest)
    evaluation_config = {
        "minimum_net_edge": 0.03,
        "fee_multiplier": 0.07,
        "slippage_per_contract": 0.01,
        "uncertainty_buffer": 0.01,
        "contracts_per_trade": 1,
        "maximum_total_exposure": 1000000,
    }
    champion_metrics, _ = run_backtest(load_model_bundle(champion), frame, evaluation_config)
    challenger_metrics, _ = run_backtest(load_model_bundle(challenger), frame, evaluation_config)
    mae_delta = float(challenger_metrics["mae"] - champion_metrics["mae"])
    champion_brier = champion_metrics.get("brier_score")
    challenger_brier = challenger_metrics.get("brier_score")
    brier_delta = 0.0
    if isinstance(champion_brier, (int, float)) and isinstance(challenger_brier, (int, float)):
        brier_delta = float(challenger_brier - champion_brier)
    pnl_delta = float(challenger_metrics.get("total_pnl", 0.0) - champion_metrics.get("total_pnl", 0.0))
    reasons: list[str] = []
    pass_mae = mae_delta <= payload.maximum_mae_degradation
    pass_brier = brier_delta <= -payload.minimum_brier_improvement
    if pass_mae:
        reasons.append("Challenger MAE is within the configured tolerance.")
    else:
        reasons.append("Challenger MAE exceeds the configured degradation tolerance.")
    if pass_brier:
        reasons.append("Challenger Brier score meets the configured improvement threshold.")
    else:
        reasons.append("Challenger Brier score does not meet the configured improvement threshold.")
    recommendation = "promote" if pass_mae and pass_brier and pnl_delta >= 0 else "continue_shadow_evaluation"
    return ModelComparisonResponse(
        champion_version=champion.version,
        challenger_version=challenger.version,
        dataset_id=manifest.id,
        champion_metrics=champion_metrics,
        challenger_metrics=challenger_metrics,
        metric_deltas={"mae": mae_delta, "brier_score": brier_delta, "total_pnl": pnl_delta},
        recommendation=recommendation,
        reasons=reasons,
    )


@router.get("/training-jobs/{job_id}", response_model=TrainingJobRead)
def compatible_training_job(
    job_id: str,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_admin_key),
) -> TrainingJobRead:
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail={"code": "TRAINING_JOB_NOT_FOUND", "message": job_id})
    return TrainingJobRead.model_validate(job)


@router.get("/backtest-runs/{run_id}", response_model=BacktestRunRead)
def compatible_backtest_run(
    run_id: str,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> BacktestRunRead:
    run = db.get(BacktestRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "BACKTEST_NOT_FOUND", "message": run_id})
    return BacktestRunRead.model_validate(run)
