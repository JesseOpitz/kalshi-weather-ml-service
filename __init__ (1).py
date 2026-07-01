from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, get_db, require_admin_key, require_api_key
from app.db.models import ModelVersion
from app.schemas.model import (
    ModelHealthResponse,
    ModelPromotionRequest,
    ModelRollbackRequest,
    ModelVersionRead,
)
from app.services.audit_service import record_audit
from app.services.model_service import load_model_bundle, model_health, promote_model

router = APIRouter(prefix="/models", tags=["models"])


def _read(item: ModelVersion) -> ModelVersionRead:
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


@router.get("/health", response_model=ModelHealthResponse)
def get_model_health(
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> ModelHealthResponse:
    return ModelHealthResponse.model_validate(model_health(db))


@router.get("", response_model=list[ModelVersionRead])
def list_models(
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> list[ModelVersionRead]:
    return [_read(item) for item in db.query(ModelVersion).order_by(ModelVersion.created_at.desc()).all()]


@router.get("/{version}", response_model=ModelVersionRead)
def get_model(
    version: str,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> ModelVersionRead:
    item = db.query(ModelVersion).filter(ModelVersion.version == version).one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": version})
    return _read(item)


@router.post("/promote", response_model=ModelVersionRead)
def promote(
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
    item = promote_model(db, item)
    record_audit(
        db,
        actor_fingerprint=principal.key_fingerprint,
        action="model.promoted",
        entity_type="model_version",
        entity_id=item.id,
        details={"version": item.version, "reason": payload.reason},
    )
    return _read(item)


@router.post("/rollback", response_model=ModelVersionRead)
def rollback(
    payload: ModelRollbackRequest,
    db: Session = Depends(get_db),
    principal: ApiPrincipal = Depends(require_admin_key),
) -> ModelVersionRead:
    item = db.query(ModelVersion).filter(ModelVersion.version == payload.model_version).one_or_none()
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "MODEL_NOT_FOUND", "message": payload.model_version})
    item = promote_model(db, item)
    record_audit(
        db,
        actor_fingerprint=principal.key_fingerprint,
        action="model.rolled_back",
        entity_type="model_version",
        entity_id=item.id,
        details={"version": item.version, "reason": payload.reason},
    )
    return _read(item)
