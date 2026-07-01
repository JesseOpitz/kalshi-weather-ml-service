from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, get_db, require_admin_key, require_api_key
from app.db.models import DatasetManifest
from app.schemas.dataset import DatasetRead, OpenMeteoDatasetRequest
from app.services.audit_service import record_audit
from app.services.dataset_service import build_open_meteo_dataset, save_uploaded_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _read(item: DatasetManifest) -> DatasetRead:
    return DatasetRead(
        id=item.id,
        name=item.name,
        source_type=item.source_type,
        station_code=item.station_code,
        file_uri=item.file_uri,
        checksum_sha256=item.checksum_sha256,
        row_count=item.row_count,
        schema_version=item.schema_version,
        date_start=item.date_start,
        date_end=item.date_end,
        quality_report=item.quality_report,
        created_at=item.created_at,
    )


@router.get("", response_model=list[DatasetRead])
def list_datasets(
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> list[DatasetRead]:
    items = db.query(DatasetManifest).order_by(DatasetManifest.created_at.desc()).all()
    return [_read(item) for item in items]


@router.get("/{dataset_id}", response_model=DatasetRead)
def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> DatasetRead:
    item = db.get(DatasetManifest, dataset_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"code": "DATASET_NOT_FOUND", "message": dataset_id})
    return _read(item)


@router.post("/upload", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
def upload_dataset(
    name: str = Form(...),
    station_code: str | None = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    principal: ApiPrincipal = Depends(require_admin_key),
) -> DatasetRead:
    try:
        manifest = save_uploaded_dataset(
            db,
            name=name,
            filename=file.filename or "dataset.csv",
            file_object=file.file,
            station_code=station_code,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"code": "INVALID_DATASET", "message": str(exc)}) from exc
    record_audit(
        db,
        actor_fingerprint=principal.key_fingerprint,
        action="dataset.uploaded",
        entity_type="dataset",
        entity_id=manifest.id,
        details={"row_count": manifest.row_count},
    )
    return _read(manifest)


@router.post("/open-meteo", response_model=DatasetRead, status_code=status.HTTP_201_CREATED)
def create_open_meteo_dataset(
    payload: OpenMeteoDatasetRequest,
    db: Session = Depends(get_db),
    principal: ApiPrincipal = Depends(require_admin_key),
) -> DatasetRead:
    try:
        manifest = build_open_meteo_dataset(db, **payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=502, detail={"code": "DATA_INGESTION_FAILED", "message": str(exc)}) from exc
    record_audit(
        db,
        actor_fingerprint=principal.key_fingerprint,
        action="dataset.open_meteo_created",
        entity_type="dataset",
        entity_id=manifest.id,
        details={"row_count": manifest.row_count, "station_code": payload.station_code},
    )
    return _read(manifest)
