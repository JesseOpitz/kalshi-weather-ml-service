from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import ModelVersion
from app.ml.artifact_store import get_artifact_store
from app.ml.bundle import ModelBundle

_cache_lock = threading.Lock()
_cache: dict[str, tuple[float, ModelBundle]] = {}


def get_model_record(db: Session, requested: str = "champion") -> ModelVersion:
    query = db.query(ModelVersion)
    if requested == "champion":
        record = query.filter(ModelVersion.status == "champion").order_by(ModelVersion.promoted_at.desc()).first()
    else:
        record = query.filter(ModelVersion.version == requested).one_or_none()
    if record is None:
        raise LookupError(f"Model '{requested}' is not available.")
    return record


def load_model_bundle(record: ModelVersion) -> ModelBundle:
    settings = get_settings()
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(record.id)
        if cached and now - cached[0] < settings.active_model_cache_seconds:
            return cached[1]
    bundle = get_artifact_store().load_bundle(record.artifact_uri, record.artifact_sha256)
    with _cache_lock:
        _cache.clear() if record.status == "champion" else None
        _cache[record.id] = (now, bundle)
    return bundle


def invalidate_model_cache() -> None:
    with _cache_lock:
        _cache.clear()


def promote_model(db: Session, record: ModelVersion) -> ModelVersion:
    now = datetime.now(UTC)
    existing = db.query(ModelVersion).filter(ModelVersion.status == "champion").all()
    for champion in existing:
        if champion.id != record.id:
            champion.status = "retired"
            champion.retired_at = now
    record.status = "champion"
    record.promoted_at = now
    record.retired_at = None
    db.commit()
    db.refresh(record)
    invalidate_model_cache()
    return record


def model_health(db: Session) -> dict[str, Any]:
    try:
        record = get_model_record(db, "champion")
    except LookupError:
        return {
            "service_status": "healthy",
            "model_status": "unavailable",
            "model_version": None,
            "model_id": None,
            "feature_schema_version": None,
            "artifact_verified": False,
            "supported_stations": [],
            "supported_market_types": [],
            "metrics": {},
            "warnings": ["No champion model has been promoted. Predictions fail closed."],
            "timestamp": datetime.now(UTC),
        }
    warnings: list[str] = []
    verified = False
    bundle: ModelBundle | None = None
    try:
        bundle = load_model_bundle(record)
        verified = True
    except Exception as exc:
        warnings.append(f"Champion artifact could not be loaded: {exc}")
    return {
        "service_status": "healthy" if verified else "degraded",
        "model_status": record.status if verified else "artifact_error",
        "model_version": record.version,
        "model_id": record.id,
        "feature_schema_version": record.feature_schema_version,
        "artifact_verified": verified,
        "supported_stations": bundle.supported_stations if bundle else [],
        "supported_market_types": bundle.supported_market_types if bundle else [],
        "metrics": record.metrics,
        "warnings": warnings,
        "timestamp": datetime.now(UTC),
    }
