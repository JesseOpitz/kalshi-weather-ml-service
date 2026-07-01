from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, get_db, require_admin_key
from app.schemas.outcome import OutcomeCreate, OutcomeRead
from app.services.audit_service import record_audit
from app.services.outcome_service import record_outcome

router = APIRouter(prefix="/outcomes", tags=["outcomes"])


@router.post("", response_model=OutcomeRead)
def create_outcome(
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
        details={"market_id": payload.market_id, "evaluations_created": evaluations},
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
