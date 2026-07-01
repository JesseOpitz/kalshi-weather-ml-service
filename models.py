from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.deps import ApiPrincipal, get_db, require_admin_key, require_api_key
from app.db.models import Station
from app.schemas.station import StationCreate, StationRead
from app.services.audit_service import record_audit

router = APIRouter(prefix="/stations", tags=["stations"])


def _read(station: Station) -> StationRead:
    return StationRead(
        id=station.id,
        station_code=station.station_code,
        station_name=station.station_name,
        latitude=station.latitude,
        longitude=station.longitude,
        elevation_m=station.elevation_m,
        timezone=station.timezone,
        official_source=station.official_source,
        metadata=station.metadata_json,
        active=station.active,
        created_at=station.created_at,
        updated_at=station.updated_at,
    )


@router.get("", response_model=list[StationRead])
def list_stations(
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> list[StationRead]:
    return [_read(item) for item in db.query(Station).order_by(Station.station_code).all()]


@router.post("", response_model=StationRead, status_code=status.HTTP_201_CREATED)
def upsert_station(
    payload: StationCreate,
    db: Session = Depends(get_db),
    principal: ApiPrincipal = Depends(require_admin_key),
) -> StationRead:
    station = db.query(Station).filter(Station.station_code == payload.station_code).one_or_none()
    created = station is None
    if station is None:
        station = Station(station_code=payload.station_code, station_name=payload.station_name, latitude=payload.latitude, longitude=payload.longitude)
        db.add(station)
    station.station_name = payload.station_name
    station.latitude = payload.latitude
    station.longitude = payload.longitude
    station.elevation_m = payload.elevation_m
    station.timezone = payload.timezone
    station.official_source = payload.official_source
    station.metadata_json = payload.metadata
    station.active = payload.active
    db.commit()
    db.refresh(station)
    record_audit(
        db,
        actor_fingerprint=principal.key_fingerprint,
        action="station.created" if created else "station.updated",
        entity_type="station",
        entity_id=station.id,
        details={"station_code": station.station_code},
    )
    return _read(station)


@router.get("/{station_code}", response_model=StationRead)
def get_station(
    station_code: str,
    db: Session = Depends(get_db),
    _: ApiPrincipal = Depends(require_api_key),
) -> StationRead:
    station = db.query(Station).filter(Station.station_code == station_code).one_or_none()
    if station is None:
        raise HTTPException(status_code=404, detail={"code": "STATION_NOT_FOUND", "message": station_code})
    return _read(station)
