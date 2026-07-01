from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO

import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import DatasetManifest, Station
from app.ml.features import TARGET_COLUMN, TIME_COLUMN, normalize_training_frame
from app.services.open_meteo import OpenMeteoClient


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_directory() -> Path:
    path = Path("./data/datasets").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    normalized = normalize_training_frame(frame)
    missing_columns = [column for column in (TARGET_COLUMN, TIME_COLUMN, "forecast_mean") if column not in normalized.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if normalized.empty:
        raise ValueError("Dataset contains no valid training rows after normalization.")
    quality = {
        "valid_rows": int(len(normalized)),
        "input_rows": int(len(frame)),
        "dropped_rows": int(len(frame) - len(normalized)),
        "station_count": int(normalized["station_code"].nunique()),
        "date_start": normalized[TIME_COLUMN].min().isoformat(),
        "date_end": normalized[TIME_COLUMN].max().isoformat(),
        "missing_rate": {
            column: round(float(normalized[column].isna().mean()), 4)
            for column in normalized.columns
            if normalized[column].isna().any()
        },
    }
    return normalized, quality


def save_uploaded_dataset(
    db: Session,
    *,
    name: str,
    filename: str,
    file_object: BinaryIO,
    station_code: str | None,
) -> DatasetManifest:
    extension = Path(filename).suffix.lower()
    if extension not in {".csv", ".parquet"}:
        raise ValueError("Only CSV and Parquet datasets are supported.")
    raw_path = _dataset_directory() / f"upload-{hashlib.sha1(name.encode()).hexdigest()[:12]}{extension}"
    with raw_path.open("wb") as output:
        output.write(file_object.read())
    frame = pd.read_csv(raw_path) if extension == ".csv" else pd.read_parquet(raw_path)
    normalized, quality = validate_dataset(frame)
    normalized_path = raw_path.with_suffix(".normalized.csv")
    normalized.to_csv(normalized_path, index=False)
    raw_path.unlink(missing_ok=True)
    manifest = DatasetManifest(
        name=name,
        source_type="upload",
        station_code=station_code,
        file_uri=str(normalized_path),
        checksum_sha256=_sha256(normalized_path),
        row_count=len(normalized),
        date_start=normalized[TIME_COLUMN].min().date(),
        date_end=normalized[TIME_COLUMN].max().date(),
        quality_report=quality,
    )
    db.add(manifest)
    db.commit()
    db.refresh(manifest)
    return manifest


def build_open_meteo_dataset(
    db: Session,
    *,
    name: str,
    station_code: str,
    start_date,
    end_date,
    market_type: str,
    assumed_lead_hours: int,
) -> DatasetManifest:
    station = db.query(Station).filter(Station.station_code == station_code).one_or_none()
    if station is None:
        raise ValueError(f"Unknown station: {station_code}")
    client = OpenMeteoClient()
    frame = client.build_training_dataset(
        station_code=station_code,
        latitude=station.latitude,
        longitude=station.longitude,
        elevation_m=station.elevation_m,
        start_date=start_date,
        end_date=end_date,
        market_type=market_type,
        assumed_lead_hours=assumed_lead_hours,
    )
    normalized, quality = validate_dataset(frame)
    quality["warning"] = (
        "This bootstrap dataset uses gridded reanalysis labels and a seamless historical forecast "
        "series. Replace or augment labels with exact official settlement-station observations for "
        "production-grade station calibration."
    )
    filename = f"open-meteo-{station_code}-{start_date}-{end_date}.csv"
    path = _dataset_directory() / filename
    normalized.to_csv(path, index=False)
    manifest = DatasetManifest(
        name=name,
        source_type="open_meteo_bootstrap",
        station_code=station_code,
        file_uri=str(path),
        checksum_sha256=_sha256(path),
        row_count=len(normalized),
        date_start=normalized[TIME_COLUMN].min().date(),
        date_end=normalized[TIME_COLUMN].max().date(),
        quality_report=quality,
    )
    db.add(manifest)
    db.commit()
    db.refresh(manifest)
    return manifest


def load_dataset(manifest: DatasetManifest) -> pd.DataFrame:
    path = Path(manifest.file_uri)
    if not path.exists():
        raise FileNotFoundError(f"Dataset artifact not found: {path}")
    actual_hash = _sha256(path)
    if actual_hash != manifest.checksum_sha256:
        raise ValueError("Dataset checksum verification failed.")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
